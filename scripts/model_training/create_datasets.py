"""Create train, validation, and test datasets of soft labels from label distribution runs.

The inputs are the JSONL files written by
``scripts/data_processing/generate_label_distributions.py``: one record per document
with its text, the URL shown in the prompt, and a probability per label
(``label_probs``, keyed by label short name). Each record becomes one row with the
document ID, its language, its URL, its text, and a ``labels`` vector of probabilities
in label-file order, the training target of ``finetune.py``. The language is the file
name up to the first hyphen (``finnish.jsonl`` and ``finnish-parallel.jsonl`` are both
``finnish``).

The URL is stored unchanged, or as null when the labelling run had none for the
document. ``finetune.py`` then uses WebOrganizer's ``{url}\\n\\n{text}`` input by
default. Records written before URLs were stored (no ``url`` key at all) raise an
error; ``backfill_result_urls.py`` adds them. With ``--no_url`` the dataset has no
``url`` column, the records' URLs are ignored (and may be absent), and ``finetune.py``
trains on the text alone, like WebOrganizer's ``-NoURL`` classifiers.

Splits are made by document ID, not by row: the parallel (translated) files hold the
same documents under the same ID in every language, and a random row split would put
translations of a test document into the training set. Every ID is assigned to a split
by a seeded hash, so all language versions of a document land in the same split, and
the assignment doesn't depend on file order or on which files are included. For native
data such as the HPLT samples, whose documents differ between languages, the hash is
simply a reproducible random split of each language.

Rows are stored in input order: one block per language (files in name order), each in
file order. Training doesn't depend on that order: ``finetune.py``'s length-grouped
sampler draws a new random permutation of the whole training split every epoch, and
``--max_train_samples`` takes a seeded random subset.

The input is streamed into memory-mapped Arrow files, so multi-gigabyte inputs don't
have to fit in memory. Malformed records raise an error instead of being skipped.

Outputs (in ``--output_dir``): a ``datasets.DatasetDict`` saved with
``save_to_disk`` (splits ``train``, ``validation``, ``test``), ``labels.json`` with
the label names in target order, ``README.md`` describing how the data was made (split
sizes, documents per language, and the label distribution of every split), and
``label_counts.tsv`` with the label distribution per split and language.

In the label statistics, a document's top label is its highest-probability label (the
first in label order on a tie, as in ``finetune.py``'s accuracy). Its soft count adds
up its probabilities, so a label's soft share is its share of the training targets.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import logging
import math
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import datasets
import numpy as np
import yaml  # type: ignore

LOGGER = logging.getLogger(__name__)

SPLITS = ("train", "validation", "test")
DEFAULT_LABELS_DIR = Path(__file__).resolve().parents[1] / "data_processing"
# The distribution pipeline accepts probabilities that sum to 1 within this tolerance.
PROBABILITY_SUM_TOLERANCE = 1e-3


def load_label_names(labels_path: Path) -> tuple[list[str], list[str]]:
    """Return the label short names and full names in label-file order.

    The short names are the keys of ``label_probs`` in the distribution records; their
    order here fixes the order of the ``labels`` vector.

    Raises:
        ValueError: If the file is not a list of labels with unique names.
    """
    raw = yaml.safe_load(labels_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"Expected a non-empty YAML list of labels in {labels_path}")
    short_names, full_names = [], []
    for item in raw:
        if not isinstance(item, dict) or not item.get("short_name") or not item.get("name"):
            raise ValueError(f"Every label in {labels_path} needs a name and a short_name: {item!r}")
        short_names.append(str(item["short_name"]))
        full_names.append(str(item["name"]))
    if len(set(short_names)) != len(short_names):
        raise ValueError(f"Duplicate label short names in {labels_path}")
    return short_names, full_names


def find_input_files(inputs: Sequence[Path]) -> list[Path]:
    """Expand input paths into a sorted list of JSONL files.

    Args:
        inputs: JSONL files, or directories whose ``*.jsonl`` files are all used.

    Raises:
        FileNotFoundError: If a path doesn't exist or a directory has no JSONL files.
        ValueError: If the same file is given twice.
    """
    files: list[Path] = []
    for path in inputs:
        if path.is_dir():
            found = sorted(path.glob("*.jsonl"))
            if not found:
                raise FileNotFoundError(f"No JSONL files found in directory {path}")
            files.extend(found)
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"Input path {path} does not exist")
    resolved = [file.resolve() for file in files]
    if len(set(resolved)) != len(resolved):
        raise ValueError("The same input file is given more than once")
    return files


def language_of(path: Path) -> str:
    """Return the language encoded in a result file name, e.g. ``finnish-parallel.jsonl``."""
    return path.stem.split("-", 1)[0]


def split_of(doc_id: str, seed: int, train_ratio: float, val_ratio: float) -> str:
    """Assign a document ID to a split with a seeded hash.

    The hash maps the ID to a number in [0, 1); the split ratios cut that interval, so
    the realized split sizes are close to, but not exactly, the requested ratios.
    """
    digest = hashlib.sha256(f"{seed}:{doc_id}".encode("utf-8")).digest()
    position = int.from_bytes(digest[:8], "big") / 2**64
    if position < train_ratio:
        return "train"
    if position < train_ratio + val_ratio:
        return "validation"
    return "test"


def label_vector(record: dict[str, Any], label_names: Sequence[str], where: str) -> list[float]:
    """Return a record's label probabilities in label-file order, validated.

    Raises:
        ValueError: If labels are missing or unexpected, or the values are not a
            probability distribution.
    """
    probs = record.get("label_probs")
    if not isinstance(probs, dict):
        raise ValueError(f"{where}: record without a label_probs mapping")
    if set(probs) != set(label_names):
        missing = sorted(set(label_names) - set(probs))
        unexpected = sorted(set(probs) - set(label_names))
        raise ValueError(
            f"{where}: label_probs don't match the label file (missing {missing}, unexpected {unexpected}); "
            "check --label_type / --labels_file"
        )
    vector = []
    for name in label_names:
        value = probs[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{where}: invalid probability {value!r} for label {name!r}")
        vector.append(float(value))
    total = math.fsum(vector)
    if abs(total - 1.0) > PROBABILITY_SUM_TOLERANCE:
        raise ValueError(f"{where}: label probabilities sum to {total}, not 1")
    return vector


def record_url(record: dict[str, Any], where: str) -> str | None:
    """Return a record's URL, or None if the labelling run had no URL for the document.

    Raises:
        ValueError: If the record has no ``url`` key (written before URLs were stored)
            or the URL is not a non-empty string or null.
    """
    if "url" not in record:
        raise ValueError(
            f"{where}: record without a 'url' field; it predates URL storage, "
            "add the URLs with scripts/data_processing/backfill_result_urls.py"
        )
    url = record["url"]
    if url is None:
        return None
    if not isinstance(url, str) or not url.strip():
        raise ValueError(f"{where}: invalid URL {url!r}; expected a non-empty string or null")
    return url


def generate_rows(
    files: Sequence[str],
    label_names: Sequence[str],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    include_url: bool,
) -> Iterator[dict[str, Any]]:
    """Stream validated rows, each tagged with its split, from the input files.

    Args:
        files: Input JSONL paths as strings (``datasets`` hashes generator arguments).
        label_names: Label short names in target order.
        seed: Seed of the split hash.
        train_ratio: Share of document IDs assigned to ``train``.
        val_ratio: Share of document IDs assigned to ``validation``.
        include_url: Add a ``url`` column (and require a ``url`` field in every record).

    Raises:
        ValueError: If a record is malformed or a document ID repeats within a language.
    """
    seen: set[tuple[str, str]] = set()
    for file_name in files:
        path = Path(file_name)
        language = language_of(path)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                where = f"{path}:{line_number}"
                if not line.strip():
                    raise ValueError(f"{where}: empty line")
                record = json.loads(line)
                doc_id, text = record.get("id"), record.get("text")
                if not isinstance(doc_id, str) or not doc_id:
                    raise ValueError(f"{where}: record without a document ID")
                if not isinstance(text, str):
                    raise ValueError(f"{where}: record {doc_id!r} has no text field")
                if not text.strip():
                    # Pages without content are labelled (usually Bad example) and kept.
                    LOGGER.warning("%s: document %r has %s; keeping it", where, doc_id,
                                   "empty text" if not text else "only whitespace as text")
                if (language, doc_id) in seen:
                    raise ValueError(f"{where}: duplicate document {doc_id!r} for language {language!r}")
                seen.add((language, doc_id))
                row = {"id": doc_id, "language": language}
                if include_url:
                    row["url"] = record_url(record, where)
                yield {
                    **row,
                    "text": text,
                    "labels": label_vector(record, label_names, where),
                    "split": split_of(doc_id, seed, train_ratio, val_ratio),
                }


@dataclass
class SplitStatistics:
    """Document and label counts of one split, per language.

    Attributes:
        num_labels: Number of labels.
        documents: Documents per language.
        top_label: Per language, the number of documents whose top label is each
            label (an array in label order).
        soft: Per language, the summed probability of each label (an array in label
            order); divided by the documents, it is the label's mean probability.
    """

    num_labels: int
    documents: Counter[str]
    top_label: dict[str, np.ndarray]
    soft: dict[str, np.ndarray]

    @property
    def total(self) -> int:
        """Number of documents in the split."""
        return sum(self.documents.values())

    def top_label_total(self) -> np.ndarray:
        """Top-label counts over all languages, in label order."""
        return np.sum([*self.top_label.values(), np.zeros(self.num_labels, dtype=np.int64)], axis=0)

    def soft_total(self) -> np.ndarray:
        """Summed probabilities over all languages, in label order."""
        return np.sum([*self.soft.values(), np.zeros(self.num_labels)], axis=0)


def split_statistics(dataset: datasets.Dataset, num_labels: int, batch_size: int = 10_000) -> SplitStatistics:
    """Count documents, top labels, and summed label probabilities per language.

    The split is read in batches from its Arrow files, so memory use doesn't grow with
    the dataset. A document's top label is ``argmax`` of its probabilities (the first
    label in label order on a tie, as in ``finetune.py``).

    Raises:
        ValueError: If a label vector doesn't have ``num_labels`` entries.
    """
    documents: Counter[str] = Counter()
    top_label: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(num_labels, dtype=np.int64))
    soft: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(num_labels, dtype=np.float64))
    columns = dataset.select_columns(["language", "labels"]).with_format("numpy")
    for batch in columns.iter(batch_size=batch_size):
        probabilities = np.asarray(batch["labels"], dtype=np.float64)
        if probabilities.ndim != 2 or probabilities.shape[1] != num_labels:
            raise ValueError(f"Expected label vectors of length {num_labels}, got shape {probabilities.shape}")
        languages = np.asarray(batch["language"])
        top = probabilities.argmax(axis=1)
        for language in np.unique(languages):
            mask = languages == language
            documents[str(language)] += int(mask.sum())
            top_label[str(language)] += np.bincount(top[mask], minlength=num_labels)
            soft[str(language)] += probabilities[mask].sum(axis=0)
    return SplitStatistics(num_labels, documents, dict(top_label), dict(soft))


def write_label_counts(path: Path, statistics: dict[str, SplitStatistics], label_names: Sequence[str]) -> None:
    """Write the label distribution per split, language, and label as a TSV file.

    Columns: split, language, label, documents (in that split and language),
    top_label_documents, and probability_sum (the label's soft count).
    """
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["split", "language", "label", "documents", "top_label_documents", "probability_sum"])
        for split, stats in statistics.items():
            for language in sorted(stats.documents):
                for index, label in enumerate(label_names):
                    writer.writerow([split, language, label, stats.documents[language],
                                     int(stats.top_label[language][index]), f"{stats.soft[language][index]:.4f}"])


def make_readme(
    args: argparse.Namespace,
    files: Sequence[Path],
    labels_path: Path,
    label_names: Sequence[str],
    statistics: dict[str, SplitStatistics],
    missing_urls: dict[str, int] | None,
) -> str:
    """Describe how the dataset was made: inputs, labels, URLs, split sizes, and label distribution.

    ``missing_urls`` counts the rows without a URL per split, or is None for a dataset
    made with ``--no_url``.
    """
    totals = {split: statistics[split].total for split in SPLITS}
    all_documents = sum(totals.values())
    ratios = {"train": args.train_ratio, "validation": args.val_ratio, "test": args.test_ratio}
    size_rows = "\n".join(
        f"| {split} | {totals[split]:,} | {totals[split] / all_documents:.1%} | {ratios[split]:.0%} |"
        for split in SPLITS
    )
    size_rows += f"\n| **Total** | **{all_documents:,}** | 100.0% | |"

    languages = sorted({language for stats in statistics.values() for language in stats.documents})
    language_rows = "\n".join(
        f"| {language} | " + " | ".join(f"{statistics[split].documents[language]:,}" for split in SPLITS)
        + f" | {sum(statistics[split].documents[language] for split in SPLITS):,} |"
        for language in languages
    )
    language_rows += "\n| **Total** | " + " | ".join(f"**{totals[split]:,}**" for split in SPLITS) + \
        f" | **{all_documents:,}** |"

    top = {split: statistics[split].top_label_total() for split in SPLITS}
    soft_train = statistics["train"].soft_total()

    def share(count: float, total: int) -> str:
        return f"{count / total:.1%}" if total else "n/a"

    label_rows = "\n".join(
        f"| {label} | "
        + " | ".join(f"{int(top[split][index]):,} | {share(top[split][index], totals[split])}" for split in SPLITS)
        + f" | {share(soft_train[index], totals['train'])} |"
        for index, label in enumerate(label_names)
    )
    label_rows += "\n| **Total** | " + " | ".join(
        f"**{totals[split]:,}** | {share(totals[split], totals[split])}" for split in SPLITS
    ) + f" | {share(soft_train.sum(), totals['train'])} |"

    input_list = "\n".join(f"- {file}" for file in files)
    if missing_urls is None:
        url_text = (
            "The dataset was made with `--no_url`: it has no `url` column, and `finetune.py`\n"
            "trains on the text alone by default."
        )
    else:
        url_text = (
            "The `url` column holds the URL shown to the labelling model, unchanged, or null if\n"
            f"there was none. Rows without a URL: {', '.join(f'{split} {missing_urls[split]}' for split in SPLITS)}.\n"
            "`finetune.py` uses the URL and the text by default; `--template` changes that."
        )
    return f"""# Soft-label classification dataset

Created on {datetime.datetime.now().strftime("%Y-%m-%d")} by `scripts/model_training/create_datasets.py`.

## Input files

{input_list}

## Labels

Label type: {args.label_type}. Labels file: `{labels_path}`.

Each row's `labels` column holds the label probabilities from the distribution run, in
this order (also in `labels.json`): {", ".join(label_names)}.

## URLs

{url_text}

## Splits

Document IDs are assigned to splits by a seeded hash (seed {args.seed}), so all language
versions of a document are in the same split. The realized shares are close to, but
not exactly, the requested ratios.

| Split | Documents | Share | Requested |
|---|---:|---:|---:|
{size_rows}

### Documents per language

| Language | train | validation | test | Total |
|---|---:|---:|---:|---:|
{language_rows}

## Label distribution

For each split, the number and share of documents whose **top label** (highest
probability; the first in label order on a tie) is each label. The last column is the
label's **soft share** in the training split: its mean probability over the training
documents, i.e. its share of the soft targets the classifier is trained on. Where the
two differ, the LLM was uncertain: the soft share is higher for labels it often gives as
a second choice, and lower for labels it picks with low confidence. Counts per language
are in `label_counts.tsv`.

| Label | train | % | validation | % | test | % | train soft share |
|---|---:|---:|---:|---:|---:|---:|---:|
{label_rows}
"""


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Create train, validation, and test datasets of soft labels from label distribution runs."
    )
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="Distribution result JSONL files, or directories of them (e.g. results/distribution/<model>/topic)",
    )
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to save the output datasets.")
    parser.add_argument("--label_type", choices=("topic", "format"), required=True)
    parser.add_argument(
        "--labels_file",
        type=Path,
        default=None,
        help="Label YAML file defining the label order (default: scripts/data_processing/<label_type>s.yaml)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing dataset in the output directory.",
    )
    parser.add_argument(
        "--no_url",
        action="store_true",
        help="Leave the URL out of the dataset (no url column), for a text-only classifier. "
        "Records then needn't have a url field.",
    )
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of training data.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Ratio of validation data.")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Ratio of test data.")
    parser.add_argument("--seed", type=int, default=42, help="Seed of the document-ID split hash.")
    parser.add_argument("--log_level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Check the split ratios and the output directory.

    Raises:
        ValueError: If the ratios are negative or don't sum to 1.
        FileExistsError: If the output directory is not empty and can't be replaced.
    """
    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    if any(ratio < 0 for ratio in ratios):
        raise ValueError("Train, validation, and test ratios must be non-negative")
    # isclose, not ==: e.g. 0.7 + 0.2 + 0.1 is 0.9999999999999999 in floating point.
    if not math.isclose(sum(ratios), 1.0):
        raise ValueError("Train, validation, and test ratios must sum to 1.0")
    output_dir: Path = args.output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory {output_dir} already exists and is not empty. Use --overwrite to overwrite."
            )
        # Only replace a directory this script wrote, never an arbitrary one.
        if not (output_dir / "dataset_dict.json").is_file():
            raise FileExistsError(f"--overwrite refuses to replace {output_dir}: it doesn't hold a saved dataset")


def main(argv: Sequence[str] | None = None) -> int:
    """Build and save the soft-label dataset."""
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s [%(levelname)s] %(message)s")
    validate_args(args)

    labels_path: Path = args.labels_file or DEFAULT_LABELS_DIR / f"{args.label_type}s.yaml"
    label_names, full_names = load_label_names(labels_path)
    files = find_input_files(args.input)
    for file in files:
        LOGGER.info("Input %s (language %s)", file, language_of(file))
    LOGGER.info("%d labels from %s", len(label_names), labels_path)

    include_url = not args.no_url
    LOGGER.info("URL column: %s", "yes" if include_url else "no (--no_url)")
    columns = {"id": datasets.Value("string"), "language": datasets.Value("string")}
    if include_url:
        columns["url"] = datasets.Value("string")  # null when the document has no URL
    features = datasets.Features(
        {
            **columns,
            "text": datasets.Value("string"),
            "labels": datasets.List(datasets.Value("float32"), length=len(label_names)),
            "split": datasets.Value("string"),
        }
    )
    output_dir: Path = args.output_dir
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    # The Arrow cache lives next to the output, not in the home directory, and is
    # removed once the splits are saved.
    with tempfile.TemporaryDirectory(dir=output_dir.parent, prefix=f".{output_dir.name}-build-") as cache_dir:
        rows = datasets.Dataset.from_generator(
            generate_rows,
            features=features,
            cache_dir=cache_dir,
            gen_kwargs={
                "files": [str(file) for file in files],
                # A tuple: datasets shards generators over list arguments, which must be only `files`.
                "label_names": tuple(label_names),
                "seed": args.seed,
                "train_ratio": args.train_ratio,
                "val_ratio": args.val_ratio,
                "include_url": include_url,
            },
        )
        dataset_dict = datasets.DatasetDict(
            {
                split: rows.filter(
                    lambda splits, name=split: [value == name for value in splits],
                    input_columns="split",
                    batched=True,
                    desc=f"Selecting {split}",
                ).remove_columns("split")
                for split in SPLITS
            }
        )
        ratios = {"train": args.train_ratio, "validation": args.val_ratio, "test": args.test_ratio}
        statistics = {split: split_statistics(dataset_dict[split], len(label_names)) for split in SPLITS}
        missing_urls = (
            {split: sum(url is None for url in dataset_dict[split]["url"]) for split in SPLITS}
            if include_url
            else None
        )
        for split in SPLITS:
            LOGGER.info("%s: %d rows %s", split, len(dataset_dict[split]),
                        dict(sorted(statistics[split].documents.items())))
            if missing_urls is not None and missing_urls[split]:
                LOGGER.warning("%s: %d rows have no URL", split, missing_urls[split])
            if ratios[split] > 0 and not len(dataset_dict[split]):
                raise ValueError(f"The {split} split is empty; the input has too few documents")
            if statistics[split].total != len(dataset_dict[split]):
                raise ValueError(f"The {split} statistics count {statistics[split].total} rows, not "
                                 f"{len(dataset_dict[split])}")
        never_top = [name for name, count in zip(label_names, statistics["train"].top_label_total()) if count == 0]
        if never_top:
            LOGGER.warning("Labels that are never the top label in the training split: %s", ", ".join(never_top))

        if output_dir.exists() and args.overwrite:
            shutil.rmtree(output_dir)
        # Save dataset with a README file containing information about the dataset
        dataset_dict.save_to_disk(str(output_dir))

    (output_dir / "labels.json").write_text(
        json.dumps(
            {"label_type": args.label_type, "labels_file": str(labels_path), "label_names": label_names,
             "label_full_names": full_names},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    write_label_counts(output_dir / "label_counts.tsv", statistics, label_names)
    (output_dir / "README.md").write_text(
        make_readme(args, files, labels_path, label_names, statistics, missing_urls), encoding="utf-8"
    )
    LOGGER.info("Saved the dataset to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
