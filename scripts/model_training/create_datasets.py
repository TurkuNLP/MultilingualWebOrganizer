"""Create train, validation, and test datasets of soft labels from label distribution runs.

The inputs are the JSONL files written by
``scripts/data_processing/generate_label_distributions.py``: one record per document
with its text and a probability per label (``label_probs``, keyed by label short name).
Each record becomes one row with the document ID, its language, its text, and a
``labels`` vector of probabilities in label-file order, the training target of
``finetune.py``.

Splits are made by document ID, not by row: the parallel files hold translations of
the same documents under the same ID, and a random row split would put translations
of a test document into the training set. Every ID is assigned to a split by a seeded
hash, so all language versions of a document land in the same split, and the
assignment doesn't depend on file order or on which files are included.

The input is streamed into memory-mapped Arrow files, so multi-gigabyte inputs don't
have to fit in memory. Malformed records raise an error instead of being skipped.

Outputs (in ``--output_dir``): a ``datasets.DatasetDict`` saved with
``save_to_disk`` (splits ``train``, ``validation``, ``test``), ``labels.json`` with
the label names in target order, and ``README.md`` describing how the data was made.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import math
import shutil
import tempfile
from collections import Counter
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import datasets
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


def generate_rows(
    files: Sequence[str],
    label_names: Sequence[str],
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> Iterator[dict[str, Any]]:
    """Stream validated rows, each tagged with its split, from the input files.

    Args:
        files: Input JSONL paths as strings (``datasets`` hashes generator arguments).
        label_names: Label short names in target order.
        seed: Seed of the split hash.
        train_ratio: Share of document IDs assigned to ``train``.
        val_ratio: Share of document IDs assigned to ``validation``.

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
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"{where}: record {doc_id!r} has no text")
                if (language, doc_id) in seen:
                    raise ValueError(f"{where}: duplicate document {doc_id!r} for language {language!r}")
                seen.add((language, doc_id))
                yield {
                    "id": doc_id,
                    "language": language,
                    "text": text,
                    "labels": label_vector(record, label_names, where),
                    "split": split_of(doc_id, seed, train_ratio, val_ratio),
                }


def make_readme(
    args: argparse.Namespace,
    files: Sequence[Path],
    labels_path: Path,
    label_names: Sequence[str],
    counts: dict[str, Counter[str]],
) -> str:
    """Describe how the dataset was made, with row counts per split and language."""
    languages = sorted({language for split_counts in counts.values() for language in split_counts})
    count_rows = "\n".join(
        f"| {split} | " + " | ".join(str(counts[split][language]) for language in languages)
        + f" | {sum(counts[split].values())} |"
        for split in SPLITS
    )
    input_list = "\n".join(f"- {file}" for file in files)
    return f"""# Soft-label classification dataset

Created on {datetime.datetime.now().strftime("%Y-%m-%d")} by `scripts/model_training/create_datasets.py`.

## Input files

{input_list}

## Labels

Label type: {args.label_type}. Labels file: `{labels_path}`.

Each row's `labels` column holds the label probabilities from the distribution run, in
this order (also in `labels.json`): {", ".join(label_names)}.

## Splits

Document IDs are assigned to splits by a seeded hash (seed {args.seed}), so all language
versions of a document are in the same split. Requested ratios: train {args.train_ratio},
validation {args.val_ratio}, test {args.test_ratio}.

| Split | {" | ".join(languages)} | Total |
|---|{"---|" * len(languages)}---|
{count_rows}
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

    features = datasets.Features(
        {
            "id": datasets.Value("string"),
            "language": datasets.Value("string"),
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
        counts = {split: Counter(dataset_dict[split]["language"]) for split in SPLITS}
        for split in SPLITS:
            LOGGER.info("%s: %d rows %s", split, len(dataset_dict[split]), dict(sorted(counts[split].items())))
            if ratios[split] > 0 and not len(dataset_dict[split]):
                raise ValueError(f"The {split} split is empty; the input has too few documents")

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
    (output_dir / "README.md").write_text(make_readme(args, files, labels_path, label_names, counts), encoding="utf-8")
    LOGGER.info("Saved the dataset to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
