"""Create train, validation, and test datasets of soft labels from label distribution runs.

The inputs are the JSONL files written by
``scripts/data_processing/generate_label_distributions.py``: one record per document
with its text, the URL shown in the prompt, and a probability per label
(``label_probs``, keyed by label short name). Each record becomes one row with the
document ID, its language, its source, its URL, its text, and a ``labels`` vector of
probabilities in label-file order, the training target of ``finetune.py``. The language
is the file name up to the first hyphen (``finnish.jsonl`` and ``finnish-parallel.jsonl``
are both ``finnish``).

The source tells which ``--input`` path a row came from, e.g. the HPLT collection, so
results can be reported per source. ``--source_names`` names the inputs, one name per
path. Without it, a source is the name of the input's directory (for a file input, the
directory it is in), or the name of the directory above it if that directory is named
after the label type: ``results/distribution/<run>/clean_random/topic`` is
``clean_random``, and ``results/distribution/<run>/topic`` is ``<run>``.

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

The hash gives each split its requested share only on average, so a small language can
end up with noticeably more or fewer validation or test documents than its share.
``--stratify_by_language`` instead splits every language in exactly the requested
ratios (rounded): within each language, the documents are ordered by the same seeded
hash and cut at the requested counts. It reads the input once more to count the
documents. A document ID present in several languages must still land in the same split
in all of them; if the cuts would separate translations, the script stops (parallel data
with the same documents in every language is split identically either way).

``--max_per_file`` caps the documents taken from each file, with one value for all inputs
or one per ``--input`` path. The first N records of a file are used, and a file with
fewer records is used whole (with a warning). Taking the first N records gives a random
subset only if the file is in random order. The HPLT result files are: their samples are
shuffled, and the labelling run processes a sample in order. For example, to take 25,000
documents per language from clean and noisy and 10,000 from poor:
``--input <run>/clean_random/topic <run>/noisy_random/topic <run>/poor_random/topic
--max_per_file 25000 25000 10000``. Files of the same name in several inputs are the
same language, so a language's rows combine all three collections and are split
together. A document ID repeated across the inputs of a language stops the script.

Rows are stored in input order: the inputs in the order given, the files of a directory
in name order, and each file in file order. Training doesn't depend on that order: ``finetune.py``'s length-grouped
sampler draws a new random permutation of the whole training split every epoch, and
``--max_train_samples`` takes a seeded random subset.

The input is streamed into memory-mapped Arrow files, so multi-gigabyte inputs don't
have to fit in memory. Malformed records raise an error instead of being skipped.

Outputs (in ``--output_dir``): a ``datasets.DatasetDict`` saved with
``save_to_disk`` (splits ``train``, ``validation``, ``test``), ``labels.json`` with
the label names in target order, ``README.md`` describing how the data was made (split
sizes, documents per source and per language, and the label distribution of every
split), and ``label_counts.tsv`` with the label distribution per split, source, and
language.

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


def input_file_limits(inputs: Sequence[Path], max_per_file: Sequence[int] | None) -> dict[str, int | None]:
    """Return the maximum number of records to read from each input file.

    Args:
        inputs: The ``--input`` paths (files or directories), in order.
        max_per_file: None for no limit, one limit for every input, or one per input.

    Returns:
        A mapping from each input file (as a string, as ``generate_rows`` receives
        it) to its limit, or None to read the whole file.

    Raises:
        ValueError: If the number of limits matches neither 1 nor the number of inputs.
    """
    if max_per_file is None:
        limits: list[int | None] = [None] * len(inputs)
    elif len(max_per_file) == 1:
        limits = [max_per_file[0]] * len(inputs)
    elif len(max_per_file) == len(inputs):
        limits = list(max_per_file)
    else:
        raise ValueError(
            f"--max_per_file has {len(max_per_file)} values for {len(inputs)} --input paths; "
            "give one value for all of them or one per path"
        )
    return {str(file): limit for path, limit in zip(inputs, limits) for file in find_input_files([path])}


def default_source_name(path: Path, label_type: str) -> str:
    """Return the source name of an input path when ``--source_names`` is not given.

    It is the name of the input's directory (for a file, the directory it is in), or of
    the directory above it if that directory is named after the label type, e.g.
    ``clean_random`` for ``results/distribution/<run>/clean_random/topic``.

    Raises:
        ValueError: If the path gives no directory name to use.
    """
    directory = path.resolve() if path.is_dir() else path.resolve().parent
    if directory.name == label_type:
        directory = directory.parent
    if not directory.name:
        raise ValueError(f"Cannot name the source of {path}; give --source_names")
    return directory.name


def input_file_sources(
    inputs: Sequence[Path], source_names: Sequence[str] | None, label_type: str
) -> dict[str, str]:
    """Return the source name of each input file.

    Args:
        inputs: The ``--input`` paths (files or directories), in order.
        source_names: One name per input, or None to name them with ``default_source_name``.
        label_type: The label type, which the default names skip as a directory name.

    Returns:
        A mapping from each input file (as a string, as ``generate_rows`` receives it)
        to the source name of its input.

    Raises:
        ValueError: If the number of names doesn't match the number of inputs, or a name is empty.
    """
    if source_names is None:
        names = [default_source_name(path, label_type) for path in inputs]
    elif len(source_names) == len(inputs):
        names = list(source_names)
    else:
        raise ValueError(f"--source_names has {len(source_names)} names for {len(inputs)} --input paths; "
                         "give one per path")
    if any(not name.strip() for name in names):
        raise ValueError(f"Source names must not be empty: {names}")
    return {str(file): name for path, name in zip(inputs, names) for file in find_input_files([path])}


def read_lines(path: Path, limit: int | None) -> Iterator[tuple[int, str]]:
    """Yield the line numbers (from 1) and lines of a file, stopping after ``limit`` lines.

    Args:
        path: The JSONL file.
        limit: The number of lines to read, or None to read them all.
    """
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if limit is not None and line_number > limit:
                return
            yield line_number, line


def language_of(path: Path) -> str:
    """Return the language encoded in a result file name, e.g. ``finnish-parallel.jsonl``."""
    return path.stem.split("-", 1)[0]


def hash_position(doc_id: str, seed: int) -> float:
    """Map a document ID to a reproducible pseudo-random number in [0, 1)."""
    digest = hashlib.sha256(f"{seed}:{doc_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def split_at(position: float, cuts: tuple[float, float]) -> str:
    """Return the split of a hash position: ``train`` below ``cuts[0]``, ``validation`` below ``cuts[1]``."""
    if position < cuts[0]:
        return "train"
    if position < cuts[1]:
        return "validation"
    return "test"


def split_of(doc_id: str, seed: int, train_ratio: float, val_ratio: float) -> str:
    """Assign a document ID to a split with a seeded hash.

    The hash maps the ID to a number in [0, 1); the split ratios cut that interval, so
    the realized split sizes are close to, but not exactly, the requested ratios.
    """
    return split_at(hash_position(doc_id, seed), (train_ratio, train_ratio + val_ratio))


def read_document_id(line: str, where: str) -> str:
    """Return the document ID of a result line, parsing it only if the ID is not its first field.

    Raises:
        ValueError: If the line has no document ID.
    """
    for prefix in ('{"id":"', '{"id": "'):
        if line.startswith(prefix):
            end = line.find('"', len(prefix))
            value = line[len(prefix):end]
            if end > len(prefix) and "\\" not in value:
                return value
    doc_id = json.loads(line).get("id")
    if not isinstance(doc_id, str) or not doc_id:
        raise ValueError(f"{where}: record without a document ID")
    return doc_id


def stratified_cuts(
    files: Sequence[Path],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    file_limits: dict[str, int | None] | None = None,
) -> dict[str, tuple[float, float]]:
    """Return, per language, the hash positions that split it in exactly the requested ratios.

    Within a language, the documents are ordered by their hash position; the first
    ``round(train_ratio * n)`` go to ``train`` and the documents up to
    ``round((train_ratio + val_ratio) * n)`` to ``validation``, the rest to ``test``.

    Args:
        files: Input JSONL files.
        seed: Seed of the split hash.
        train_ratio: Share of each language assigned to ``train``.
        val_ratio: Share of each language assigned to ``validation``.
        file_limits: Records to read per file (from ``input_file_limits``), so that
            only the documents that go into the dataset are counted; None reads all.

    Raises:
        ValueError: If a document ID present in several languages would land in
            different splits in them.
    """
    positions: dict[str, list[float]] = defaultdict(list)
    first_language: dict[str, str] = {}
    shared: dict[str, set[str]] = defaultdict(set)
    for path in files:
        language = language_of(path)
        limit = file_limits[str(path)] if file_limits is not None else None
        for line_number, line in read_lines(path, limit):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: empty line")
            doc_id = read_document_id(line, f"{path}:{line_number}")
            positions[language].append(hash_position(doc_id, seed))
            other = first_language.setdefault(doc_id, language)
            if other != language:
                shared[doc_id].update((other, language))

    cuts: dict[str, tuple[float, float]] = {}
    for language, values in positions.items():
        values.sort()
        n = len(values)
        n_train = round(train_ratio * n)
        n_train_val = max(n_train, round((train_ratio + val_ratio) * n))
        # The cut is the first position outside the split; 1.0 is above every position.
        cuts[language] = (
            values[n_train] if n_train < n else 1.0,
            values[n_train_val] if n_train_val < n else 1.0,
        )
        LOGGER.info("%s: %d documents, stratified into %d / %d / %d", language, n, n_train,
                    n_train_val - n_train, n - n_train_val)

    conflicts = [
        doc_id for doc_id, languages in shared.items()
        if len({split_at(hash_position(doc_id, seed), cuts[language]) for language in languages}) > 1
    ]
    if conflicts:
        raise ValueError(
            f"--stratify_by_language would put {len(conflicts)} document(s) in different splits in different "
            f"languages, e.g. {conflicts[0]!r}; leave it out to keep translations in the same split"
        )
    return cuts


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
    file_sources: dict[str, str],
    language_cuts: dict[str, tuple[float, float]] | None = None,
    file_limits: dict[str, int | None] | None = None,
) -> Iterator[dict[str, Any]]:
    """Stream validated rows, each tagged with its split, from the input files.

    Args:
        files: Input JSONL paths as strings (``datasets`` hashes generator arguments).
        label_names: Label short names in target order.
        seed: Seed of the split hash.
        train_ratio: Share of document IDs assigned to ``train``.
        val_ratio: Share of document IDs assigned to ``validation``.
        include_url: Add a ``url`` column (and require a ``url`` field in every record).
        file_sources: The source name of each file (from ``input_file_sources``), the
            ``source`` column of its rows.
        language_cuts: Per-language split cuts from ``stratified_cuts``, which replace
            the ratios; None to split every language by the ratios.
        file_limits: Records to read per file (from ``input_file_limits``): the first
            ones are used, and a file with fewer is used whole. None reads every file whole.

    Raises:
        ValueError: If a record is malformed or a document ID repeats within a language.
    """
    seen: set[tuple[str, str]] = set()
    for file_name in files:
        path = Path(file_name)
        language = language_of(path)
        source = file_sources[file_name]
        limit = file_limits[file_name] if file_limits is not None else None
        rows = 0
        for line_number, line in read_lines(path, limit):
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
            row = {"id": doc_id, "language": language, "source": source}
            if include_url:
                row["url"] = record_url(record, where)
            rows += 1
            yield {
                **row,
                "text": text,
                "labels": label_vector(record, label_names, where),
                "split": split_at(
                    hash_position(doc_id, seed),
                    language_cuts[language] if language_cuts is not None
                    else (train_ratio, train_ratio + val_ratio),
                ),
            }
        if limit is not None and rows < limit:
            LOGGER.warning("%s: %d documents, fewer than the limit of %d; using all of them", path, rows, limit)
        else:
            LOGGER.info("%s: %d documents (language %s, source %s)", path, rows, language, source)


@dataclass
class SplitStatistics:
    """Document and label counts of one split, per source and language.

    The keys are ``(source, language)`` pairs.

    Attributes:
        num_labels: Number of labels.
        documents: Documents per source and language.
        top_label: Per source and language, the number of documents whose top label is
            each label (an array in label order).
        soft: Per source and language, the summed probability of each label (an array
            in label order); divided by the documents, it is the label's mean probability.
    """

    num_labels: int
    documents: Counter[tuple[str, str]]
    top_label: dict[tuple[str, str], np.ndarray]
    soft: dict[tuple[str, str], np.ndarray]

    @property
    def total(self) -> int:
        """Number of documents in the split."""
        return sum(self.documents.values())

    def documents_per_source(self) -> Counter[str]:
        """Documents per source, over all languages."""
        counts: Counter[str] = Counter()
        for (source, _), count in self.documents.items():
            counts[source] += count
        return counts

    def documents_per_language(self) -> Counter[str]:
        """Documents per language, over all sources."""
        counts: Counter[str] = Counter()
        for (_, language), count in self.documents.items():
            counts[language] += count
        return counts

    def top_label_total(self) -> np.ndarray:
        """Top-label counts over all languages, in label order."""
        return np.sum([*self.top_label.values(), np.zeros(self.num_labels, dtype=np.int64)], axis=0)

    def soft_total(self) -> np.ndarray:
        """Summed probabilities over all languages, in label order."""
        return np.sum([*self.soft.values(), np.zeros(self.num_labels)], axis=0)


def split_statistics(dataset: datasets.Dataset, num_labels: int, batch_size: int = 10_000) -> SplitStatistics:
    """Count documents, top labels, and summed label probabilities per source and language.

    The split is read in batches from its Arrow files, so memory use doesn't grow with
    the dataset. A document's top label is ``argmax`` of its probabilities (the first
    label in label order on a tie, as in ``finetune.py``).

    Raises:
        ValueError: If a label vector doesn't have ``num_labels`` entries.
    """
    documents: Counter[tuple[str, str]] = Counter()
    top_label: dict[tuple[str, str], np.ndarray] = defaultdict(lambda: np.zeros(num_labels, dtype=np.int64))
    soft: dict[tuple[str, str], np.ndarray] = defaultdict(lambda: np.zeros(num_labels, dtype=np.float64))
    columns = dataset.select_columns(["source", "language", "labels"]).with_format("numpy")
    for batch in columns.iter(batch_size=batch_size):
        probabilities = np.asarray(batch["labels"], dtype=np.float64)
        if probabilities.ndim != 2 or probabilities.shape[1] != num_labels:
            raise ValueError(f"Expected label vectors of length {num_labels}, got shape {probabilities.shape}")
        sources = np.asarray(batch["source"])
        languages = np.asarray(batch["language"])
        top = probabilities.argmax(axis=1)
        for source in np.unique(sources):
            source_mask = sources == source
            for language in np.unique(languages[source_mask]):
                mask = source_mask & (languages == language)
                key = (str(source), str(language))
                documents[key] += int(mask.sum())
                top_label[key] += np.bincount(top[mask], minlength=num_labels)
                soft[key] += probabilities[mask].sum(axis=0)
    return SplitStatistics(num_labels, documents, dict(top_label), dict(soft))


def write_label_counts(path: Path, statistics: dict[str, SplitStatistics], label_names: Sequence[str]) -> None:
    """Write the label distribution per split, source, language, and label as a TSV file.

    Columns: split, source, language, label, documents (in that split, source, and
    language), top_label_documents, and probability_sum (the label's soft count).
    """
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["split", "source", "language", "label", "documents", "top_label_documents",
                         "probability_sum"])
        for split, stats in statistics.items():
            for key in sorted(stats.documents):
                for index, label in enumerate(label_names):
                    writer.writerow([split, *key, label, stats.documents[key],
                                     int(stats.top_label[key][index]), f"{stats.soft[key][index]:.4f}"])


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

    def count_rows(per_split: dict[str, Counter[str]]) -> str:
        """Table rows of documents per split for each name, with a total row."""
        names = sorted({name for counts in per_split.values() for name in counts})
        rows = "\n".join(
            f"| {name} | " + " | ".join(f"{per_split[split][name]:,}" for split in SPLITS)
            + f" | {sum(per_split[split][name] for split in SPLITS):,} |"
            for name in names
        )
        return rows + "\n| **Total** | " + " | ".join(f"**{totals[split]:,}**" for split in SPLITS) + \
            f" | **{all_documents:,}** |"

    source_rows = count_rows({split: statistics[split].documents_per_source() for split in SPLITS})
    language_rows = count_rows({split: statistics[split].documents_per_language() for split in SPLITS})

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

    file_limits = input_file_limits(args.input, args.max_per_file)
    file_sources = input_file_sources(args.input, args.source_names, args.label_type)
    input_list = "\n".join(
        f"- {file} (source {file_sources[str(file)]}"
        + (f", first {file_limits[str(file)]:,} documents at most" if file_limits[str(file)] else "") + ")"
        for file in files
    )
    if args.stratify_by_language:
        split_text = (
            f"Stratified by language (`--stratify_by_language`): within each language, the documents\n"
            f"are ordered by a seeded hash of their ID (seed {args.seed}) and cut at the requested\n"
            "ratios, so every language is split in exactly those shares (rounded). Documents present\n"
            "in several languages are in the same split in all of them."
        )
    else:
        split_text = (
            f"Document IDs are assigned to splits by a seeded hash (seed {args.seed}), so all language\n"
            "versions of a document are in the same split. The realized shares are close to, but\n"
            "not exactly, the requested ratios."
        )
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

{split_text}

| Split | Documents | Share | Requested |
|---|---:|---:|---:|
{size_rows}

### Documents per source

The `source` column names the input each document came from.

| Source | train | validation | test | Total |
|---|---:|---:|---:|---:|
{source_rows}

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
a second choice, and lower for labels it picks with low confidence. Counts per source and
language are in `label_counts.tsv`.

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
    parser.add_argument(
        "--max_per_file",
        type=int,
        nargs="+",
        default=None,
        help="Use at most this many documents (the first ones) from each input file: one value for "
        "all --input paths or one per path, in the same order. A file with fewer is used whole. "
        "The files must be in random order, as the HPLT results are.",
    )
    parser.add_argument(
        "--source_names",
        nargs="+",
        default=None,
        help="The source column value of each --input path, one per path in the same order "
        "(default: the input's directory name, skipping a directory named after the label type).",
    )
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of training data.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Ratio of validation data.")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Ratio of test data.")
    parser.add_argument("--seed", type=int, default=42, help="Seed of the document-ID split hash.")
    parser.add_argument(
        "--stratify_by_language",
        action="store_true",
        help="Split every language in exactly the requested ratios instead of approximately "
        "(reads the input once more to count the documents).",
    )
    parser.add_argument("--log_level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Check the split ratios, the per-file limits, the source names, and the output directory.

    Raises:
        ValueError: If the ratios are negative or don't sum to 1, a per-file limit is
            not positive or there are neither one nor one per input of them, or the
            source names are not one per input.
        FileExistsError: If the output directory is not empty and can't be replaced.
    """
    if args.source_names is not None and len(args.source_names) != len(args.input):
        raise ValueError(f"--source_names has {len(args.source_names)} names for {len(args.input)} --input paths; "
                         "give one per path")
    if args.max_per_file is not None:
        if any(limit < 1 for limit in args.max_per_file):
            raise ValueError("--max_per_file values must be positive")
        if len(args.max_per_file) not in (1, len(args.input)):
            raise ValueError(
                f"--max_per_file has {len(args.max_per_file)} values for {len(args.input)} --input paths; "
                "give one value for all of them or one per path"
            )
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
    file_limits = input_file_limits(args.input, args.max_per_file)
    file_sources = input_file_sources(args.input, args.source_names, args.label_type)
    for file in files:
        limit = file_limits[str(file)]
        LOGGER.info("Input %s (language %s, source %s%s)", file, language_of(file), file_sources[str(file)],
                    f", first {limit} documents" if limit else "")
    LOGGER.info("%d labels from %s", len(label_names), labels_path)

    language_cuts = (
        stratified_cuts(files, args.seed, args.train_ratio, args.val_ratio, file_limits)
        if args.stratify_by_language else None
    )
    include_url = not args.no_url
    LOGGER.info("URL column: %s", "yes" if include_url else "no (--no_url)")
    columns = {
        "id": datasets.Value("string"),
        "language": datasets.Value("string"),
        "source": datasets.Value("string"),
    }
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
                "file_sources": file_sources,
                "language_cuts": language_cuts,
                "file_limits": file_limits,
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
            LOGGER.info("%s: %d rows, per source %s, per language %s", split, len(dataset_dict[split]),
                        dict(sorted(statistics[split].documents_per_source().items())),
                        dict(sorted(statistics[split].documents_per_language().items())))
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
