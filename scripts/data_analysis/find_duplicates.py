"""Find exact and near-duplicate documents in a dataset made by ``create_datasets.py``.

Every document has a unique ID, but the same or nearly the same text can still occur
several times: mirrored pages, boilerplate templates, the same page in two HPLT samples.
Duplicates inside the training split over-weight those pages, and duplicates shared
between the training split and the validation or test split leak training documents into
the evaluation and inflate its scores. This script finds both, cheaply:

* **Exact duplicates:** documents whose *normalised* text is identical (Unicode NFKC,
  case-folded, runs of whitespace collapsed to one space), found by a 64-bit hash of the
  whole normalised text.
* **Near duplicates:** MinHash over the set of character n-grams (default 5 characters)
  of the first ``--max_chars`` normalised characters, with locality-sensitive hashing
  (LSH) in bands. Character n-grams work the same in every script, including Chinese,
  Japanese, Thai and other languages written without spaces. Candidate pairs that share a
  band are kept if their estimated Jaccard similarity (the share of equal MinHash values)
  is at least ``--threshold`` (default 0.8), and the kept pairs are joined into clusters.
  With 128 hash functions in 16 bands of 8, a pair with Jaccard similarity 0.8 becomes a
  candidate with probability 0.95, and one with 0.5 with probability 0.06. Within an LSH
  bucket every member is compared with the bucket's first document only, which keeps the
  cost linear; a near-duplicate pair can therefore be missed when it shares no bucket with
  a common first document, so the near-duplicate counts are a slight underestimate.

Only the ``text`` column is compared, not the URL. Documents that are empty after
normalisation are counted separately and left out of both comparisons. Exact URL
duplicates are reported too, if the dataset has a ``url`` column.

The dataset is read with pyarrow directly from the Arrow files listed in each split's
``state.json``; only numpy and pyarrow are needed.

Outputs in ``--output_dir``:

* ``summary.json``: counts per split, cross-split leaks, the largest clusters, settings.
* ``languages.csv``: per language, documents, duplicates and leaks.
* ``duplicates.tsv``: one row per document in a near-duplicate cluster (a cluster
  includes its exact duplicates).
* ``url_duplicates.tsv``: one row per document whose URL occurs more than once (if the
  dataset has a ``url`` column), with its near-duplicate cluster, if any, and the
  estimated similarity of its text to the first document with that URL.
* ``examples.jsonl``: the largest clusters and a random sample of clusters, with the
  beginning of a few members' text, for checking by eye.

Both TSV files give each document's top label (``top_label``, the most probable label of
its ``labels`` distribution, named from the dataset's ``labels.json``) if the dataset has
a ``labels`` column, so the labels of duplicates can be compared.

A document is counted as a *redundant copy* if its cluster has a member earlier in the
dataset order (splits in ``--splits`` order, then row order); removing all redundant
copies would leave one document per cluster. A validation or test document *leaks* if its
cluster contains a training document.

Example (from the repository root, or with hpc/roihu/find_duplicates.sh):
    python scripts/data_analysis/find_duplicates.py \\
        --dataset results/datasets/topic/hplt-finetune \\
        --output_dir results/statistics/duplicates/topic/hplt-finetune
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import multiprocessing
import re
import time
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

LOGGER = logging.getLogger(__name__)

TEXT_COLUMN = "text"
METADATA_COLUMNS = ("id", "language", "source", "url", "labels")
REQUIRED_COLUMNS = ("id", "language", TEXT_COLUMN)
TRAIN_SPLIT = "train"
WHITESPACE = re.compile(r"\s+")
# Multiplier of the rolling n-gram hash (any odd 64-bit constant works).
ROLLING_PRIME = np.uint64(0x100000001B3)
MASK64 = (1 << 64) - 1
NO_SIGNATURE = np.uint32(0xFFFFFFFF)

# Per-process settings of the worker pool, set by _init_worker.
_SETTINGS: dict[str, Any] = {}


@dataclass(frozen=True)
class Chunk:
    """A run of consecutive record batches of one Arrow file, processed as one task.

    Attributes:
        split: Split name.
        path: Arrow file path.
        first_batch: Index of the first record batch in the file.
        batch_count: Number of record batches.
        first_row: Global row index (over all splits) of the chunk's first document.
        rows: Number of documents in the chunk.
    """

    split: str
    path: str
    first_batch: int
    batch_count: int
    first_row: int
    rows: int


def splitmix64(values: np.ndarray) -> np.ndarray:
    """Mix 64-bit integers into well-distributed 64-bit hashes (SplitMix64 finaliser)."""
    with np.errstate(over="ignore"):
        values = values.astype(np.uint64, copy=True)
        values += np.uint64(0x9E3779B97F4A7C15)
        values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return values ^ (values >> np.uint64(31))


def normalize_text(text: str) -> str:
    """Return the text as compared: NFKC, case-folded, whitespace runs collapsed, stripped."""
    return WHITESPACE.sub(" ", unicodedata.normalize("NFKC", text).casefold()).strip()


def exact_hash(normalized: str) -> int:
    """Return a 64-bit hash of a whole normalised text."""
    return int.from_bytes(hashlib.blake2b(normalized.encode("utf-8"), digest_size=8).digest(), "little")


def ngram_hashes(normalized: str, size: int, max_chars: int) -> np.ndarray:
    """Return the distinct 64-bit hashes of the character n-grams of a normalised text.

    Only the first ``max_chars`` characters are used. A text shorter than ``size`` is one
    n-gram.

    Raises:
        ValueError: If the text is empty.
    """
    codes = np.frombuffer(normalized[:max_chars].encode("utf-32-le"), dtype=np.uint32).astype(np.uint64)
    if not len(codes):
        raise ValueError("Can't hash the n-grams of an empty text")
    size = min(size, len(codes))
    count = len(codes) - size + 1
    hashes = np.zeros(count, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for offset in range(size):
            hashes = hashes * ROLLING_PRIME + codes[offset:offset + count]
    return np.unique(splitmix64(hashes))


def minhash_signature(hashes: np.ndarray, multipliers: np.ndarray, increments: np.ndarray,
                      block: int = 4096) -> np.ndarray:
    """Return the MinHash signature of a set of n-gram hashes.

    Each of the ``len(multipliers)`` hash functions is ``(a * h + b) >> 32`` over 64-bit
    integers (multiply-shift hashing); the signature holds each function's minimum.
    """
    signature = np.full(len(multipliers), np.iinfo(np.uint64).max, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for start in range(0, len(hashes), block):
            values = np.multiply.outer(multipliers, hashes[start:start + block])
            values += increments[:, None]
            values >>= np.uint64(32)
            np.minimum(signature, values.min(axis=1), out=signature)
    return signature.astype(np.uint32)


def read_split_files(dataset: Path, split: str) -> list[Path]:
    """Return the Arrow data files of a saved split, in order.

    Raises:
        ValueError: If the split has an indices mapping (it should be saved flattened).
    """
    state = json.loads((dataset / split / "state.json").read_text(encoding="utf-8"))
    if state.get("_indices_data_files"):
        raise ValueError(f"{dataset / split} has an indices mapping; save it with flatten_indices() first")
    files = [dataset / split / item["filename"] for item in state["_data_files"]]
    if not files:
        raise ValueError(f"{dataset / split} lists no data files")
    return files


def open_batches(path: Path) -> pa.RecordBatchStreamReader:
    """Open an Arrow stream file memory-mapped (only the columns used are read from disk)."""
    return pa.ipc.open_stream(pa.memory_map(str(path)))


def plan_chunks(dataset: Path, splits: Sequence[str], chunk_rows: int,
                max_documents: int | None) -> tuple[list[Chunk], dict[str, tuple[int, int]]]:
    """Split the dataset into tasks of about ``chunk_rows`` documents.

    Returns:
        The chunks, and each split's (first global row, number of rows).

    Raises:
        ValueError: If a split lacks a required column.
    """
    chunks: list[Chunk] = []
    ranges: dict[str, tuple[int, int]] = {}
    row = 0
    for split in splits:
        split_start = row
        for path in read_split_files(dataset, split):
            reader = open_batches(path)
            missing = [name for name in REQUIRED_COLUMNS if name not in reader.schema.names]
            if missing:
                raise ValueError(f"{path} lacks columns {missing}; it has {reader.schema.names}")
            first_batch, batch_count, rows = 0, 0, 0
            for index, batch in enumerate(reader):
                # Documents of this split still wanted (all of them without --max_documents).
                remaining = None if max_documents is None else max_documents - (row + rows - split_start)
                if remaining is not None and remaining <= 0:
                    break
                if batch_count == 0:
                    first_batch = index
                batch_count += 1
                rows += batch.num_rows if remaining is None else min(batch.num_rows, remaining)
                if rows >= chunk_rows:
                    chunks.append(Chunk(split, str(path), first_batch, batch_count, row, rows))
                    row += rows
                    batch_count, rows = 0, 0
            if rows:
                chunks.append(Chunk(split, str(path), first_batch, batch_count, row, rows))
                row += rows
        ranges[split] = (split_start, row - split_start)
        LOGGER.info("%s: %d documents", split, row - split_start)
    return chunks, ranges


def chunk_texts(chunk: Chunk) -> list[str]:
    """Return the texts of a chunk's documents.

    Raises:
        ValueError: If a text is null.
    """
    texts: list[str] = []
    for index, batch in enumerate(open_batches(Path(chunk.path))):
        if index < chunk.first_batch:
            continue
        if index >= chunk.first_batch + chunk.batch_count:
            break
        texts.extend(batch.column(TEXT_COLUMN).to_pylist())
    texts = texts[:chunk.rows]
    if len(texts) != chunk.rows:
        raise ValueError(f"{chunk.path}: expected {chunk.rows} documents from batch {chunk.first_batch}, got {len(texts)}")
    for offset, text in enumerate(texts):
        if text is None:
            raise ValueError(f"{chunk.path}: document {chunk.first_row + offset} has no text")
    return texts


def _init_worker(ngram_size: int, max_chars: int, multipliers: np.ndarray, increments: np.ndarray) -> None:
    """Store the hashing settings in a worker process."""
    _SETTINGS.update(ngram_size=ngram_size, max_chars=max_chars, multipliers=multipliers, increments=increments)


def process_chunk(chunk: Chunk) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Hash a chunk's documents.

    Returns:
        The chunk's first row, MinHash signatures (rows x hash functions; all
        ``NO_SIGNATURE`` for empty texts), exact hashes, and normalised text lengths.
    """
    texts = chunk_texts(chunk)
    permutations = len(_SETTINGS["multipliers"])
    signatures = np.full((len(texts), permutations), NO_SIGNATURE, dtype=np.uint32)
    exact = np.zeros(len(texts), dtype=np.uint64)
    lengths = np.zeros(len(texts), dtype=np.int64)
    for index, text in enumerate(texts):
        normalized = normalize_text(text)
        lengths[index] = len(normalized)
        exact[index] = exact_hash(normalized)
        if normalized:
            hashes = ngram_hashes(normalized, _SETTINGS["ngram_size"], _SETTINGS["max_chars"])
            signatures[index] = minhash_signature(hashes, _SETTINGS["multipliers"], _SETTINGS["increments"])
    return chunk.first_row, signatures, exact, lengths


def band_keys(signatures: np.ndarray, start: int, stop: int) -> np.ndarray:
    """Return a 64-bit key per row from the signature columns ``start:stop`` (one LSH band)."""
    keys = np.full(len(signatures), start + 1, dtype=np.uint64)
    for column in range(start, stop):
        keys = splitmix64(keys ^ signatures[:, column].astype(np.uint64))
    return keys


def candidate_pairs(keys: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pair every row with the first row of its bucket (rows sharing a key).

    Args:
        keys: One band key per row in ``rows``.
        rows: Global row indices, ascending.

    Returns:
        First rows and other rows of the buckets with more than one member.
    """
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    starts = np.ones(len(order), dtype=bool)
    starts[1:] = sorted_keys[1:] != sorted_keys[:-1]
    first = np.maximum.accumulate(np.where(starts, np.arange(len(order)), 0))
    members = ~starts
    return rows[order[first[members]]], rows[order[members]]


def estimated_jaccard(signatures: np.ndarray, left: np.ndarray, right: np.ndarray,
                      block: int = 200_000) -> np.ndarray:
    """Return the share of equal MinHash values for each pair of rows."""
    similarity = np.empty(len(left), dtype=np.float32)
    for start in range(0, len(left), block):
        stop = start + block
        similarity[start:stop] = (signatures[left[start:stop]] == signatures[right[start:stop]]).mean(axis=1)
    return similarity


def connected_components(count: int, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return a component label per node (the smallest node index in its component)."""
    labels = np.arange(count, dtype=np.int64)
    while True:
        previous = labels.copy()
        np.minimum.at(labels, left, labels[right])
        np.minimum.at(labels, right, labels[left])
        while True:  # pointer jumping
            jumped = labels[labels]
            if np.array_equal(jumped, labels):
                break
            labels = jumped
        if np.array_equal(labels, previous):
            return labels


def group_sizes(labels: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Return the size of each row's group, counting valid rows only (0 for invalid rows)."""
    counts = np.bincount(labels[valid], minlength=len(labels))
    return np.where(valid, counts[labels], 0)


def first_rows(labels: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Return, per row, the first valid row with the same label (len(labels) if there is none)."""
    first = np.full(labels.max() + 1 if len(labels) else 0, len(labels), dtype=np.int64)
    rows = np.flatnonzero(valid)
    np.minimum.at(first, labels[rows], rows)
    return first[labels]


def redundant_mask(labels: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Mark valid rows that have an earlier valid row with the same label."""
    return valid & (first_rows(labels, valid) < np.arange(len(labels)))


def top_labels(metadata: dict[str, pa.ChunkedArray], rows: np.ndarray, label_names: Sequence[str] | None) -> list[str]:
    """Return the most probable label of each row's ``labels`` distribution ("" without labels).

    Raises:
        ValueError: If a distribution's length doesn't match the label names.
    """
    if "labels" not in metadata:
        return [""] * len(rows)
    result = []
    for row, distribution in zip(rows, metadata["labels"].take(pa.array(rows)).to_pylist()):
        if not distribution:
            raise ValueError(f"Document {int(row)} has no label distribution")
        index = int(np.argmax(distribution))
        if label_names is None:
            result.append(str(index))
        elif len(distribution) != len(label_names):
            raise ValueError(f"Document {int(row)} has {len(distribution)} label probabilities for {len(label_names)} labels")
        else:
            result.append(label_names[index])
    return result


def read_metadata(dataset: Path, splits: Sequence[str], ranges: dict[str, tuple[int, int]]) -> dict[str, pa.ChunkedArray]:
    """Read the metadata columns of all documents (memory-mapped), in global row order."""
    columns: dict[str, list[pa.Array]] = {}
    for split in splits:
        wanted = ranges[split][1]
        for path in read_split_files(dataset, split):
            if wanted <= 0:
                break
            table = open_batches(path).read_all()
            table = table.slice(0, min(wanted, table.num_rows))
            wanted -= table.num_rows
            for name in METADATA_COLUMNS:
                if name in table.column_names:
                    columns.setdefault(name, []).extend(table.column(name).chunks)
    total = sum(count for _, count in ranges.values())
    result = {}
    for name, chunks in columns.items():
        array = pa.chunked_array(chunks)
        if len(array) != total:
            raise ValueError(f"Column {name!r} has {len(array)} values for {total} documents; is it in every split?")
        result[name] = array
    return result


def codes_of(array: pa.ChunkedArray) -> tuple[np.ndarray, list[Any]]:
    """Return integer codes and the distinct values of a column (nulls get code -1)."""
    encoded = array.dictionary_encode().combine_chunks()
    codes = encoded.indices.to_numpy(zero_copy_only=False)
    codes = np.where(np.isnan(codes), -1, codes).astype(np.int64) if codes.dtype.kind == "f" else codes.astype(np.int64)
    return codes, encoded.dictionary.to_pylist()


def split_codes(splits: Sequence[str], ranges: dict[str, tuple[int, int]]) -> np.ndarray:
    """Return the index (into ``splits``) of each row's split."""
    codes = np.empty(sum(count for _, count in ranges.values()), dtype=np.int64)
    for index, split in enumerate(splits):
        start, count = ranges[split]
        codes[start:start + count] = index
    return codes


def leak_mask(labels: np.ndarray, valid: np.ndarray, split_index: np.ndarray, train_index: int | None) -> np.ndarray:
    """Mark valid non-training rows whose group contains a valid training row."""
    if train_index is None:
        return np.zeros(len(labels), dtype=bool)
    has_train = np.zeros(labels.max() + 1 if len(labels) else 0, dtype=bool)
    has_train[labels[valid & (split_index == train_index)]] = True
    return valid & (split_index != train_index) & has_train[labels]


def read_texts(chunks: Sequence[Chunk], rows: Sequence[int]) -> dict[int, str]:
    """Return the texts of the given global rows."""
    wanted = sorted(set(rows))
    texts: dict[int, str] = {}
    for chunk in chunks:
        inside = [row for row in wanted if chunk.first_row <= row < chunk.first_row + chunk.rows]
        if inside:
            chunk_text = chunk_texts(chunk)
            texts.update({row: chunk_text[row - chunk.first_row] for row in inside})
    return texts


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description="Find exact and near-duplicate documents in a saved dataset.")
    parser.add_argument("--dataset", type=Path, required=True, help="Dataset directory made by create_datasets.py")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", help="Splits to compare (default: all, training split first)")
    parser.add_argument("--ngram_size", type=int, default=5, help="Characters per n-gram")
    parser.add_argument("--max_chars", type=int, default=20_000,
                        help="Normalised characters per document used for near-duplicate detection "
                             "(exact duplicates use the whole text)")
    parser.add_argument("--permutations", type=int, default=128, help="MinHash hash functions")
    parser.add_argument("--bands", type=int, default=16, help="LSH bands (must divide --permutations)")
    parser.add_argument("--threshold", type=float, default=0.8, help="Minimum estimated Jaccard similarity")
    parser.add_argument("--examples", type=int, default=30,
                        help="Clusters written to examples.jsonl: this many largest and this many random")
    parser.add_argument("--max_documents", type=int, help="Use only the first N documents of each split (testing)")
    parser.add_argument("--chunk_rows", type=int, default=20_000, help="Documents per worker task")
    parser.add_argument("--workers", type=int, default=max(1, (multiprocessing.cpu_count() or 2) - 1))
    parser.add_argument("--seed", type=int, default=0, help="Seed of the hash functions and example sample")
    parser.add_argument("--overwrite", action="store_true", help="Write into a non-empty --output_dir")
    parser.add_argument("--log_level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Find duplicates and write the reports."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s [%(levelname)s] %(message)s")
    if args.ngram_size < 1 or args.max_chars < 1 or args.permutations < 1 or args.workers < 1 or args.chunk_rows < 1:
        parser.error("--ngram_size, --max_chars, --permutations, --workers and --chunk_rows must be positive")
    if args.bands < 1 or args.permutations % args.bands:
        parser.error("--bands must be positive and divide --permutations")
    if not 0 < args.threshold <= 1:
        parser.error("--threshold must be in (0, 1]")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        parser.error(f"{args.output_dir} is not empty; pass --overwrite to write into it")

    started = time.monotonic()
    available = json.loads((args.dataset / "dataset_dict.json").read_text(encoding="utf-8"))["splits"]
    splits = args.splits or sorted(available, key=lambda name: (name != TRAIN_SPLIT, available.index(name)))
    unknown = [split for split in splits if split not in available]
    if unknown:
        parser.error(f"{args.dataset} has no splits {unknown}; it has {available}")

    chunks, ranges = plan_chunks(args.dataset, splits, args.chunk_rows, args.max_documents)
    total = sum(count for _, count in ranges.values())
    LOGGER.info("%d documents in %d tasks, %d workers; %d-character n-grams of the first %d characters, "
                "%d hash functions in %d bands, threshold %.2f", total, len(chunks), args.workers,
                args.ngram_size, args.max_chars, args.permutations, args.bands, args.threshold)

    # Hash functions: odd 64-bit multipliers and arbitrary increments.
    rng = np.random.default_rng(args.seed)
    multipliers = rng.integers(0, 2**63, size=args.permutations, dtype=np.uint64) * np.uint64(2) + np.uint64(1)
    increments = rng.integers(0, 2**63, size=args.permutations, dtype=np.uint64)

    signatures = np.empty((total, args.permutations), dtype=np.uint32)
    exact = np.empty(total, dtype=np.uint64)
    lengths = np.empty(total, dtype=np.int64)
    done = np.zeros(total, dtype=bool)
    init_args = (args.ngram_size, args.max_chars, multipliers, increments)
    with multiprocessing.Pool(args.workers, initializer=_init_worker, initargs=init_args) as pool:
        for count, (first_row, chunk_signatures, chunk_exact, chunk_lengths) in enumerate(
                pool.imap_unordered(process_chunk, chunks), start=1):
            stop = first_row + len(chunk_exact)
            signatures[first_row:stop] = chunk_signatures
            exact[first_row:stop] = chunk_exact
            lengths[first_row:stop] = chunk_lengths
            done[first_row:stop] = True
            if count % 20 == 0 or count == len(chunks):
                LOGGER.info("Hashed %d of %d tasks (%.0f s)", count, len(chunks), time.monotonic() - started)
    if not done.all():
        raise RuntimeError(f"{int((~done).sum())} documents were not hashed")

    valid = lengths > 0
    LOGGER.info("%d documents are empty after normalisation and are left out", int((~valid).sum()))

    # Exact duplicates.
    _, exact_labels = np.unique(exact, return_inverse=True)
    exact_labels = exact_labels.astype(np.int64)

    # Near duplicates: LSH candidates, verified by estimated Jaccard similarity.
    rows = np.flatnonzero(valid)
    band_width = args.permutations // args.bands
    lefts, rights = [], []
    for band in range(args.bands):
        keys = band_keys(signatures, band * band_width, (band + 1) * band_width)[rows]
        left, right = candidate_pairs(keys, rows)
        lefts.append(left)
        rights.append(right)
    pair_keys = np.unique(np.concatenate(lefts).astype(np.uint64) * np.uint64(total)
                          + np.concatenate(rights).astype(np.uint64))
    del lefts, rights
    left = (pair_keys // np.uint64(total)).astype(np.int64)
    right = (pair_keys % np.uint64(total)).astype(np.int64)
    similarity = estimated_jaccard(signatures, left, right)
    kept = similarity >= args.threshold
    LOGGER.info("%d candidate pairs, %d at estimated Jaccard >= %.2f (%.0f s)", len(left), int(kept.sum()),
                args.threshold, time.monotonic() - started)
    # A cluster's label is its smallest row, which is also its first member.
    near_labels = connected_components(total, left[kept], right[kept])

    # Per-row facts.
    split_index = split_codes(splits, ranges)
    train_index = splits.index(TRAIN_SPLIT) if TRAIN_SPLIT in splits else None
    near_size = group_sizes(near_labels, valid)
    exact_size = group_sizes(exact_labels, valid)
    near_redundant = redundant_mask(near_labels, valid)
    exact_redundant = redundant_mask(exact_labels, valid)
    near_leak = leak_mask(near_labels, valid, split_index, train_index)
    exact_leak = leak_mask(exact_labels, valid, split_index, train_index)
    metadata = read_metadata(args.dataset, splits, ranges)
    language_codes, language_names = codes_of(metadata["language"])
    labels_file = args.dataset / "labels.json"
    label_names = (json.loads(labels_file.read_text(encoding="utf-8"))["label_names"]
                   if labels_file.is_file() else None)
    url_redundant = url_leak = url_labels = has_url = None
    if "url" in metadata:
        url_labels, _ = codes_of(metadata["url"])
        has_url = url_labels >= 0
        url_labels = np.where(has_url, url_labels, np.arange(total) + url_labels.max() + 1)
        url_redundant = redundant_mask(url_labels, has_url)
        url_leak = leak_mask(url_labels, has_url, split_index, train_index)
    else:
        LOGGER.warning("The dataset has no url column; URL duplicates are not reported")

    def counts(mask: np.ndarray) -> dict[str, int]:
        return {split: int(mask[split_index == index].sum()) for index, split in enumerate(splits)}

    in_cluster = near_size > 1
    multilingual = np.zeros(total, dtype=bool)
    if in_cluster.any():
        pairs = np.unique(np.stack([near_labels[in_cluster], language_codes[in_cluster]], axis=1), axis=0)
        languages_per_cluster = np.bincount(pairs[:, 0], minlength=total)
        multilingual = in_cluster & (languages_per_cluster[near_labels] > 1)

    cluster_ids, cluster_sizes = np.unique(near_labels[in_cluster], return_counts=True)
    largest = cluster_ids[np.argsort(-cluster_sizes, kind="stable")][:20]
    largest_clusters = []
    for label in largest:
        members = np.flatnonzero(in_cluster & (near_labels == label))
        largest_clusters.append({
            "cluster": int(label),
            "documents": len(members),
            "splits": {split: int((split_index[members] == index).sum()) for index, split in enumerate(splits)
                       if (split_index[members] == index).any()},
            "languages": sorted({language_names[code] for code in language_codes[members]}),
            "exact_copies_of_first": int((exact_labels[members] == exact_labels[members[0]]).sum()),
            "first_id": metadata["id"][int(members[0])].as_py(),
        })

    summary: dict[str, Any] = {
        "dataset": str(args.dataset),
        "settings": {name: getattr(args, name) for name in (
            "ngram_size", "max_chars", "permutations", "bands", "threshold", "seed", "max_documents")},
        "documents": {split: ranges[split][1] for split in splits},
        "empty_after_normalisation": counts(~valid),
        "exact": {
            "documents_with_a_copy": counts(exact_size > 1),
            "redundant_copies": counts(exact_redundant),
            "groups": int(len(np.unique(exact_labels[exact_size > 1]))),
        },
        "near": {
            "documents_in_clusters": counts(in_cluster),
            "redundant_copies": counts(near_redundant),
            "clusters": int(len(cluster_ids)),
            "documents_in_clusters_spanning_languages": counts(multilingual),
            "candidate_pairs": int(len(left)),
            "verified_pairs": int(kept.sum()),
        },
        "leaks_into_training_split": {
            "near": counts(near_leak),
            "exact": counts(exact_leak),
        } if train_index is not None else None,
        "largest_clusters": largest_clusters,
    }
    if url_redundant is not None:
        summary["url"] = {"redundant_copies": counts(url_redundant)}
        if train_index is not None:
            summary["leaks_into_training_split"]["url"] = counts(url_leak)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                                  encoding="utf-8")

    # Per language.
    with (args.output_dir / "languages.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        header = ["Language", "Documents", "Empty", "Exact redundant copies", "Near redundant copies",
                  "In clusters spanning languages"]
        if train_index is not None:
            header += ["Evaluation documents", "Near leaks into training", "Exact leaks into training"]
        writer.writerow(header)
        evaluation = valid & (split_index != train_index) if train_index is not None else None
        for code, name in sorted(enumerate(language_names), key=lambda item: str(item[1])):
            mine = language_codes == code
            row = [name, int(mine.sum()), int((mine & ~valid).sum()), int((mine & exact_redundant).sum()),
                   int((mine & near_redundant).sum()), int((mine & multilingual).sum())]
            if evaluation is not None:
                row += [int((mine & evaluation).sum()), int((mine & near_leak).sum()), int((mine & exact_leak).sum())]
            writer.writerow(row)

    # One row per document in a cluster.
    members = np.flatnonzero(in_cluster)
    members = members[np.lexsort((members, near_labels[members]))]
    similarity_to_first = estimated_jaccard(signatures, near_labels[members], members)
    ids = metadata["id"].take(pa.array(members)).to_pylist()
    sources = metadata["source"].take(pa.array(members)).to_pylist() if "source" in metadata else [None] * len(members)
    urls = metadata["url"].take(pa.array(members)).to_pylist() if "url" in metadata else [None] * len(members)
    with (args.output_dir / "duplicates.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["cluster", "cluster_size", "split", "row", "id", "language", "source", "url",
                         "exact_group_size", "estimated_jaccard_to_first", "redundant", "leak", "top_label"])
        for index, (row, label) in enumerate(zip(members, top_labels(metadata, members, label_names))):
            writer.writerow([int(near_labels[row]), int(near_size[row]), splits[split_index[row]], int(row), ids[index],
                             language_names[language_codes[row]], sources[index], urls[index], int(exact_size[row]),
                             f"{similarity_to_first[index]:.3f}", int(near_redundant[row]), int(near_leak[row]), label])

    # One row per document whose URL occurs more than once.
    if url_labels is not None:
        url_size = group_sizes(url_labels, has_url)
        url_first = first_rows(url_labels, has_url)
        repeated = np.flatnonzero(url_size > 1)
        repeated = repeated[np.lexsort((repeated, url_first[repeated]))]
        firsts = url_first[repeated]
        # Text similarity to the URL's first document; undefined if either text is empty.
        comparable = valid[repeated] & valid[firsts]
        similarity = estimated_jaccard(signatures, firsts, repeated)
        columns = {name: metadata[name].take(pa.array(repeated)).to_pylist() if name in metadata else [None] * len(repeated)
                   for name in ("id", "source", "url")}
        with (args.output_dir / "url_duplicates.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(["url_group", "url_group_size", "split", "row", "id", "language", "source", "url",
                             "near_cluster", "estimated_jaccard_to_first", "redundant", "leak", "top_label"])
            for index, (row, label) in enumerate(zip(repeated, top_labels(metadata, repeated, label_names))):
                writer.writerow([int(firsts[index]), int(url_size[row]), splits[split_index[row]], int(row),
                                 columns["id"][index], language_names[language_codes[row]], columns["source"][index],
                                 columns["url"][index], int(near_labels[row]) if near_size[row] > 1 else "",
                                 f"{similarity[index]:.3f}" if comparable[index] else "", int(url_redundant[row]),
                                 int(url_leak[row]), label])

    # Examples to read: the largest clusters and a random sample.
    sample_rng = np.random.default_rng(args.seed)
    random_clusters = (sample_rng.choice(cluster_ids, size=min(args.examples, len(cluster_ids)), replace=False)
                       if len(cluster_ids) else np.array([], dtype=np.int64))
    chosen = [("largest", int(label)) for label in largest[:args.examples]]
    chosen += [("random", int(label)) for label in random_clusters]
    shown = {label: np.flatnonzero(in_cluster & (near_labels == label))[:3] for _, label in chosen}
    texts = read_texts(chunks, [int(row) for rows_ in shown.values() for row in rows_])
    with (args.output_dir / "examples.jsonl").open("w", encoding="utf-8") as handle:
        for kind, label in chosen:
            # The label is the cluster's first row, so near_size[label] is the cluster's size.
            record = {"kind": kind, "cluster": label, "cluster_size": int(near_size[label]), "members": [
                {"split": splits[split_index[row]], "id": metadata["id"][int(row)].as_py(),
                 "language": language_names[language_codes[row]],
                 "url": metadata["url"][int(row)].as_py() if "url" in metadata else None,
                 "estimated_jaccard_to_first": float(estimated_jaccard(
                     signatures, np.array([shown[label][0]]), np.array([row]))[0]),
                 "text_start": texts[int(row)][:500]}
                for row in shown[label]]}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    for split in splits:
        LOGGER.info("%s: %d documents, %d exact and %d near redundant copies%s", split, ranges[split][1],
                    summary["exact"]["redundant_copies"][split], summary["near"]["redundant_copies"][split],
                    "" if train_index is None or split == TRAIN_SPLIT else
                    f"; {summary['leaks_into_training_split']['near'][split]} near and "
                    f"{summary['leaks_into_training_split']['exact'][split]} exact leaks into {TRAIN_SPLIT}")
    LOGGER.info("Wrote %s (%.0f s)", args.output_dir, time.monotonic() - started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
