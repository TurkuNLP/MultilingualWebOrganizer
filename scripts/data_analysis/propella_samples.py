"""Draw and cache a sample of documents labelled by WebOrganizer and by propella-1.

The WebOrganizer classifier labels in ``data/WO_labelled`` and the propella-1
annotations of Nemotron-CC (``openeurollm/propella-annotations`` on HuggingFace)
share their document IDs, the WARC record IDs, but both are far too large to join
in full. This module joins a sample:

1. A few row groups of the propella parquet shards are read over HTTP. Only the
   requested columns of those row groups are downloaded, a few megabytes each.
   Their documents are the candidates.
2. Both WebOrganizer JSONL files are streamed once, keeping the rows whose ID is a
   candidate. Nothing else is held in memory.
3. The documents found in all three sources are subsampled with a fixed seed and
   written to a JSONL cache, next to a ``.spec.json`` file describing the request.
   A cached file is only reused when its spec matches the request exactly.

Within a shard the propella IDs are in random order, but the shards are not
interchangeable: the documents of the WebOrganizer files are in the first four
shards. Most row groups tested there were almost entirely in the WebOrganizer
files (one only partly and one not at all), and none of the row groups tested in
later shards were. Row groups are therefore chosen at random from
``PROPELLA_SHARDS``, and the spec records how many documents each one matched.

Neither source is clean, and the sample only keeps documents with one unambiguous
record in each. Everything left out is counted, logged as a warning and stored in
the spec:

- Some propella IDs appear twice in a row group (about 0.1% in most row groups
  tested), mostly with different annotations, so it is not known which annotation
  belongs to the text WebOrganizer labelled. These IDs are left out of the
  candidates.
- The WebOrganizer files repeat some rows. A repeated ID with the same label is
  kept once; one with different labels is left out.
- Some documents are in only one of the two WebOrganizer files (and reading only
  the start of the files adds more at the cut-off, because the files are not in
  exactly the same row order). Only documents found in both are matched.

Rows of the WebOrganizer files that are not candidates are checked for the
expected line layout only; candidate rows are parsed and validated in full.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import random
import re
from collections import Counter
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

try:
    from .evaluate_distribution_reference import load_label_names
except ImportError:  # Allows running the script directly.
    from evaluate_distribution_reference import load_label_names

LOGGER = logging.getLogger(__name__)

PROPELLA_REPO = "openeurollm/propella-annotations"
# The propella annotations of Nemotron-CC cover its high-quality "actual" subset.
PROPELLA_DIRECTORY = "data/propella-1-4b/nemotron-cc/high-actual"
PROPELLA_PROPERTIES = ("content_type", "business_sector", "technical_content")
# The shards that contain the documents of data/WO_labelled/WO_MultiSynt_*.jsonl.
PROPELLA_SHARDS = (
    "shard000000.parquet",
    "shard000001.parquet",
    "shard000002.parquet",
    "shard000003.parquet",
)

UUID_PATTERN = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
# Some propella subsets wrap the WARC record ID as "<urn:uuid:...>".
URN_PREFIX = "<urn:uuid:"

# Every row of the WebOrganizer files starts with this, followed by the 36
# character ID. Fixed positions let the scan avoid parsing rows it will not keep.
WO_LINE_PREFIX = b'{"warc_record_id": "'
WO_ID_START = len(WO_LINE_PREFIX)
WO_ID_END = WO_ID_START + 36
WO_FIELDS = {"warc_record_id", "predicted_label"}
PROGRESS_ROWS = 20_000_000
# Rows per table in the packed row positions of PropellaCandidates.
ROW_STRIDE = 2**32


def normalize_record_id(value: str) -> str:
    """Return a WARC record ID as a bare UUID.

    Args:
        value: The ID as stored, either a bare UUID or ``<urn:uuid:...>``.

    Returns:
        The bare UUID.

    Raises:
        ValueError: If the value is not a UUID in either form.
    """
    bare = value[len(URN_PREFIX):-1] if value.startswith(URN_PREFIX) and value.endswith(">") else value
    if not UUID_PATTERN.fullmatch(bare):
        raise ValueError(f"Not a WARC record ID: {value!r}")
    return bare


def choose_row_groups(
    shard_sizes: dict[str, int], num_row_groups: int, seed: int
) -> list[str]:
    """Choose the shard of each row group to read.

    Shards are drawn with replacement, weighted by file size, so that every
    document is roughly equally likely to be chosen without reading the footer of
    every shard first.

    Args:
        shard_sizes: File size in bytes by shard path.
        num_row_groups: Number of row groups to read.
        seed: Seed of the draw.

    Returns:
        One shard path per row group, sorted.
    """
    if not shard_sizes:
        raise ValueError("No parquet shards to choose from")
    if num_row_groups < 1:
        raise ValueError("num_row_groups must be >= 1")
    paths = sorted(shard_sizes)
    rng = random.Random(seed)
    return sorted(rng.choices(paths, weights=[shard_sizes[p] for p in paths], k=num_row_groups))


def _check_column(column: pa.ChunkedArray, name: str) -> None:
    """Check that a property column holds strings or non-empty lists of strings."""
    kind = column.type
    is_list = pa.types.is_list(kind) or pa.types.is_large_list(kind)
    is_string = pa.types.is_string(kind) or pa.types.is_large_string(kind)
    if not is_list and not is_string:
        raise ValueError(f"propella column {name!r} has unsupported type {kind}")
    strings = column
    if is_list:
        if column.null_count or pc.min(pc.list_value_length(column)).as_py() < 1:
            raise ValueError(f"propella column {name!r} has a missing or empty list")
        strings = pc.list_flatten(column)
    if strings.null_count or pc.min(pc.utf8_length(strings)).as_py() < 1:
        raise ValueError(f"propella column {name!r} has a missing or empty value")


@dataclass
class PropellaCandidates:
    """Candidate documents read from the propella annotations.

    The annotations stay in their column tables, and only the IDs become Python
    objects, so that several million candidates fit in memory.

    Attributes:
        tables: One table per row group read, with the property columns.
        rows: Where each candidate is, by ID: ``table number * ROW_STRIDE + row``.
            The IDs are ASCII bytes, the form in which they are looked up while
            scanning the WebOrganizer files.
        row_groups: The row groups read (shard, row group index, rows).
        ambiguous: Number of IDs left out because they appeared more than once.
    """

    tables: list[pa.Table] = field(default_factory=list)
    rows: dict[bytes, int] = field(default_factory=dict)
    row_groups: list[dict[str, Any]] = field(default_factory=list)
    ambiguous: int = 0

    def properties(self, doc_id: str) -> dict[str, Any]:
        """Return the propella properties of one candidate document."""
        table, row = divmod(self.rows[doc_id.encode("ascii")], ROW_STRIDE)
        return {name: self.tables[table][name][row].as_py() for name in self.tables[table].column_names}


def read_propella_candidates(
    repo: str,
    directory: str,
    properties: Sequence[str],
    num_row_groups: int,
    seed: int,
    shards: Sequence[str] | None = None,
) -> PropellaCandidates:
    """Read randomly chosen row groups of the propella annotations.

    Documents whose ID appears more than once in the rows read are ambiguous and
    are not among the candidates.

    Args:
        repo: HuggingFace dataset repository.
        directory: Directory of the parquet shards inside the repository.
        properties: propella properties (columns) to read.
        num_row_groups: Number of row groups to read.
        seed: Seed for choosing shards and row groups.
        shards: File names of the shards to choose from; ``None`` uses every
            shard in the directory.

    Returns:
        The candidate documents.

    Raises:
        ValueError: If a shard is missing, a column is missing or malformed, or a
            document ID is invalid.
    """
    if not properties:
        raise ValueError("At least one propella property is required")
    filesystem = HfFileSystem()
    listing = filesystem.ls(f"datasets/{repo}/{directory}", detail=True)
    shard_sizes = {entry["name"]: entry["size"] for entry in listing if entry["name"].endswith(".parquet")}
    if shards is not None:
        absent = set(shards) - {Path(name).name for name in shard_sizes}
        if absent:
            raise ValueError(f"No such shard(s) in {repo}/{directory}: {sorted(absent)}")
        shard_sizes = {name: size for name, size in shard_sizes.items() if Path(name).name in set(shards)}
    chosen = Counter(choose_row_groups(shard_sizes, num_row_groups, seed))
    rng = random.Random(seed)
    candidates = PropellaCandidates()
    ambiguous: set[bytes] = set()
    for shard, count in sorted(chosen.items()):
        with filesystem.open(shard, "rb") as handle:
            parquet = pq.ParquetFile(handle)
            missing = {"id", *properties} - set(parquet.schema_arrow.names)
            if missing:
                raise ValueError(f"{shard} has no column(s) {sorted(missing)}")
            if count > parquet.metadata.num_row_groups:
                raise ValueError(f"{shard} has fewer than {count} row groups")
            for index in sorted(rng.sample(range(parquet.metadata.num_row_groups), count)):
                table = parquet.read_row_group(index, columns=["id", *properties])
                if not 0 < table.num_rows < ROW_STRIDE:
                    raise ValueError(f"{shard} row group {index} has {table.num_rows} rows")
                for name in properties:
                    _check_column(table[name], name)
                offset = len(candidates.tables) * ROW_STRIDE
                for row, value in enumerate(table["id"].to_pylist()):
                    doc_id = normalize_record_id(value).encode("ascii")
                    if doc_id in candidates.rows:
                        ambiguous.add(doc_id)
                    candidates.rows[doc_id] = offset + row
                candidates.tables.append(table.select(list(properties)))
                candidates.row_groups.append({"shard": Path(shard).name, "row_group": index, "rows": table.num_rows})
                LOGGER.info("Read %d propella rows from %s row group %d", table.num_rows, Path(shard).name, index)
    for doc_id in ambiguous:
        del candidates.rows[doc_id]
    candidates.ambiguous = len(ambiguous)
    if ambiguous:
        LOGGER.warning(
            "Left out %d propella IDs that appear more than once (%d candidates remain)",
            len(ambiguous), len(candidates.rows),
        )
    return candidates


def scan_weborganizer_labels(
    path: Path,
    wanted_ids: Collection[bytes],
    valid_labels: Collection[str],
    max_rows: int | None = None,
) -> tuple[dict[str, str], dict[str, int]]:
    """Stream a WebOrganizer label file and return the labels of the wanted documents.

    A wanted ID that appears again with the same label is kept once. One that
    appears with different labels is ambiguous and is not returned.

    Args:
        path: JSONL file with ``warc_record_id`` and ``predicted_label`` per row.
        wanted_ids: IDs of the documents to keep, as ASCII bytes. A set or a
            dictionary keeps the lookup per row fast.
        valid_labels: The labels the file may contain.
        max_rows: Stop after this many rows; ``None`` reads the whole file.

    Returns:
        The label of each wanted document found, and counts: ``rows_read``,
        ``repeated_rows`` (rows of a wanted ID after its first) and
        ``conflicting_ids_left_out``.

    Raises:
        ValueError: If a row does not have the expected layout, or a kept row has
            unexpected fields or an unknown label.
    """
    if max_rows is not None and max_rows < 1:
        raise ValueError("max_rows must be >= 1 or None")
    valid = set(valid_labels)
    found: dict[str, str] = {}
    conflicting: set[str] = set()
    repeated = 0
    rows = 0
    with path.open("rb", buffering=16 * 1024 * 1024) as handle:
        for rows, line in enumerate(itertools.islice(handle, max_rows), start=1):
            if line[:WO_ID_START] != WO_LINE_PREFIX or line[WO_ID_END:WO_ID_END + 1] != b'"':
                raise ValueError(f"Unexpected row layout in {path}:{rows}: {line[:80]!r}")
            if line[WO_ID_START:WO_ID_END] in wanted_ids:
                record = json.loads(line)
                if not isinstance(record, dict) or set(record) != WO_FIELDS:
                    raise ValueError(f"Unexpected fields in {path}:{rows}: {line[:200]!r}")
                doc_id, label = record["warc_record_id"], record["predicted_label"]
                if label not in valid:
                    raise ValueError(f"Unknown label {label!r} in {path}:{rows}")
                if doc_id in found:
                    repeated += 1
                    if found[doc_id] != label:
                        conflicting.add(doc_id)
                found.setdefault(doc_id, label)
            if rows % PROGRESS_ROWS == 0:
                LOGGER.info("%s: %d rows read, %d documents found", path.name, rows, len(found))
    if rows == 0:
        raise ValueError(f"{path} is empty")
    for doc_id in conflicting:
        del found[doc_id]
    LOGGER.info("%s: %d rows read, %d documents found", path.name, rows, len(found))
    if repeated:
        LOGGER.warning(
            "%s: %d repeated rows of wanted documents; left out %d documents with conflicting labels",
            path.name, repeated, len(conflicting),
        )
    return found, {"rows_read": rows, "repeated_rows": repeated, "conflicting_ids_left_out": len(conflicting)}


def load_matched_sample(
    cache_path: Path,
    topics_path: Path,
    formats_path: Path,
    topic_labels: Collection[str],
    format_labels: Collection[str],
    *,
    sample_size: int | None = 10_000,
    num_row_groups: int = 8,
    seed: int = 0,
    wo_max_rows: int | None = None,
    properties: Sequence[str] = PROPELLA_PROPERTIES,
    shards: Sequence[str] | None = PROPELLA_SHARDS,
    repo: str = PROPELLA_REPO,
    directory: str = PROPELLA_DIRECTORY,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return documents labelled by both WebOrganizer classifiers and by propella-1.

    The sample is read from ``cache_path`` if it was made for the same request,
    and drawn and cached otherwise.

    Args:
        cache_path: JSONL cache file; a ``.spec.json`` file is written next to it.
        topics_path: WebOrganizer topic labels (JSONL).
        formats_path: WebOrganizer format labels (JSONL), with the same documents.
        topic_labels: The labels the topic file may contain.
        format_labels: The labels the format file may contain.
        sample_size: Documents to return; ``None`` returns every matched document.
        num_row_groups: propella row groups to read as candidates.
        seed: Seed for choosing row groups and for subsampling.
        wo_max_rows: Only read this many rows of each WebOrganizer file; ``None``
            reads them in full.
        properties: propella properties to include.
        shards: File names of the propella shards to choose row groups from;
            ``None`` uses every shard.
        repo: HuggingFace dataset repository of the propella annotations.
        directory: Directory of the parquet shards inside the repository.

    Returns:
        The sample, one record per document with ``id``, ``wo_topic``,
        ``wo_format`` and the propella properties, and the spec of the sample:
        the request, the row groups read with the number of documents matched in
        each, and the counts of candidates, matches and records left out.

    Raises:
        ValueError: If the cache was made for a different request, an input is
            malformed, or fewer than ``sample_size`` documents were matched.
    """
    if sample_size is not None and sample_size < 1:
        raise ValueError("sample_size must be >= 1 or None")
    request = {
        "repo": repo,
        "directory": directory,
        "properties": list(properties),
        "shards": None if shards is None else sorted(shards),
        "num_row_groups": num_row_groups,
        "seed": seed,
        "sample_size": sample_size,
        "wo_max_rows": wo_max_rows,
        # Name and size instead of the path, which differs between Roihu and a mount.
        "wo_files": [{"name": p.name, "size_bytes": p.stat().st_size} for p in (topics_path, formats_path)],
    }
    spec_path = cache_path.with_suffix(".spec.json")
    if cache_path.exists():
        spec = json.loads(spec_path.read_text(encoding="utf-8")) if spec_path.exists() else None
        if spec is None or spec.get("request") != request:
            raise ValueError(
                f"{cache_path} was cached for a different request; delete it and its "
                ".spec.json file to draw the sample again"
            )
        with cache_path.open("r", encoding="utf-8") as handle:
            sample = [json.loads(line) for line in handle]
        if len(sample) != spec["sampled"]:
            raise ValueError(f"{cache_path} has {len(sample)} rows but its spec says {spec['sampled']}")
        LOGGER.info("Loaded %d cached documents from %s", len(sample), cache_path)
        return sample, spec

    candidates = read_propella_candidates(repo, directory, properties, num_row_groups, seed, shards)
    topics, topic_counts = scan_weborganizer_labels(topics_path, candidates.rows, topic_labels, wo_max_rows)
    formats, format_counts = scan_weborganizer_labels(formats_path, candidates.rows, format_labels, wo_max_rows)
    matched = sorted(topics.keys() & formats.keys())
    topic_counts.update(candidates_found=len(topics), not_in_other_file=len(topics) - len(matched))
    format_counts.update(candidates_found=len(formats), not_in_other_file=len(formats) - len(matched))
    if len(topics) != len(matched) or len(formats) != len(matched):
        LOGGER.warning(
            "Left out %d candidates found only in %s and %d found only in %s",
            len(topics) - len(matched), topics_path.name, len(formats) - len(matched), formats_path.name,
        )
    if sample_size is not None and len(matched) < sample_size:
        needed = math.ceil(1.2 * num_row_groups * sample_size / max(len(matched), 1))
        raise ValueError(
            f"Only {len(matched)} of {len(candidates.rows)} propella candidates are in both WebOrganizer "
            f"files ({topic_counts['rows_read']} rows read); lower sample_size or increase "
            f"num_row_groups, to about {needed} at this match rate"
        )
    matched_per_table = Counter(candidates.rows[doc_id.encode("ascii")] // ROW_STRIDE for doc_id in matched)
    for number, row_group in enumerate(candidates.row_groups):
        row_group["matched"] = matched_per_table[number]
    chosen = matched if sample_size is None else random.Random(seed).sample(matched, sample_size)
    sample = [
        {"id": doc_id, "wo_topic": topics[doc_id], "wo_format": formats[doc_id], **candidates.properties(doc_id)}
        for doc_id in chosen
    ]
    spec = {
        "request": request,
        "row_groups": candidates.row_groups,
        "candidates": len(candidates.rows),
        "ambiguous_ids_left_out": candidates.ambiguous,
        "weborganizer": {"topics": topic_counts, "formats": format_counts},
        "matched": len(matched),
        "sampled": len(sample),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in sample:
            handle.write(json.dumps(record) + "\n")
    temporary.replace(cache_path)
    spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    LOGGER.info(
        "Matched %d of %d candidates; wrote %d documents to %s",
        len(matched), len(candidates.rows), len(sample), cache_path,
    )
    return sample, spec


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Draw and cache a sample of documents labelled by WebOrganizer and by propella-1."
    )
    parser.add_argument("--output", type=Path, required=True, help="JSONL cache file to write")
    parser.add_argument("--topics", type=Path, required=True, help="WebOrganizer topic labels (JSONL)")
    parser.add_argument("--formats", type=Path, required=True, help="WebOrganizer format labels (JSONL)")
    parser.add_argument("--topic-labels", type=Path, required=True, help="Label YAML of the topics (topics.yaml)")
    parser.add_argument("--format-labels", type=Path, required=True, help="Label YAML of the formats (formats.yaml)")
    parser.add_argument("--sample-size", type=int, default=10_000, help="Documents to sample")
    parser.add_argument("--num-row-groups", type=int, default=8, help="propella row groups to read")
    parser.add_argument(
        "--shards",
        nargs="+",
        default=list(PROPELLA_SHARDS),
        help="propella shards to choose row groups from (default: those with the WebOrganizer documents)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--wo-max-rows", type=int, help="Only read this many rows of each WebOrganizer file"
    )
    parser.add_argument("--properties", nargs="+", default=list(PROPELLA_PROPERTIES), help="propella properties")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default="INFO",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Draw the sample, or confirm that the cached one matches the request."""
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    # huggingface_hub logs every HTTP request at INFO level.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    _, topic_labels = load_label_names(args.topic_labels)
    _, format_labels = load_label_names(args.format_labels)
    load_matched_sample(
        args.output,
        args.topics,
        args.formats,
        topic_labels,
        format_labels,
        sample_size=args.sample_size,
        num_row_groups=args.num_row_groups,
        seed=args.seed,
        wo_max_rows=args.wo_max_rows,
        properties=args.properties,
        shards=args.shards,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
