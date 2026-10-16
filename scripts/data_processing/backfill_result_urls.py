"""Add the document URL to label distribution results written before records stored it.

Since 2026-10-14 every result record has a ``url`` field: the URL shown to the model in
the prompt, or ``null``. Results written earlier lack it although the prompt may have
contained it. This script recovers it from the input file the results were made from,
with the same function the labelling scripts use to put the URL into the prompt
(``labeling_common.document_url``), so the stored value is exactly the prompt's.

Each result file ``<name>.jsonl`` is matched to ``<input-dir>/<name>.jsonl``. Every
record is checked before anything is written:

* its document ID must be in the input file, exactly once;
* its text must equal the input document's text, so it is the same document;
* a record that already has a URL must have the one the input gives (records written
  by a resumed run after the fix already have it); these are left as they are.

The updated file is written next to the original and swapped in atomically. The
original is kept as ``<name>.jsonl.before-url-backfill`` unless ``--no-backup`` is
given. If the file changes while it is being processed (a labelling job is still
appending to it), it is left untouched and the script stops with an error, so run it
only on results no job is writing to.

Example (on a Roihu login node, from the repository root):
    module load python-vllm/0.19.1
    python scripts/data_processing/backfill_result_urls.py --input-dir data/hplt4 \
        results/distribution/Qwen3.8-27B-hplt-annotations/topic
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

try:
    from .distribution_records import SavedDistributionRecord
    from .labeling_common import document_url, stream_jsonl
except ImportError:
    from distribution_records import SavedDistributionRecord
    from labeling_common import document_url, stream_jsonl

LOGGER = logging.getLogger(__name__)
BACKUP_SUFFIX = ".before-url-backfill"


@dataclass(frozen=True)
class BackfillSummary:
    """What happened to one result file.

    Attributes:
        records: Records in the file.
        filled: Records that got a URL from the input.
        without_url: Records whose input document has no URL (stored as null).
        already_set: Records that already had the URL.
    """

    records: int
    filled: int
    without_url: int
    already_set: int


def find_result_files(paths: Sequence[Path]) -> list[Path]:
    """Expand result paths into a sorted list of JSONL files.

    Raises:
        FileNotFoundError: If a path doesn't exist or a directory has no JSONL files.
    """
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            found = sorted(path.glob("*.jsonl"))
            if not found:
                raise FileNotFoundError(f"No JSONL files in {path}")
            files.extend(found)
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(path)
    return files


def input_documents(input_path: Path, wanted: set[str]) -> dict[str, tuple[str | None, str]]:
    """Return document ID -> (URL as the prompt showed it, text) for the wanted IDs.

    The input is streamed and validated exactly as the labelling scripts read it.

    Raises:
        ValueError: If a wanted document ID occurs more than once in the input.
    """
    documents: dict[str, tuple[str | None, str]] = {}
    for line_number, document in stream_jsonl(input_path):
        if document.doc_id not in wanted:
            continue
        if document.doc_id in documents:
            raise ValueError(f"{input_path}:{line_number}: document {document.doc_id!r} occurs more than once")
        documents[document.doc_id] = (document_url(document), document.text)
    return documents


def backfill_file(result_path: Path, input_path: Path, *, dry_run: bool, backup: bool) -> BackfillSummary:
    """Add URLs to one result file, checking every record against the input first.

    Raises:
        ValueError: If a record's document is missing from the input, its text differs,
            it has a different URL already, or a record is malformed.
        RuntimeError: If the result file changed while it was processed.
    """
    before = result_path.stat()
    with result_path.open("r", encoding="utf-8") as handle:
        raw_records = [json.loads(line) for line in handle if line.strip()]
    ids = [record["id"] for record in raw_records]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{result_path}: duplicate document IDs")
    documents = input_documents(input_path, set(ids))
    missing = [doc_id for doc_id in ids if doc_id not in documents]
    if missing:
        raise ValueError(f"{result_path}: {len(missing)} documents are not in {input_path}, e.g. {missing[:3]}")

    updated, filled, without_url, already_set = [], 0, 0, 0
    for line_number, record in enumerate(raw_records, start=1):
        url, text = documents[record["id"]]
        where = f"{result_path}:{line_number}"
        if record.get("text") != text:
            raise ValueError(f"{where}: text differs from the input document {record['id']!r}")
        if "url" in record:
            if record["url"] != url:
                raise ValueError(f"{where}: stored URL {record['url']!r} differs from the input's {url!r}")
            already_set += 1
        else:
            filled += url is not None
            without_url += url is None
        rest = {key: value for key, value in record.items() if key not in ("id", "text", "url")}
        # url goes after text, where the labelling scripts write it; then validate the record.
        completed = {"id": record["id"], "text": record["text"], "url": url, **rest}
        updated.append(SavedDistributionRecord.model_validate(completed).model_dump_json())

    summary = BackfillSummary(len(raw_records), filled, without_url, already_set)
    if dry_run or (filled + without_url) == 0:
        return summary

    temporary = result_path.with_name(f".{result_path.name}.url-backfill.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(updated) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    shutil.copymode(result_path, temporary)   # keep the original's permissions (e.g. group write)
    after = result_path.stat()
    if (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
        temporary.unlink()
        raise RuntimeError(f"{result_path} changed while it was processed; is a labelling job still writing it?")
    if backup:
        backup_path = result_path.with_name(result_path.name + BACKUP_SUFFIX)
        if backup_path.exists():
            temporary.unlink()
            raise FileExistsError(f"Backup {backup_path} already exists; move it away first")
        try:
            os.link(result_path, backup_path)   # a second name for the original, no copy needed
        except OSError:
            shutil.copy2(result_path, backup_path)   # file systems without hard links
    os.replace(temporary, result_path)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description="Add prompt URLs to label distribution results that lack them.")
    parser.add_argument("results", type=Path, nargs="+", help="Result JSONL files, or directories of them")
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Directory with the input JSONL files, named like the result files")
    parser.add_argument("--dry-run", action="store_true", help="Check and report, but don't change any file")
    parser.add_argument("--no-backup", action="store_true", help=f"Don't keep the original as *{BACKUP_SUFFIX}")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Backfill URLs into every given result file."""
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s [%(levelname)s] %(message)s")
    files = find_result_files(args.results)
    for result_path in files:
        input_path = args.input_dir / result_path.name
        if not input_path.is_file():
            raise FileNotFoundError(f"No input file {input_path} for {result_path}")
        summary = backfill_file(result_path, input_path, dry_run=args.dry_run, backup=not args.no_backup)
        LOGGER.info(
            "%s%s: %d records, %d URLs added, %d without a URL in the input, %d already had one",
            "[dry run] " if args.dry_run else "", result_path, summary.records, summary.filled,
            summary.without_url, summary.already_set,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
