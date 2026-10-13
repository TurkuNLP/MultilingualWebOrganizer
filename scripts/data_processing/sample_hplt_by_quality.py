"""Sample documents per language from HPLT, stratified by document quality bucket.

The languages come from a file with one ``<Language>: <code> [<code> ...]`` line per
language. Each language code is a directory holding files named
``<bucket>_<shard>.jsonl.zst``, where ``<bucket>`` is the approximate document quality.
Large languages also have the source of the documents in the name,
``<bucket>_<shard>.<source>.jsonl.zst``.

The sample size is per language: it is split evenly over the language's codes, the
share of a code is split evenly over the quality buckets that code has, and the share
of a bucket is split evenly over the sources that bucket has. A code, bucket or source
with fewer documents than its share contributes all of them, and the shortfall is
split evenly over the larger ones, so every language ends up with exactly
``--sample-size`` documents. A language code without a directory is skipped with a
warning, and so is a language none of whose codes has one. A language whose output
file already exists is also skipped with a warning, unless ``--overwrite`` is given.

The documents of a source in a bucket are taken from the start of its first shard,
moving on to the next shard only when one runs out, so only the head of each file is
read. The output is shuffled, so documents of different quality, source and language
code are mixed.

A document whose ID is already in the sample of the language is skipped, with a warning
that counts them, and the next document of the same source takes its place. If the
source has none left, the next larger source supplies the difference.

Outputs (in ``--output-dir``, ``<language>`` being the lower-cased language name):
    <language>.jsonl            the sampled documents in shuffled order, each with
                                added ``hplt_language_code``, ``hplt_quality_bucket``,
                                ``hplt_source_file`` and ``hplt_source_line`` fields
    <language>.manifest.json    seed, input files, and per-code, per-bucket and
                                per-source counts
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import random
import re
import shutil
import subprocess
from collections.abc import Hashable, Iterator, Mapping, Sequence
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, TypeVar

# Decompress with the standard library where it can (Python >= 3.14), else with the
# zstandard package, else with the zstd command-line tool.
try:
    from compression import zstd as stdlib_zstd
except ImportError:
    stdlib_zstd = None
try:
    import zstandard  # type: ignore
except ImportError:
    zstandard = None

LOGGER = logging.getLogger(__name__)

FILE_PATTERN = re.compile(r"^(?P<bucket>\d+)_(?P<shard>\d+)(?:\.(?P<source>[A-Za-z0-9]+))?\.jsonl\.zst$")
FILE_NAME_SCHEME = "<bucket>_<shard>[.<source>].jsonl.zst"
PROVENANCE_FIELDS = ("hplt_language_code", "hplt_quality_bucket", "hplt_source_file", "hplt_source_line")
# Accept frames written with a long-distance matching window (zstd --long=31).
ZSTD_WINDOW_LOG_MAX = 31

Key = TypeVar("Key", bound=Hashable)


def zstd_backend() -> str:
    """Return the name of the zstandard decompressor in use.

    Raises:
        RuntimeError: If no decompressor is available.
    """
    if stdlib_zstd is not None:
        return "compression.zstd (standard library)"
    if zstandard is not None:
        return "zstandard package"
    if shutil.which("zstd") is not None:
        return "zstd command-line tool"
    raise RuntimeError("No zstandard decompressor: need Python >= 3.14, the zstandard package, or zstd on PATH")


@contextmanager
def open_zst(path: Path) -> Iterator[BinaryIO]:
    """Open a zstandard-compressed file as a buffered binary stream of its content.

    Raises:
        OSError: If the zstd command-line tool fails, e.g. on a cut-off file.
    """
    if stdlib_zstd is not None:
        options = {stdlib_zstd.DecompressionParameter.window_log_max: ZSTD_WINDOW_LOG_MAX}
        with stdlib_zstd.open(path, "rb", options=options) as handle:
            yield handle
        return
    if zstandard is not None:
        decompressor = zstandard.ZstdDecompressor(max_window_size=2**ZSTD_WINDOW_LOG_MAX)
        # Without read_across_frames the reader stops silently after the first frame.
        reader = decompressor.stream_reader(path.open("rb"), read_across_frames=True, closefd=True)
        with io.BufferedReader(reader) as handle:
            yield handle
        return
    command = ["zstd", "--decompress", "--stdout", f"--long={ZSTD_WINDOW_LOG_MAX}", "--", str(path)]
    with subprocess.Popen(command, stdout=subprocess.PIPE) as process:
        if process.stdout is None:
            raise RuntimeError("zstd was started without an output pipe")
        try:
            yield process.stdout
        except BaseException:
            process.kill()
            raise
        if process.stdout.read(1):
            # The caller stopped before the end of the file.
            process.kill()
        elif process.wait() != 0:
            raise OSError(f"zstd exited with code {process.returncode} on {path}")


def iter_head(paths: Sequence[Path], limit: int | None) -> Iterator[tuple[Path, int, bytes]]:
    """Yield the first ``limit`` documents of a list of files, or all of them if it has fewer.

    The files are read in order, and a file is opened only when the ones before it
    hold fewer than ``limit`` documents together.

    Args:
        paths: The files, in reading order.
        limit: Maximum number of documents to yield, or ``None`` for all of them.

    Yields:
        The file, the 1-based line number in it, and the raw line of each document.

    Raises:
        ValueError: If a file that is read has no documents or a blank line.
    """
    remaining = limit
    for path in paths:
        if remaining == 0:
            return
        line_number = 0
        with open_zst(path) as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.isspace():
                    raise ValueError(f"{path}:{line_number}: blank line")
                yield path, line_number, line
                if remaining is not None:
                    remaining -= 1
                    if remaining == 0:
                        break
        if line_number == 0:
            raise ValueError(f"{path}: no documents")


def parse_record(line: bytes, source: str) -> dict[str, Any]:
    """Parse one JSONL line into a record.

    Args:
        line: The raw line.
        source: ``file:line`` location for error messages.

    Raises:
        ValueError: If the line is not a UTF-8 encoded JSON object.
    """
    try:
        record = json.loads(line)
    except ValueError as exc:  # JSONDecodeError and UnicodeDecodeError
        raise ValueError(f"{source}: invalid JSON: {exc}") from exc
    if not isinstance(record, dict):
        raise ValueError(f"{source}: record must be a JSON object, got {type(record).__name__}")
    return record


def read_language_codes(path: Path) -> dict[str, list[str]]:
    """Read the language codes per language name from a ``<Language>: <code> ...`` file.

    Raises:
        ValueError: If a line is malformed, or a language name or code repeats.
    """
    languages: dict[str, list[str]] = {}
    seen_codes: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            name, separator, code_list = line.partition(":")
            name, codes = name.strip(), code_list.split()
            if not separator or not name or not codes:
                raise ValueError(f"{path}:{line_number}: expected '<Language>: <code> [<code> ...]'")
            if name in languages:
                raise ValueError(f"{path}:{line_number}: duplicate language {name!r}")
            repeated = [code for code in codes if code in seen_codes or codes.count(code) > 1]
            if repeated:
                raise ValueError(f"{path}:{line_number}: language code(s) listed more than once: {', '.join(repeated)}")
            seen_codes.update(codes)
            languages[name] = codes
    if not languages:
        raise ValueError(f"{path}: no languages")
    return languages


def find_data_files(language_dir: Path) -> dict[int, dict[str, list[Path]]]:
    """Return the files of a language code directory per quality bucket and source, in shard order.

    The source of files without one in their name is the empty string.

    Raises:
        FileNotFoundError: If the directory has no data files.
        ValueError: If a ``.jsonl.zst`` file does not follow the naming scheme.
    """
    shards: dict[int, dict[str, list[tuple[int, Path]]]] = {}
    for path in sorted(language_dir.iterdir()):
        match = FILE_PATTERN.match(path.name)
        if match is None:
            if path.name.endswith(".jsonl.zst"):
                raise ValueError(f"Unexpected data file name, expected {FILE_NAME_SCHEME}: {path}")
            # The directories also hold metadata, such as counts.json and .urls.zst.
            LOGGER.debug("Ignoring %s", path)
            continue
        sources = shards.setdefault(int(match["bucket"]), {})
        sources.setdefault(match["source"] or "", []).append((int(match["shard"]), path))
    if not shards:
        raise FileNotFoundError(f"No {FILE_NAME_SCHEME} files in {language_dir}")
    return {
        bucket: {source: [path for _shard, path in sorted(shards[bucket][source])] for source in sorted(shards[bucket])}
        for bucket in sorted(shards)
    }


def allocate_quotas(available: Mapping[Key, int], total: int) -> dict[Key, int]:
    """Split ``total`` evenly over strata, moving the shortfall of small strata to larger ones.

    Args:
        available: Number of documents available per stratum.
        total: Number of documents to allocate.

    Returns:
        The number of documents to take per stratum, summing to ``total``.

    Raises:
        ValueError: If fewer than ``total`` documents are available.
    """
    if sum(available.values()) < total:
        raise ValueError(f"Cannot allocate {total} documents from {sum(available.values())} available")
    quotas: dict[Key, int] = {}
    remaining = total
    # Smallest strata first, so what they cannot fill raises the share of the rest.
    order = sorted(available, key=lambda stratum: (available[stratum], stratum))
    for position, stratum in enumerate(order):
        share = remaining // (len(order) - position)
        quotas[stratum] = min(available[stratum], share)
        remaining -= quotas[stratum]
    return {stratum: quotas[stratum] for stratum in available}


def output_paths(output_dir: Path, name: str) -> tuple[Path, Path]:
    """Return the document and manifest paths of a language."""
    stem = "_".join(name.lower().split())
    return output_dir / f"{stem}.jsonl", output_dir / f"{stem}.manifest.json"


def sample_language(name: str, codes: Sequence[str], missing_codes: Sequence[str], args: argparse.Namespace) -> None:
    """Sample one language over its codes and write its documents and manifest.

    Args:
        name: Language name, used for the output file names.
        codes: Language codes to sample from; each has a directory in ``args.input_dir``.
        missing_codes: Language codes of the language without a directory, for the manifest.
        args: Parsed command-line arguments.

    Raises:
        ValueError: If the language has too few documents with distinct IDs or a sampled
            record is unusable.
    """
    id_field: str = args.id_field
    code_files = {code: find_data_files(args.input_dir / code) for code in codes}
    # One source of one bucket may have to supply the whole sample, so count up to
    # --sample-size documents in each; the documents are read once the quotas are known.
    available = {
        code: {
            bucket: {source: sum(1 for _ in iter_head(paths, args.sample_size)) for source, paths in source_files.items()}
            for bucket, source_files in bucket_files.items()
        }
        for code, bucket_files in code_files.items()
    }
    bucket_available = {
        code: {bucket: sum(counts.values()) for bucket, counts in available[code].items()} for code in codes
    }
    code_available = {code: sum(bucket_available[code].values()) for code in codes}
    total_available = sum(code_available.values())
    target = args.sample_size
    if total_available < target:
        if not args.allow_fewer:
            raise ValueError(
                f"{name}: only {total_available} documents available; --sample-size is {target}. "
                "Pass --allow-fewer to take all of them."
            )
        LOGGER.warning("%s: only %d documents available; taking all of them", name, total_available)
        target = total_available
    # First over the language codes, then over the quality buckets of each code, then
    # over the sources of each bucket.
    code_quotas = allocate_quotas(code_available, target)
    bucket_quotas = {code: allocate_quotas(bucket_available[code], code_quotas[code]) for code in codes}
    quotas = {
        code: {
            bucket: allocate_quotas(available[code][bucket], bucket_quota)
            for bucket, bucket_quota in bucket_quotas[code].items()
        }
        for code in codes
    }

    strata = [(code, bucket, source) for code in codes for bucket in quotas[code] for source in quotas[code][bucket]]
    stratum_records: dict[tuple[str, int, str], list[dict[str, Any]]] = {stratum: [] for stratum in strata}
    seen_ids: set[str] = set()
    duplicates: list[str] = []
    # What a source cannot supply because of skipped duplicates is carried over to the
    # next one. Smallest sources first, so that it ends up with one that has documents left.
    carry = 0
    for stratum in sorted(strata, key=lambda stratum: (available[stratum[0]][stratum[1]][stratum[2]], stratum)):
        code, bucket, source = stratum
        need = quotas[code][bucket][source] + carry
        taken = stratum_records[stratum]
        if need:
            # Not limited to ``need`` documents, as duplicates among them are replaced.
            with closing(iter_head(code_files[code][bucket][source], None)) as documents:
                for path, line_number, line in documents:
                    location = f"{path}:{line_number}"
                    record = parse_record(line, location)
                    document_id = record.get(id_field)
                    if not isinstance(document_id, str) or not document_id:
                        raise ValueError(f"{location}: field {id_field!r} must be a non-empty string")
                    if document_id in seen_ids:
                        duplicates.append(f"{document_id} ({location})")
                        continue
                    seen_ids.add(document_id)
                    clashes = [field for field in PROVENANCE_FIELDS if field in record]
                    if clashes:
                        raise ValueError(f"{location}: record already has field(s) {', '.join(clashes)}")
                    record.update(zip(PROVENANCE_FIELDS, (code, bucket, path.name, line_number)))
                    taken.append(record)
                    if len(taken) == need:
                        break
        carry = need - len(taken)
    if duplicates:
        LOGGER.warning(
            "%s: skipped %d document(s) whose ID was already sampled, e.g. %s",
            name, len(duplicates), "; ".join(duplicates[:3]),
        )
    if carry:
        if not args.allow_fewer:
            raise ValueError(
                f"{name}: only {target - carry} documents with distinct IDs available; "
                f"--sample-size is {args.sample_size}. Pass --allow-fewer to take all of them."
            )
        LOGGER.warning("%s: only %d documents with distinct IDs available; taking all of them", name, target - carry)
    # In the order of the codes, buckets and sources, whatever order they were read in.
    records = [record for stratum in strata for record in stratum_records[stratum]]
    sampled = {
        code: {
            bucket: {source: len(stratum_records[code, bucket, source]) for source in quotas[code][bucket]}
            for bucket in quotas[code]
        }
        for code in codes
    }
    bucket_sampled = {code: {bucket: sum(counts.values()) for bucket, counts in sampled[code].items()} for code in codes}
    code_sampled = {code: sum(bucket_sampled[code].values()) for code in codes}
    # Shuffled, so that documents of different quality, source and language code are mixed.
    random.Random(f"{args.seed}:{name}").shuffle(records)

    output_path, manifest_path = output_paths(args.output_dir, name)
    # Write to a temporary file first, so an interrupted run leaves no partial output.
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary_path.replace(output_path)

    manifest = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "language": name,
        "input_dir": str(args.input_dir),
        "output": str(output_path),
        "selection": "first documents of each source in each quality bucket, in shard order",
        "row_order": f"shuffled (seed {args.seed})",
        "requested_sample_size": args.sample_size,
        "sample_size": len(records),
        "missing_language_codes": list(missing_codes),
        "id_field": id_field,
        "duplicate_ids_skipped": len(duplicates),
        "language_codes": {
            code: {
                "documents_sampled": code_sampled[code],
                "buckets": {
                    str(bucket): {
                        "documents_sampled": bucket_sampled[code][bucket],
                        # Keyed by source; "" for files without a source in their name.
                        "sources": {
                            source: {
                                "files": [path.name for path in paths],
                                # Counting stops at the requested sample size.
                                "documents_counted": available[code][bucket][source],
                                "documents_sampled": sampled[code][bucket][source],
                            }
                            for source, paths in source_files.items()
                        },
                    }
                    for bucket, source_files in code_files[code].items()
                },
            }
            for code in codes
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("%s: wrote %d documents to %s", name, len(records), output_path)
    for code in codes:
        per_source: dict[str, int] = {}
        for source_counts in sampled[code].values():
            for source, count in source_counts.items():
                per_source[source] = per_source.get(source, 0) + count
        LOGGER.info(
            "%s: %s: %d documents (per bucket: %s; per source: %s)",
            name, code, code_sampled[code],
            ", ".join(f"{bucket}: {count}" for bucket, count in bucket_sampled[code].items()),
            ", ".join(f"{source or 'unnamed'}: {count}" for source, count in sorted(per_source.items())),
        )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description="Sample HPLT documents per language, stratified by quality bucket.")
    parser.add_argument(
        "--languages-file",
        type=Path,
        default=Path("data/oellm_languages.txt"),
        help="File with one '<Language>: <code> [<code> ...]' line per language",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=None,
        help="Language names from --languages-file to sample, e.g. Swedish Norwegian (default: all of them)",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/appl/local/openeurollm/training/catalogue/hplt/4.0/noisy"),
        help=f"Directory with one subdirectory of {FILE_NAME_SCHEME} files per language code",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/hplt_noisy_sample"))
    parser.add_argument("--sample-size", type=int, default=10_000, help="Documents to sample per language")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the order of the output")
    parser.add_argument(
        "--id-field",
        default="id",
        help="Record field with the document ID; a document whose ID is already sampled is skipped",
    )
    parser.add_argument(
        "--allow-fewer",
        action="store_true",
        help="Take all documents of a language that has fewer than --sample-size with distinct IDs instead of failing",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Sample languages again whose output file exists, replacing it, instead of skipping them",
    )
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Sample every requested language."""
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s [%(levelname)s] %(message)s")
    if args.sample_size < 1:
        raise ValueError("--sample-size must be >= 1")
    LOGGER.info("Decompressing with %s", zstd_backend())

    language_codes = read_language_codes(args.languages_file)
    names = list(language_codes) if args.languages is None else args.languages
    unknown = [name for name in names if name not in language_codes]
    if unknown:
        raise ValueError(f"Not in {args.languages_file}: {', '.join(unknown)}. Known: {', '.join(language_codes)}")
    if len(set(names)) != len(names):
        raise ValueError("--languages contains duplicates")
    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    # Check everything that can be checked before the first language is read.
    present: dict[str, list[str]] = {}
    missing: dict[str, list[str]] = {}
    existing: list[str] = []
    for name in names:
        output_path, _manifest_path = output_paths(args.output_dir, name)
        if output_path.exists() and not args.overwrite:
            LOGGER.warning("%s: skipped, %s exists; pass --overwrite to replace it", name, output_path)
            existing.append(name)
            continue
        present[name] = [code for code in language_codes[name] if (args.input_dir / code).is_dir()]
        missing[name] = [code for code in language_codes[name] if code not in present[name]]
        if not present[name]:
            LOGGER.warning("%s: skipped, no directory in %s for %s", name, args.input_dir, ", ".join(missing[name]))
            continue
        if missing[name]:
            LOGGER.warning(
                "%s: no directory in %s for %s; sampling only from %s",
                name, args.input_dir, ", ".join(missing[name]), ", ".join(present[name]),
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, codes in present.items():
        if codes:
            sample_language(name, codes, missing[name], args)

    # Repeat the warnings at the end of the log, where they are easy to find.
    if existing:
        LOGGER.warning("Skipped %d language(s) with existing output: %s", len(existing), ", ".join(existing))
    skipped = [name for name, codes in present.items() if not codes]
    if skipped:
        LOGGER.warning("Skipped %d language(s) without data: %s", len(skipped), ", ".join(skipped))
    partial = {name: codes for name, codes in missing.items() if codes and present[name]}
    if partial:
        LOGGER.warning(
            "Sampled without missing language codes: %s",
            "; ".join(f"{name}: {', '.join(codes)}" for name, codes in partial.items()),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
