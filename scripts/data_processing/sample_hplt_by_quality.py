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

By default, the documents of a source in a bucket are taken from the start of its first
shard, moving on to the next shard only when one runs out, so only the head of each file
is read. With ``--scan-factor N`` (N > 1), up to N times as many documents as the source
needs are read from the start of its files, and the sample is drawn at random among
them; ``--scan-factor 0`` reads the whole source, so the sample is uniform over all of
it. Reading takes about N times as long. The output is shuffled, so documents of
different quality, source and language code are mixed.

A document whose ID is already in the sample of the language is skipped, with a warning
that counts them, and the next document of the same source takes its place (with
``--scan-factor``, the next one drawn). If the source has none left, the next larger
source supplies the difference.

``--method bernoulli`` replaces all of the above with a plain random sample: every
document in every file of the language is kept with the same probability (``--ratio``),
so the sample follows the composition of the full data, with no quotas per code, bucket
or source. Without ``--ratio``, the documents are counted first and the ratio is set to
draw slightly more than ``--sample-size`` documents. The drawn documents are shuffled
and the first ``--sample-size`` of them kept, so the sample has exactly that size and is
a simple random sample of the language. Every file is read in full (twice without
``--ratio``); ``--workers`` reads files in parallel. A document whose ID is already in the
sample is skipped.

Outputs (in ``--output-dir``, ``<language>`` being the lower-cased language name):
    <language>.jsonl            the sampled documents in shuffled order, each with
                                added ``hplt_language_code``, ``hplt_quality_bucket``,
                                ``hplt_source_file`` and ``hplt_source_line`` fields
    <language>.manifest.json    seed, input files, and per-code, per-bucket and
                                per-source counts
"""

from __future__ import annotations

import argparse
import heapq
import io
import json
import logging
import math
import random
import re
import shutil
import subprocess
from collections import Counter
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
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
Result = TypeVar("Result")


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


def iter_candidates(
    paths: Sequence[Path], need: int, scan_factor: int, rng: random.Random
) -> Iterator[tuple[Path, int, bytes]]:
    """Yield a source's documents in the order they are to be taken.

    Args:
        paths: The source's files, in shard order.
        need: Number of documents the source has to supply.
        scan_factor: 1 to yield the documents from the start of the files; N > 1 to
            read up to ``N * need`` documents from the start and yield them in random
            order; 0 to do the same over all documents of the source.
        rng: Random generator for the order.

    Yields:
        The file, the 1-based line number in it, and the raw line of each document.
    """
    if scan_factor == 1:
        yield from iter_head(paths, None)
        return
    # A few spare documents replace those skipped as duplicates of documents already sampled.
    capacity = need + max(10, need // 100)
    window = None if scan_factor == 0 else scan_factor * need
    # Every document gets a uniform random key; the ones with the smallest keys are a uniform
    # sample of the window. Max-heap on the key (negated), so the largest kept key goes first.
    heap: list[tuple[float, int, Path, int, bytes]] = []
    for index, (path, line_number, line) in enumerate(iter_head(paths, window)):
        key = rng.random()
        if len(heap) < capacity:
            heapq.heappush(heap, (-key, index, path, line_number, line))
        elif key < -heap[0][0]:
            heapq.heapreplace(heap, (-key, index, path, line_number, line))
    for _negated_key, _index, path, line_number, line in sorted(heap, key=lambda item: (-item[0], item[1])):
        yield path, line_number, line


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


def selection_description(scan_factor: int) -> str:
    """Describe for the manifest how the documents of each source were chosen."""
    if scan_factor == 1:
        return "first documents of each source in each quality bucket, in shard order"
    if scan_factor == 0:
        return "random documents among all documents of each source in each quality bucket"
    return (
        f"random documents among the first {scan_factor} times as many documents as needed "
        "of each source in each quality bucket, in shard order"
    )


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """Write records as JSONL through a temporary file, so an interrupted run leaves no partial output."""
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary_path.replace(path)


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
            rng = random.Random(f"{args.seed}:{code}:{bucket}:{source}")
            # More than ``need`` documents, as duplicates among them are replaced.
            candidates = iter_candidates(code_files[code][bucket][source], need, args.scan_factor, rng)
            with closing(candidates) as documents:
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
    write_jsonl(output_path, records)

    manifest = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "language": name,
        "input_dir": str(args.input_dir),
        "output": str(output_path),
        "selection": selection_description(args.scan_factor),
        "scan_factor": args.scan_factor,
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


def count_documents(path: Path) -> int:
    """Return the number of documents in a file.

    Raises:
        ValueError: If the file has no documents or a blank line.
    """
    return sum(1 for _ in iter_head([path], None))


def bernoulli_file(path: Path, ratio: float, rng_seed: str) -> tuple[int, list[tuple[int, bytes]]]:
    """Keep each document of a file with probability ``ratio``.

    Returns:
        The number of documents in the file, and the 1-based line number and raw line
        of each kept document.

    Raises:
        ValueError: If the file has no documents or a blank line.
    """
    rng = random.Random(rng_seed)
    documents = 0
    kept: list[tuple[int, bytes]] = []
    for _path, line_number, line in iter_head([path], None):
        documents += 1
        if rng.random() < ratio:
            kept.append((line_number, line))
    return documents, kept


def parallel_map(function: Callable[..., Result], workers: int, *iterables: Iterable[Any]) -> Iterator[Result]:
    """Map ``function`` over the iterables in order, in ``workers`` processes if more than one."""
    if workers == 1:
        yield from map(function, *iterables)
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        yield from pool.map(function, *iterables)


def bernoulli_sample_language(
    name: str, codes: Sequence[str], missing_codes: Sequence[str], args: argparse.Namespace
) -> None:
    """Sample one language by keeping every document with the same probability.

    Args:
        name: Language name, used for the output file names.
        codes: Language codes to sample from; each has a directory in ``args.input_dir``.
        missing_codes: Language codes of the language without a directory, for the manifest.
        args: Parsed command-line arguments.

    Raises:
        ValueError: If the language has fewer documents than ``--sample-size`` (when the
            ratio is derived from it) or a sampled record is unusable.
    """
    id_field: str = args.id_field
    files = [
        (code, bucket, path)
        for code in codes
        for bucket, source_files in find_data_files(args.input_dir / code).items()
        for paths in source_files.values()
        for path in paths
    ]
    paths = [path for _code, _bucket, path in files]
    if args.ratio is not None:
        ratio = args.ratio
    else:
        LOGGER.info("%s: counting the documents of %d files", name, len(files))
        total = sum(parallel_map(count_documents, args.workers, paths))
        if total < args.sample_size and not args.allow_fewer:
            raise ValueError(
                f"{name}: only {total} documents; --sample-size is {args.sample_size}. "
                "Pass --allow-fewer to take all of them."
            )
        # Six standard deviations of the number drawn, so that drawing fewer than
        # --sample-size documents is practically impossible.
        margin = max(100, math.ceil(6 * math.sqrt(args.sample_size)))
        ratio = min(1.0, (args.sample_size + margin) / total)
        LOGGER.info("%s: %d documents; keeping each with probability %.6g", name, total, ratio)

    seeds = [f"{args.seed}:{code}:{path.name}" for code, _bucket, path in files]
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    duplicates: list[str] = []
    per_file: dict[str, dict[str, int]] = {}
    results = parallel_map(bernoulli_file, args.workers, paths, [ratio] * len(paths), seeds)
    for index, ((code, bucket, path), (documents, kept)) in enumerate(zip(files, results), start=1):
        taken = 0
        for line_number, line in kept:
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
            records.append(record)
            taken += 1
        per_file[f"{code}/{path.name}"] = {"documents": documents, "documents_drawn": taken}
        LOGGER.info("%s: [%d/%d] %s/%s: drew %d of %d documents", name, index, len(files), code, path.name, taken, documents)
    if duplicates:
        LOGGER.warning(
            "%s: skipped %d document(s) whose ID was already sampled, e.g. %s",
            name, len(duplicates), "; ".join(duplicates[:3]),
        )
    drawn = len(records)
    if drawn < args.sample_size:
        if not args.allow_fewer:
            raise ValueError(
                f"{name}: drew only {drawn} documents with distinct IDs (ratio {ratio:.6g}); --sample-size is "
                f"{args.sample_size}. Pass a larger --ratio, or --allow-fewer to keep all of them."
            )
        LOGGER.warning("%s: drew only %d documents; keeping all of them", name, drawn)
    # Shuffled, as the files hold the documents grouped by crawl; the first --sample-size
    # of a shuffled random sample are themselves a random sample.
    random.Random(f"{args.seed}:{name}").shuffle(records)
    del records[args.sample_size:]
    output_path, manifest_path = output_paths(args.output_dir, name)
    write_jsonl(output_path, records)

    sampled = Counter((record["hplt_language_code"], record["hplt_source_file"]) for record in records)
    per_bucket: dict[str, dict[str, dict[str, int]]] = {code: {} for code in codes}
    for code, bucket, path in files:
        counts = per_file[f"{code}/{path.name}"]
        counts["documents_sampled"] = sampled[code, path.name]
        bucket_counts = per_bucket[code].setdefault(str(bucket), {"documents": 0, "documents_sampled": 0})
        bucket_counts["documents"] += counts["documents"]
        bucket_counts["documents_sampled"] += counts["documents_sampled"]
    total_documents = sum(counts["documents"] for counts in per_file.values())
    manifest = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "language": name,
        "input_dir": str(args.input_dir),
        "output": str(output_path),
        "selection": "every document of every file kept with the same probability (Bernoulli sampling)",
        "ratio": ratio,
        "ratio_from": "--ratio" if args.ratio is not None else "slightly more than --sample-size / documents",
        "row_order": f"shuffled (seed {args.seed}), then cut to --sample-size",
        "requested_sample_size": args.sample_size,
        "documents_drawn": drawn,
        "sample_size": len(records),
        "documents": total_documents,
        "missing_language_codes": list(missing_codes),
        "id_field": id_field,
        "duplicate_ids_skipped": len(duplicates),
        "buckets": per_bucket,
        "files": per_file,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    LOGGER.info(
        "%s: wrote %d of %d documents (ratio %.6g) to %s", name, len(records), total_documents, ratio, output_path
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
    parser.add_argument("--seed", type=int, default=0, help="Seed for the random choices and the order of the output")
    parser.add_argument(
        "--scan-factor",
        type=int,
        default=1,
        help="Draw each source's documents at random from the first N times as many as it needs "
        "(0: from all of them); 1 takes the first documents (default). Reading takes about N times as long",
    )
    parser.add_argument(
        "--method",
        choices=("stratified", "bernoulli"),
        default="stratified",
        help="stratified: quotas per code, bucket and source (default); bernoulli: keep every document "
        "with the same probability, reading every file in full",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=None,
        help="Bernoulli only: probability of keeping each document "
        "(default: --sample-size divided by the language's document count, which takes an extra pass)",
    )
    parser.add_argument("--workers", type=int, default=1, help="Bernoulli only: files to read in parallel")
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
    if args.scan_factor < 0:
        raise ValueError("--scan-factor must be >= 0")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.method == "bernoulli":
        if args.scan_factor != 1:
            raise ValueError("--scan-factor applies to --method stratified only")
        if args.ratio is not None and not 0.0 < args.ratio <= 1.0:
            raise ValueError("--ratio must be in (0, 1]")
    elif args.ratio is not None or args.workers != 1:
        raise ValueError("--ratio and --workers apply to --method bernoulli only")
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
    sample = bernoulli_sample_language if args.method == "bernoulli" else sample_language
    for name, codes in present.items():
        if codes:
            sample(name, codes, missing[name], args)

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
