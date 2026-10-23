"""Collect hplt-sampler runs into one random sample file per language.

hplt-sampler (https://github.com/TurkuNLP/hplt-sampler) makes one run per HPLT language
code, in ``results-v4-<code>-all/`` (clean) or ``results-v4-noisy-<code>-all/`` (noisy)
under its own directory; ``hpc/lumi/submit_hplt_sampler.sh`` submits them. Each run's
``samples.jsonl`` is a shuffled simple random sample of the code.

For each language of the languages file, this script checks its runs and writes
``<output-dir>/<language>.jsonl``, which is where the label reuse and labelling scripts
read the samples:

- A language with one code gets that run's sample, line for line.
- A language with several codes (Serbian in Cyrillic and Latin) gets ``--n-samples``
  documents split over its codes in proportion to their document counts in the
  collection, each code's share taken from the start of its shuffled sample, and the
  result shuffled again. This is a stratified sample with proportional allocation,
  as hplt-sampler itself does across shards.

A run must be complete and shuffled: its ``samples.jsonl`` exists, the per-shard samples
were removed by its final step, it has as many records as its plan, and it was not made
with ``--no-shuffle``. Every record must parse and
have an ID and text, and IDs must be unique within a language; anything else stops the
script. A language with fewer documents than ``--n-samples`` stops it too, unless
``--allow-fewer`` is given, which takes all of them. Codes without a directory in the
collection are skipped with a warning, as the sampler does. A language whose runs are not finished stops the script, unless
``--allow-incomplete`` is given, which skips it with a warning.

Outputs (in ``--output-dir``, ``<language>`` being the lower-cased language name):
    <language>.jsonl            the sample, in random order
    <language>.manifest.json    the runs, their seeds and commit, and the counts per code
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sample_hplt_by_quality import output_paths, parse_record, read_language_codes

LOGGER = logging.getLogger(__name__)

HPLT_V4_ROOT = Path("/appl/local/openeurollm/training/catalogue/hplt/4.0")


class IncompleteRunError(RuntimeError):
    """A sampler run has not finished."""


@dataclass(frozen=True)
class SamplerRun:
    """A finished hplt-sampler run of one language code.

    Attributes:
        code: HPLT language code.
        directory: The run's results directory.
        documents: Documents of the code in the collection, from the run's plan.
        lines: The sample's raw lines, in the run's random order.
        run_record: The run's ``run.json``.
    """

    code: str
    directory: Path
    documents: int
    lines: list[bytes]
    run_record: dict[str, Any]


def run_directory(sampler_dir: Path, collection: str, code: str) -> Path:
    """Return the results directory of a code's run, as the sampler names it.

    Finished runs may have been moved from the sampler directory to
    ``results/<collection>/`` under it; that copy is used when it exists.
    """
    tag = "v4" if collection == "clean" else f"v4-{collection}"
    name = f"results-{tag}-{code}-all"
    moved = sampler_dir / "results" / collection / name
    return moved if moved.is_dir() else sampler_dir / name


def proportional_quotas(documents: dict[str, int], total: int) -> dict[str, int]:
    """Split ``total`` over codes in proportion to their document counts.

    Rounded down, with the remainder going to the codes with the largest fractions, as
    hplt-sampler allocates over shards.

    Raises:
        ValueError: If the codes have fewer than ``total`` documents together.
    """
    available = sum(documents.values())
    if available < total:
        raise ValueError(f"{available} documents in the collection, fewer than {total}")
    exact = {code: total * count / available for code, count in documents.items()}
    quotas = {code: int(value) for code, value in exact.items()}
    by_fraction = sorted(documents, key=lambda code: (-(exact[code] - quotas[code]), code))
    for code in by_fraction[: total - sum(quotas.values())]:
        quotas[code] += 1
    return quotas


def read_run(directory: Path, code: str) -> SamplerRun:
    """Read and check a finished sampler run.

    Raises:
        IncompleteRunError: If the run has not finished.
        ValueError: If the run was not shuffled, does not hold what its plan says, or a
            record is unusable.
    """
    samples_path, plan_path, run_path = (directory / name for name in ("samples.jsonl", "plan.jsonl", "run.json"))
    if not samples_path.is_file() or (directory / "samples").exists() or not plan_path.is_file():
        raise IncompleteRunError(f"{code}: run in {directory} has not finished")
    if not run_path.is_file():
        raise ValueError(f"{code}: run in {directory} has no run.json")
    run_record = json.loads(run_path.read_text(encoding="utf-8"))
    # The start of the sample is taken as a random sample, which needs the shuffled order.
    if run_record.get("shuffle") is not True:
        raise ValueError(f"{code}: run in {directory} was made without shuffling (--no-shuffle)")
    plan = [json.loads(line) for line in plan_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    documents = sum(int(entry["count"]) for entry in plan)
    planned = sum(int(entry["n_take"]) for entry in plan)
    lines: list[bytes] = []
    with samples_path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            location = f"{samples_path}:{line_number}"
            record = parse_record(line, location)
            if not isinstance(record.get("id"), str) or not record["id"] or not isinstance(record.get("text"), str):
                raise ValueError(f"{location}: record needs a string id and text")
            lines.append(line if line.endswith(b"\n") else line + b"\n")
    if len(lines) != planned:
        raise ValueError(f"{code}: {samples_path} has {len(lines)} records, its plan {planned}")
    return SamplerRun(code, directory, documents, lines, run_record)


def collect_language(name: str, runs: Sequence[SamplerRun], missing_codes: Sequence[str], args: argparse.Namespace) -> None:
    """Write the sample and manifest of one language from its runs.

    Raises:
        ValueError: If the language has too few documents, a run too few for its share,
            or a document ID repeats.
    """
    documents = {run.code: run.documents for run in runs}
    n_samples = args.n_samples
    if sum(documents.values()) < n_samples:
        if not args.allow_fewer:
            raise ValueError(
                f"{name}: {sum(documents.values())} documents in the collection, fewer than {n_samples}; "
                "pass --allow-fewer to take all of them"
            )
        n_samples = sum(documents.values())
        LOGGER.warning("%s: only %d documents in the collection; taking all of them", name, n_samples)
    taken = proportional_quotas(documents, n_samples)
    # Each run is shuffled, so the start of each is a random sample of its code.
    short = [f"{run.code} ({len(run.lines)} of {taken[run.code]})" for run in runs if len(run.lines) < taken[run.code]]
    if short:
        raise ValueError(f"{name}: runs with fewer documents than needed: {', '.join(short)}; sample them with more")
    lines = [line for run in runs for line in run.lines[: taken[run.code]]]
    if len(runs) > 1:
        random.Random(f"{args.seed}:{name}").shuffle(lines)

    seen: dict[str, int] = {}
    for position, line in enumerate(lines, start=1):
        document_id = json.loads(line)["id"]
        if document_id in seen:
            raise ValueError(f"{name}: document ID {document_id!r} is in the sample twice (lines {seen[document_id]} and {position})")
        seen[document_id] = position

    output_path, manifest_path = output_paths(args.output_dir, name)
    write_jsonl_lines(output_path, lines)
    manifest = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "language": name,
        "collection": args.collection,
        "output": str(output_path),
        "selection": "hplt-sampler: simple random sample per language code, shards allocated in proportion "
        "to their document counts"
        + ("; codes allocated in proportion to their document counts, then shuffled" if len(runs) > 1 else ""),
        "sample_size": len(lines),
        "missing_language_codes": list(missing_codes),
        "language_codes": {
            run.code: {
                "run_directory": str(run.directory),
                "documents": run.documents,
                "documents_sampled": taken[run.code],
                "seed": run.run_record.get("seed"),
                "sampler_commit": run.run_record.get("sampler_commit"),
                "sampler_dirty": run.run_record.get("sampler_dirty"),
                "local_root": run.run_record.get("local_root"),
            }
            for run in runs
        },
    }
    if len(runs) > 1:
        manifest["seed"] = args.seed
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    LOGGER.info(
        "%s: wrote %d documents to %s (%s)",
        name, len(lines), output_path,
        ", ".join(f"{code}: {count} of {run.documents}" for run, (code, count) in zip(runs, taken.items())),
    )


def write_jsonl_lines(path: Path, lines: Sequence[bytes]) -> None:
    """Write raw JSONL lines through a temporary file, so an interrupted run leaves no partial file."""
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("wb") as handle:
        handle.writelines(lines)
    temporary_path.replace(path)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description="Collect hplt-sampler runs into one sample file per language.")
    parser.add_argument("--collection", choices=("clean", "noisy", "poor"), required=True)
    parser.add_argument("--sampler-dir", type=Path, required=True, help="The hplt-sampler directory the runs were made in")
    parser.add_argument("--languages-file", type=Path, default=Path("data/oellm_languages.txt"))
    parser.add_argument("--languages", nargs="+", default=None, help="Language names from --languages-file (default: all)")
    parser.add_argument(
        "--hplt-root",
        type=Path,
        default=HPLT_V4_ROOT,
        help="HPLT 4.0 directory with the collections, to tell codes without data from unfinished runs",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: data/hplt_<collection>_random_sample")
    parser.add_argument("--n-samples", type=int, default=30_000, help="Documents per language")
    parser.add_argument("--seed", type=int, default=0, help="Seed for shuffling languages with several codes")
    parser.add_argument("--allow-incomplete", action="store_true", help="Skip languages whose runs have not finished")
    parser.add_argument(
        "--allow-fewer",
        action="store_true",
        help="Take all documents of a language that has fewer than --n-samples instead of failing",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing sample files")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Collect every requested language."""
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s [%(levelname)s] %(message)s")
    if args.n_samples < 1:
        raise ValueError("--n-samples must be >= 1")
    collection_dir = args.hplt_root / args.collection
    for directory in (args.sampler_dir, collection_dir):
        if not directory.is_dir():
            raise FileNotFoundError(f"Directory not found: {directory}")
    if args.output_dir is None:
        args.output_dir = Path(f"data/hplt_{args.collection}_random_sample")

    language_codes = read_language_codes(args.languages_file)
    names = list(language_codes) if args.languages is None else args.languages
    unknown = [name for name in names if name not in language_codes]
    if unknown:
        raise ValueError(f"Not in {args.languages_file}: {', '.join(unknown)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing, incomplete, without_data = [], [], []
    for name in names:
        if output_paths(args.output_dir, name)[0].exists() and not args.overwrite:
            LOGGER.warning("%s: skipped, its sample file exists; pass --overwrite to replace it", name)
            existing.append(name)
            continue
        codes = [code for code in language_codes[name] if (collection_dir / code).is_dir()]
        missing = [code for code in language_codes[name] if code not in codes]
        if missing:
            LOGGER.warning("%s: no directory in %s for %s", name, collection_dir, ", ".join(missing))
        if not codes:
            without_data.append(name)
            continue
        try:
            runs = [read_run(run_directory(args.sampler_dir, args.collection, code), code) for code in codes]
        except IncompleteRunError as exc:
            if not args.allow_incomplete:
                raise
            LOGGER.warning("%s: skipped: %s", name, exc)
            incomplete.append(name)
            continue
        collect_language(name, runs, missing, args)

    if existing:
        LOGGER.warning("Skipped %d language(s) with existing samples: %s", len(existing), ", ".join(existing))
    if without_data:
        LOGGER.warning("Skipped %d language(s) without data in %s: %s", len(without_data), collection_dir, ", ".join(without_data))
    if incomplete:
        LOGGER.warning("Skipped %d language(s) whose runs have not finished: %s", len(incomplete), ", ".join(incomplete))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
