"""Report which HPLT random-sample labelling results are complete and which need a rerun.

For every collection (clean_random, noisy_random, poor_random), label type (topic,
format), and language of the languages file, the script counts the documents in the
result file written by ``hpc/lumi/array_generate_distribution_hplt.sh``
(``results/distribution/<run>/<collection>/<label type>/<language>.jsonl``) and compares
it with the collection's target and with the language's sample size (from the sample's
``.manifest.json``):

- ``complete``: at least the target number of documents (25,000 for clean and noisy,
  10,000 for poor by default).
- ``complete_all_data``: fewer than the target, but every document of the sample is
  labelled: the language has no more data.
- ``running``: fewer than the target, and the newest labelling log of the file was
  written to within ``--active-minutes`` (default 60), so a job is probably still on it.
- ``incomplete``: some documents, fewer than the target and than the sample, and no
  recent log: the run crashed or ran out of time; relaunch it to resume.
- ``missing``: no result file or an empty one, and no recent log.

The newest log of each file (``logs/distribution_hplt_<job>_<task>.out`` and ``.err``) is
found by its ``output=`` line, and its end state is reported (completed, time limit,
cancelled, crashed, or no end marker) to help tell crashes from time limits. Queued
jobs have no log yet, so check ``squeue --me`` before relaunching ``missing`` or
``incomplete`` tasks.

Outputs (by default next to the results): ``labelling_status.tsv`` with one row per
collection, label type, and language, and ``labelling_status.md`` with counts per
status and, per collection, the array task IDs to relaunch with the submit commands.

Usage (on LUMI, or anywhere the project directory is mounted)::

    python scripts/data_processing/labelling_status.py
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import logging
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

LOGGER = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
COLLECTIONS = ("clean", "noisy", "poor")
LABEL_TYPES = ("topic", "format")
DEFAULT_TARGETS = {"clean": 25_000, "noisy": 25_000, "poor": 10_000}
STATUSES = ("complete", "complete_all_data", "running", "incomplete", "missing")
# The end of a log is searched for these markers; vLLM's shutdown noise follows them.
LOG_TAIL_BYTES = 64 * 1024
LOG_NAME = re.compile(r"distribution_hplt_(\d+)_(\d+)\.out$")


@dataclass
class LogInfo:
    """The newest labelling log of one result file.

    Attributes:
        job: SLURM array job ID.
        task: Array task ID.
        state: completed, time_limit, cancelled, crashed, or no_end_marker.
        modified: Last modification time of the log (.err if present, else .out).
    """

    job: int
    task: int
    state: str
    modified: float


@dataclass
class Status:
    """The labelling status of one collection, label type, and language."""

    collection: str
    label_type: str
    task: int
    language: str
    sample_size: int
    target: int
    labelled: int
    status: str
    log: LogInfo | None


def read_language_names(path: Path) -> list[str]:
    """Return the language names of a languages file, in file order.

    The name is the part of each non-empty line before the colon, as the labelling
    launcher reads it (task N is language N, task 128 + N its format labels).

    Raises:
        ValueError: If a non-empty line has no colon or the file has no languages.
    """
    names = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        if ":" not in line:
            raise ValueError(f"{path}:{line_number}: expected '<language>: <codes>', got {line!r}")
        names.append(line.split(":", 1)[0].strip())
    if not names:
        raise ValueError(f"No languages in {path}")
    return names


def file_stem(language: str) -> str:
    """Return a language's sample and result file name stem: lower case, spaces as underscores."""
    return language.lower().replace(" ", "_")


def count_lines(path: Path, chunk_size: int = 16 * 1024 * 1024) -> int:
    """Count the complete lines of a file; a partial last line (a run still writing) is not counted.

    Raises:
        OSError: If the file can't be read.
    """
    lines = 0
    last = b"\n"
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            lines += chunk.count(b"\n")
            last = chunk[-1:]
    if last != b"\n":
        LOGGER.warning("%s ends in a partial line (is a run still writing it?); not counted", path)
    return lines


def sample_size(sample_dir: Path, stem: str) -> int:
    """Return a sample's document count from its manifest.

    Raises:
        FileNotFoundError: If the sample or its manifest is missing.
        ValueError: If the manifest has no valid sample_size.
    """
    sample = sample_dir / f"{stem}.jsonl"
    manifest = sample_dir / f"{stem}.manifest.json"
    if not sample.is_file() or not manifest.is_file():
        raise FileNotFoundError(f"No sample or manifest for {stem} in {sample_dir}")
    size = json.loads(manifest.read_text(encoding="utf-8")).get("sample_size")
    if not isinstance(size, int) or size < 0:
        raise ValueError(f"{manifest}: invalid sample_size {size!r}")
    return size


def log_state(err_path: Path) -> str:
    """Return how a labelling job ended, from the end of its .err log."""
    if not err_path.is_file():
        return "no_end_marker"
    with err_path.open("rb") as handle:
        handle.seek(max(0, err_path.stat().st_size - LOG_TAIL_BYTES))
        tail = handle.read().decode("utf-8", errors="replace")
    if re.search(r"Completed \d+ new documents; output contains|max_documents=\d+ reached", tail):
        return "completed"
    if "DUE TO TIME LIMIT" in tail:
        return "time_limit"
    if "CANCELLED" in tail:
        return "cancelled"
    if "srun: error" in tail or "Traceback (most recent call last)" in tail:
        return "crashed"
    return "no_end_marker"


def newest_logs(log_dir: Path) -> dict[str, LogInfo]:
    """Map each result path (the logs' ``output=`` line) to its newest labelling log.

    Logs are ordered by job ID, then by modification time.
    """
    newest: dict[str, LogInfo] = {}
    for out_path in sorted(log_dir.glob("distribution_hplt_*_*.out")):
        match = LOG_NAME.match(out_path.name)
        if match is None:
            continue
        with out_path.open("r", encoding="utf-8", errors="replace") as handle:
            head = "".join(line for _, line in zip(range(20), handle))
        output = re.search(r"^output=(\S+)$", head, re.MULTILINE)
        if output is None:
            continue
        err_path = out_path.with_suffix(".err")
        modified = (err_path if err_path.is_file() else out_path).stat().st_mtime
        info = LogInfo(int(match[1]), int(match[2]), log_state(err_path), modified)
        previous = newest.get(output[1])
        if previous is None or (info.job, info.modified) > (previous.job, previous.modified):
            newest[output[1]] = info
    return newest


def classify(labelled: int, size: int, target: int, log: LogInfo | None, active_seconds: float) -> str:
    """Return the status of one result file (see the module docstring).

    Raises:
        ValueError: If the result has more documents than the sample.
    """
    if labelled > size:
        raise ValueError(f"{labelled} labelled documents but only {size} in the sample")
    if labelled >= target:
        return "complete"
    if labelled == size:
        return "complete_all_data"
    if log is not None and log.state == "no_end_marker" and time.time() - log.modified < active_seconds:
        return "running"
    return "incomplete" if labelled else "missing"


def compress_ranges(numbers: list[int]) -> str:
    """Write sorted integers as Slurm array ranges, e.g. [0, 1, 2, 5] -> '0-2,5'."""
    parts = []
    start = previous = None
    for number in sorted(numbers):
        if start is None:
            start = previous = number
        elif number == previous + 1:
            previous = number
        else:
            parts.append(f"{start}" if start == previous else f"{start}-{previous}")
            start = previous = number
    if start is not None:
        parts.append(f"{start}" if start == previous else f"{start}-{previous}")
    return ",".join(parts)


def write_tsv(path: Path, statuses: list[Status]) -> None:
    """Write one row per collection, label type, and language."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["collection", "label_type", "task", "language", "sample_size", "target", "labelled",
                         "status", "last_job", "last_log_state", "last_log_time"])
        for item in statuses:
            log = item.log
            writer.writerow([
                f"{item.collection}_random", item.label_type, item.task, item.language, item.sample_size,
                item.target, item.labelled, item.status,
                f"{log.job}_{log.task}" if log else "", log.state if log else "",
                datetime.datetime.fromtimestamp(log.modified).strftime("%Y-%m-%d %H:%M") if log else "",
            ])


def make_markdown(statuses: list[Status], targets: dict[str, int], stray: list[Path]) -> str:
    """Summarize the statuses and list the tasks to relaunch, with submit commands."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# HPLT random-sample labelling status",
        "",
        f"Made {now} by `scripts/data_processing/labelling_status.py`; details per language in "
        "`labelling_status.tsv`. Targets: " + ", ".join(f"{c} {t:,}" for c, t in targets.items()) + ".",
        "",
        "| Collection | Label type | " + " | ".join(STATUSES) + " |",
        "|---|---|" + "---:|" * len(STATUSES),
    ]
    counts = Counter((item.collection, item.label_type, item.status) for item in statuses)
    for collection in COLLECTIONS:
        for label_type in LABEL_TYPES:
            lines.append(f"| {collection}_random | {label_type} | "
                         + " | ".join(str(counts[collection, label_type, status]) for status in STATUSES) + " |")

    lines += ["", "## To relaunch", "",
              "`incomplete` and `missing` tasks, per collection. Queued jobs have no log yet: check "
              "`squeue --me` first. Runs resume from what they have written, and the last argument "
              "stops them at the target. From `hpc/lumi/`:", ""]
    relaunch: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for item in statuses:
        if item.status in ("incomplete", "missing"):
            relaunch[item.collection][item.status].append(item.task)
    if not relaunch:
        lines.append("Nothing to relaunch.")
    for collection in COLLECTIONS:
        if collection not in relaunch:
            continue
        tasks = sorted(task for group in relaunch[collection].values() for task in group)
        lines += [
            f"### {collection}_random: {len(tasks)} tasks",
            "",
            *(f"- {status}: `{compress_ranges(group)}`" for status, group in sorted(relaunch[collection].items())),
            "",
            "```bash",
            f"bash submit_array_in_chunks.sh {compress_ranges(tasks)} 40 36 --time=24:00:00 \\",
            f"    --export=ALL,DATA_QUALITY={collection}_random array_generate_distribution_hplt.sh {targets[collection]}",
            "```",
            "",
        ]
    if stray:
        lines += ["## Result files not in the languages file", "", *(f"- `{path}`" for path in stray), ""]
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base-dir", type=Path, default=REPO_ROOT, help="Repository root (default: this repository).")
    parser.add_argument("--run", default="Qwen3.8-27B-hplt-annotations",
                        help="Annotation run directory under results/distribution/.")
    parser.add_argument("--languages-file", type=Path, default=None,
                        help="Languages file (default: data/qwen_label_languages.txt).")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Where to write labelling_status.tsv and .md (default: the run directory).")
    parser.add_argument("--targets", nargs=3, type=int, metavar=("CLEAN", "NOISY", "POOR"),
                        default=[DEFAULT_TARGETS[c] for c in COLLECTIONS], help="Documents needed per collection.")
    parser.add_argument("--active-minutes", type=float, default=60.0,
                        help="A log written to this recently, without an end marker, counts as a running job.")
    parser.add_argument("--workers", type=int, default=16, help="Files counted in parallel.")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Count the result files, classify them, and write the status files."""
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s [%(levelname)s] %(message)s")
    base: Path = args.base_dir
    run_dir = base / "results/distribution" / args.run
    if not run_dir.is_dir():
        raise FileNotFoundError(f"No annotation run directory {run_dir}")
    targets = dict(zip(COLLECTIONS, args.targets))
    if any(target < 1 for target in targets.values()):
        raise ValueError(f"Targets must be positive: {targets}")
    languages = read_language_names(args.languages_file or base / "data/qwen_label_languages.txt")

    LOGGER.info("Reading the labelling logs")
    logs = newest_logs(base / "logs")
    # The logs name the result files by their path on LUMI; match them by the part
    # under results/, so the script also works on a mounted copy.
    logs_by_suffix = {path.split("/results/", 1)[-1]: info for path, info in logs.items() if "/results/" in path}

    jobs = []
    for collection in COLLECTIONS:
        sample_dir = base / f"data/hplt_{collection}_random_sample"
        for type_index, label_type in enumerate(LABEL_TYPES):
            for index, language in enumerate(languages):
                stem = file_stem(language)
                result = run_dir / f"{collection}_random" / label_type / f"{stem}.jsonl"
                jobs.append((collection, label_type, index + type_index * len(languages), language, stem,
                             sample_size(sample_dir, stem), result))

    def count(job: tuple) -> int:
        result = job[-1]
        return count_lines(result) if result.is_file() else 0

    LOGGER.info("Counting the documents of %d result files", len(jobs))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        counts = list(pool.map(count, jobs))

    statuses = []
    active_seconds = args.active_minutes * 60
    for (collection, label_type, task, language, stem, size, result), labelled in zip(jobs, counts):
        log = logs_by_suffix.get(str(result.relative_to(base)).split("results/", 1)[-1])
        try:
            status = classify(labelled, size, targets[collection], log, active_seconds)
        except ValueError as error:
            raise ValueError(f"{result}: {error}") from error
        statuses.append(Status(collection, label_type, task, language, size, targets[collection], labelled,
                               status, log))

    expected = {job[-1] for job in jobs}
    stray = sorted(path for collection in COLLECTIONS for label_type in LABEL_TYPES
                   for path in (run_dir / f"{collection}_random" / label_type).glob("*.jsonl")
                   if path not in expected)
    for path in stray:
        LOGGER.warning("Result file not in the languages file: %s", path)

    output_dir = args.output_dir or run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(output_dir / "labelling_status.tsv", statuses)
    (output_dir / "labelling_status.md").write_text(make_markdown(statuses, targets, stray), encoding="utf-8")
    for (collection, label_type), group in sorted(Counter((s.collection, s.label_type) for s in statuses).items()):
        summary = Counter(s.status for s in statuses if (s.collection, s.label_type) == (collection, label_type))
        LOGGER.info("%s_random %s: %s", collection, label_type, dict(summary))
    LOGGER.info("Wrote %s and %s", output_dir / "labelling_status.tsv", output_dir / "labelling_status.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
