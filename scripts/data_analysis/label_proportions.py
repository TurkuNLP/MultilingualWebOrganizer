"""Count the topic and format proportions of one language's classifier labels.

Reads every finished file of ``classify_hplt.py``'s output for one language
(``<output_root>/<language>/*.jsonl.zst``), in parallel, and counts per label and per
topic x format pair:

* **documents** whose top label it is (the classifiers' most probable label, as WebOrganizer
  uses to assign documents to domains);
* **characters** of text in those documents (the ``text`` field, as it goes into the
  training data; a proxy for tokens until the training data is tokenized);
* **probability mass**: the sum of the label's probability over all documents, i.e. the
  expected number of documents of the label under the full predicted distributions, to show
  how much the top label alone shifts the proportions.

The topic x format pair proportions are computed in two ways, by characters and by documents:

* **all labels:** every pair at its share of the corpus;
* **Bad example dropped, Other fixed:** documents labelled *Bad example* as topic or format are
  dropped (a quality filter); pairs with *Other* as topic or format (and no *Bad example*)
  keep their share of the full corpus; and the remaining pairs of the 24 WebOrganizer labels
  are scaled up together to fill the rest.

Both are written to ``<output>`` and, one row per pair, to ``<output stem>.pairs.csv``.

The label order and output fields come from ``<output_root>/metadata.json``. The script stops
with an error if the language has unfinished files (``.partial`` or ``.lock`` files), if a
line is malformed, or, with ``--expected_counts_dir``, if a file holds another number of
documents than hplt-sampler counted for its input (checked after the results are written).

Writes a JSON file with the totals, the per-file counts and the label order, and logs the
proportions as tables.

Example::

    python scripts/data_analysis/label_proportions.py \\
        --input_dir results/classifier_predictions/mmbert-1/hplt-4.0-clean/deu_Latn \\
        --output results/statistics/classifier_label_proportions/hplt-4.0-clean/deu_Latn.json \\
        --expected_counts_dir ../hplt-sampler/counts/v4/deu_Latn --workers 32
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# The zstandard reader of the HPLT sampler, so the output is read like the HPLT input.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_processing"))
from sample_hplt_by_quality import open_zst  # noqa: E402

LOGGER = logging.getLogger(__name__)

KINDS = ("topic", "format")
OUTPUT_SUFFIX = ".jsonl.zst"
# Our two labels beyond WebOrganizer's 24, in both taxonomies.
BAD_EXAMPLE = "Bad example"
OTHER = "Other"
MEASURES = ("characters", "documents")
VARIANTS = ("all_labels", "bad_example_dropped_other_fixed")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command-line arguments.

    Args:
        argv: Arguments to parse instead of ``sys.argv``.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_dir", type=Path, required=True, help="One language's output, <output_root>/<language>")
    parser.add_argument("--output", type=Path, required=True, help="JSON file to write the counts to")
    parser.add_argument(
        "--expected_counts_dir",
        type=Path,
        default=None,
        help="hplt-sampler's per-file counts for the language (<name>.jsonl.zst.json with a 'count'), to check that "
        "every output file holds one line per input document",
    )
    parser.add_argument("--workers", type=int, default=8, help="Files read in parallel")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    return args


def load_metadata(input_dir: Path) -> tuple[list[str], dict[str, list[str]]]:
    """Read the output fields and label orders from the output root's metadata.json.

    Args:
        input_dir: One language's output directory.

    Returns:
        The output fields, in order, and the labels of each classifier, in output order.

    Raises:
        FileNotFoundError: If there is no metadata.json next to the language directory.
        ValueError: If it lacks a classifier or the output fields.
    """
    path = input_dir.parent / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    missing = [kind for kind in KINDS if kind not in metadata.get("classifiers", {})]
    if missing or "output_fields" not in metadata:
        raise ValueError(f"{path} lacks output_fields or the classifiers {missing}")
    labels = {kind: list(metadata["classifiers"][kind]["labels"]) for kind in KINDS}
    return list(metadata["output_fields"]), labels


def empty_counts(labels: dict[str, list[str]]) -> dict[str, Any]:
    """Return zeroed counters for one file or the total.

    Args:
        labels: The labels of each classifier.

    Returns:
        Counters: documents and characters in all, per label (documents, characters,
        probability mass) and per topic x format pair (documents, characters).
    """
    n_topic, n_format = len(labels["topic"]), len(labels["format"])
    counts: dict[str, Any] = {"documents": 0, "characters": 0}
    for kind in KINDS:
        n = len(labels[kind])
        counts[kind] = {"documents": [0] * n, "characters": [0] * n, "probability_mass": [0.0] * n}
    counts["pairs"] = {
        "documents": [[0] * n_format for _ in range(n_topic)],
        "characters": [[0] * n_format for _ in range(n_topic)],
    }
    return counts


def count_file(path: Path, fields: list[str], labels: dict[str, list[str]]) -> dict[str, Any]:
    """Count the labels of one output file.

    Args:
        path: The ``.jsonl.zst`` output file.
        fields: The expected output fields, in order.
        labels: The labels of each classifier.

    Returns:
        The file's counters (see ``empty_counts``).

    Raises:
        ValueError: On a malformed line: invalid JSON, other fields, an unknown label, or a
            probability list of the wrong length.
    """
    index = {kind: {label: i for i, label in enumerate(labels[kind])} for kind in KINDS}
    n_labels = {kind: len(labels[kind]) for kind in KINDS}
    counts = empty_counts(labels)
    expected_fields = tuple(fields)
    with open_zst(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if tuple(record) != expected_fields:
                raise ValueError(f"{path}:{line_number}: fields {list(record)}, expected {fields}")
            characters = len(record["text"])
            counts["documents"] += 1
            counts["characters"] += characters
            top = {}
            for kind in KINDS:
                label = record[kind]
                probs = record[f"{kind}_probs"]
                if label not in index[kind] or len(probs) != n_labels[kind]:
                    raise ValueError(f"{path}:{line_number}: unknown {kind} label {label!r} or {len(probs)} probabilities")
                i = index[kind][label]
                top[kind] = i
                counts[kind]["documents"][i] += 1
                counts[kind]["characters"][i] += characters
                mass = counts[kind]["probability_mass"]
                for j, p in enumerate(probs):
                    mass[j] += p
            counts["pairs"]["documents"][top["topic"]][top["format"]] += 1
            counts["pairs"]["characters"][top["topic"]][top["format"]] += characters
    return counts


def add_counts(total: dict[str, Any], part: dict[str, Any]) -> None:
    """Add one file's counters to the total, in place.

    Args:
        total: The running total.
        part: One file's counters.
    """
    total["documents"] += part["documents"]
    total["characters"] += part["characters"]
    for kind in KINDS:
        for measure, values in part[kind].items():
            for i, value in enumerate(values):
                total[kind][measure][i] += value
    for measure, rows in part["pairs"].items():
        for i, row in enumerate(rows):
            for j, value in enumerate(row):
                total["pairs"][measure][i][j] += value


def expected_count(counts_dir: Path, name: str) -> int:
    """Return hplt-sampler's document count for an input file.

    Args:
        counts_dir: hplt-sampler's count directory for the language.
        name: The file name, ``<bucket>_<part>.jsonl.zst``.

    Returns:
        The number of documents in the input file.

    Raises:
        FileNotFoundError: If there is no count for the file.
        ValueError: If the count file has no integer ``count``.
    """
    record = json.loads((counts_dir / f"{name}.json").read_text(encoding="utf-8"))
    count = record.get("count")
    if not isinstance(count, int):
        raise ValueError(f"{counts_dir / f'{name}.json'} has no integer count: {count!r}")
    return count


def pair_proportions(pairs: list[list[int]], labels: dict[str, list[str]]) -> dict[str, list[list[float]]]:
    """Compute the topic x format pair proportions in both variants.

    Args:
        pairs: Counts (characters or documents) per pair, indexed [topic][format].
        labels: The labels of each classifier; both must include *Bad example* and *Other*.

    Returns:
        Proportions per pair, indexed [topic][format], for each variant in ``VARIANTS``:
        ``all_labels`` (every pair at its corpus share) and ``bad_example_dropped_other_fixed``
        (pairs with *Bad example* dropped, pairs with *Other* at their corpus share, the
        other pairs scaled up together to fill the rest).

    Raises:
        ValueError: If a taxonomy lacks *Bad example* or *Other*, the counts are all zero, or
            no pair is left to scale.
    """
    for kind in KINDS:
        missing = [label for label in (BAD_EXAMPLE, OTHER) if label not in labels[kind]]
        if missing:
            raise ValueError(f"The {kind} labels lack {missing}")
    bad = (labels["topic"].index(BAD_EXAMPLE), labels["format"].index(BAD_EXAMPLE))
    other = (labels["topic"].index(OTHER), labels["format"].index(OTHER))
    total = sum(map(sum, pairs))
    if total == 0:
        raise ValueError("No counts")
    shares = [[value / total for value in row] for row in pairs]

    def group(i: int, j: int) -> str:
        if i == bad[0] or j == bad[1]:
            return "bad"
        if i == other[0] or j == other[1]:
            return "other"
        return "rest"

    cells = [(i, j) for i in range(len(pairs)) for j in range(len(pairs[0]))]
    other_share = sum(shares[i][j] for i, j in cells if group(i, j) == "other")
    rest_share = sum(shares[i][j] for i, j in cells if group(i, j) == "rest")
    if rest_share == 0:
        raise ValueError("No document outside Bad example and Other to scale")
    scale = (1 - other_share) / rest_share
    filtered = [[0.0] * len(row) for row in pairs]
    for i, j in cells:
        kind = group(i, j)
        filtered[i][j] = 0.0 if kind == "bad" else shares[i][j] if kind == "other" else shares[i][j] * scale
    return {VARIANTS[0]: shares, VARIANTS[1]: filtered}


def write_pairs_csv(path: Path, labels: dict[str, list[str]], total: dict[str, Any],
                    proportions: dict[str, dict[str, list[list[float]]]]) -> None:
    """Write one CSV row per topic x format pair with its counts and proportions.

    Args:
        path: The CSV file.
        labels: The labels of each classifier.
        total: The total counters.
        proportions: Pair proportions per measure and variant (see ``pair_proportions``).
    """
    columns = ["topic", "format", "documents", "characters"] + [
        f"{measure}_share_{variant}" for measure in MEASURES for variant in VARIANTS
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for i, topic in enumerate(labels["topic"]):
            for j, format_label in enumerate(labels["format"]):
                row = [topic, format_label, total["pairs"]["documents"][i][j], total["pairs"]["characters"][i][j]]
                row += [f"{proportions[measure][variant][i][j]:.8f}" for measure in MEASURES for variant in VARIANTS]
                writer.writerow(row)


def log_variants(labels: dict[str, list[str]], proportions: dict[str, list[list[float]]]) -> None:
    """Log the topic and format shares (by characters) of both variants, and the largest pairs.

    Args:
        labels: The labels of each classifier.
        proportions: Pair proportions by characters, per variant.
    """
    for kind, axis in (("topic", 1), ("format", 0)):
        marginals = {}
        for variant, matrix in proportions.items():
            if axis == 1:
                marginals[variant] = [sum(row) for row in matrix]
            else:
                marginals[variant] = [sum(row[j] for row in matrix) for j in range(len(matrix[0]))]
        order = sorted(range(len(labels[kind])), key=lambda i: -marginals[VARIANTS[0]][i])
        lines = [f"{kind.capitalize()} shares of characters: all labels | Bad example dropped, Other fixed"]
        lines += [f"{labels[kind][i]:<20} {marginals[VARIANTS[0]][i]:7.2%} {marginals[VARIANTS[1]][i]:7.2%}" for i in order]
        LOGGER.info("\n".join(lines))
    matrix = proportions[VARIANTS[0]]
    cells = sorted(((i, j) for i in range(len(matrix)) for j in range(len(matrix[0]))), key=lambda c: -matrix[c[0]][c[1]])
    lines = ["Largest topic x format pairs by characters: all labels | Bad example dropped, Other fixed"]
    lines += [f"{labels['topic'][i]:<20} x {labels['format'][j]:<20} {matrix[i][j]:7.2%} {proportions[VARIANTS[1]][i][j]:7.2%}"
              for i, j in cells[:25]]
    LOGGER.info("\n".join(lines))


def log_table(title: str, labels: list[str], total: dict[str, Any], kind: str) -> None:
    """Log one classifier's proportions, largest share of characters first.

    Args:
        title: Table title.
        labels: The classifier's labels.
        total: The total counters.
        kind: ``topic`` or ``format``.
    """
    counts = total[kind]
    documents, characters = total["documents"], total["characters"]
    rows = sorted(range(len(labels)), key=lambda i: -counts["characters"][i])
    lines = [title, f"{'label':<20} {'documents':>9} {'characters':>10} {'prob. mass':>10} {'chars/doc':>9}"]
    for i in rows:
        docs = counts["documents"][i]
        lines.append(
            f"{labels[i]:<20} {docs / documents:9.2%} {counts['characters'][i] / characters:10.2%} "
            f"{counts['probability_mass'][i] / documents:10.2%} {counts['characters'][i] / max(docs, 1):9,.0f}"
        )
    LOGGER.info("\n".join(lines))


def main(argv: Sequence[str] | None = None) -> None:
    """Count the label proportions of one language's classifier output.

    Args:
        argv: Arguments to parse instead of ``sys.argv``.

    Raises:
        ValueError: If the language has unfinished files, a line is malformed, or (with
            ``--expected_counts_dir``) a file's document count differs from its input's.
    """
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    unfinished = sorted(p.name for p in args.input_dir.iterdir() if p.suffix in (".partial", ".lock", ".tmp"))
    if unfinished:
        raise ValueError(f"{args.input_dir} has unfinished files: {unfinished}; wait until the language is labelled")
    files = sorted(args.input_dir.glob(f"*{OUTPUT_SUFFIX}"), key=lambda p: -p.stat().st_size)
    if not files:
        raise ValueError(f"No {OUTPUT_SUFFIX} files in {args.input_dir}")
    fields, labels = load_metadata(args.input_dir)
    LOGGER.info("Counting %d files of %s with %d workers", len(files), args.input_dir, args.workers)

    started = time.monotonic()
    total = empty_counts(labels)
    per_file: dict[str, dict[str, int]] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(count_file, path, fields, labels): path for path in files}
        for done, future in enumerate(as_completed(futures), start=1):
            path = futures[future]
            counts = future.result()
            add_counts(total, counts)
            per_file[path.name] = {"documents": counts["documents"], "characters": counts["characters"]}
            LOGGER.info("%d/%d files done (%s: %d documents), %.0f s", done, len(files), path.name,
                        counts["documents"], time.monotonic() - started)

    proportions = {measure: pair_proportions(total["pairs"][measure], labels) for measure in MEASURES}
    result = {
        "input_dir": str(args.input_dir.resolve()),
        "labels": labels,
        "total": total,
        "pair_proportions": proportions,
        "files": dict(sorted(per_file.items())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    pairs_csv = args.output.with_name(f"{args.output.stem}.pairs.csv")
    write_pairs_csv(pairs_csv, labels, total, proportions)
    LOGGER.info("%d documents, %d characters; wrote %s and %s", total["documents"], total["characters"], args.output, pairs_csv)
    for kind in KINDS:
        log_table(f"{kind.capitalize()} proportions (top label; probability mass for comparison):", labels[kind], total, kind)
    log_variants(labels, proportions["characters"])

    if args.expected_counts_dir is not None:
        mismatches = []
        for name, counts in sorted(per_file.items()):
            expected = expected_count(args.expected_counts_dir, name)
            if counts["documents"] != expected:
                mismatches.append(f"{name}: {counts['documents']} documents, input has {expected}")
        expected_files = {p.name.removesuffix(".json") for p in args.expected_counts_dir.glob(f"*{OUTPUT_SUFFIX}.json")}
        missing = sorted(expected_files - set(per_file))
        if mismatches or missing:
            raise ValueError(f"Output does not match the input: {mismatches}; files without output: {missing}")
        LOGGER.info("Every input file has its output, with one line per input document")


if __name__ == "__main__":
    main()
