"""Compare label distributions with WebOrganizer's original annotations.

The candidate file is an output of ``generate_label_distributions.py`` run on a
WebOrganizer annotation dataset (e.g. ``array_generate_distribution_weborganizer.sh``),
so its document IDs are WARC record IDs. The reference is the same WebOrganizer
dataset, streamed from HuggingFace or read from a local JSONL export. Documents
are joined by ID, and the report covers top-label agreement, how close the
distributions are, calibration against the reference labels, confidence,
label shares, and letter-position bias.

WebOrganizer's labels are a reference, not ground truth, so "accuracy" here
means agreement with WebOrganizer.

Our taxonomies add labels WebOrganizer doesn't have (``Other`` and ``Bad
example``). For candidates that include them, the comparison routes them back:
their probability is removed and the distribution is renormalized over
WebOrganizer's labels. The report's ``label_routing`` section says how often the
added labels were chosen and how much probability they held.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import yaml  # type: ignore

try:
    from .distribution_metrics import (
        DistributionRecord,
        cohen_kappa,
        confidence_profile,
        expected_calibration_error,
        exit_without_finalization,
        f1_scores,
        jensen_shannon_bits,
        label_shares,
        letter_bias,
        normalize_logprobs,
        probabilities,
        restrict_to_labels,
        ranked_indices,
        read_distribution_file,
        share_divergence_bits,
        summarize,
        top_confusions,
        top_index,
        write_report,
    )
except ImportError:
    from distribution_metrics import (
        DistributionRecord,
        cohen_kappa,
        confidence_profile,
        expected_calibration_error,
        exit_without_finalization,
        f1_scores,
        jensen_shannon_bits,
        label_shares,
        letter_bias,
        normalize_logprobs,
        probabilities,
        restrict_to_labels,
        ranked_indices,
        read_distribution_file,
        share_divergence_bits,
        summarize,
        top_confusions,
        top_index,
        write_report,
    )

LOGGER = logging.getLogger(__name__)

# Spelling differences between WebOrganizer's taxonomy and the label YAML files.
# WebOrganizer's topic taxonomy misspells "Hardware".
WEBORGANIZER_NAME_ALIASES = {"Electronics & Hardare": "Electronics & Hardware"}
# Labels this project added to the taxonomies; WebOrganizer's data doesn't have them.
EXTRA_LABEL_NAMES = ("Other", "Bad example")


def weborganizer_label_subset(
    label_names: Sequence[str], short_names: Sequence[str]
) -> tuple[list[str], list[str], list[str]]:
    """Split our labels into WebOrganizer's labels and the labels we added.

    Args:
        label_names: Full label names from a label YAML file, in file order.
        short_names: Short names in the same order.

    Returns:
        WebOrganizer's label names and short names, in taxonomy order, and the
        short names of the added labels.

    Raises:
        ValueError: If an added label is not after all of WebOrganizer's labels,
            which would change the taxonomy order.
    """
    extra_positions = [i for i, name in enumerate(label_names) if name in EXTRA_LABEL_NAMES]
    first_extra = len(label_names) - len(extra_positions)
    if extra_positions != list(range(first_extra, len(label_names))):
        raise ValueError(f"Added labels {EXTRA_LABEL_NAMES} must come after WebOrganizer's labels")
    keep = [i for i in range(len(label_names)) if i not in extra_positions]
    return (
        [label_names[i] for i in keep],
        [short_names[i] for i in keep],
        [short_names[i] for i in extra_positions],
    )


def load_label_names(path: Path) -> tuple[list[str], list[str]]:
    """Return label names and short names from a label YAML file, in file order."""
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"Expected a non-empty YAML list in {path}")
    try:
        return [item["name"] for item in raw], [item["short_name"] for item in raw]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Label entries in {path} need 'name' and 'short_name'") from exc


def stream_reference_rows(
    dataset: str | None, split: str, jsonl: Path | None
) -> Iterator[dict[str, Any]]:
    """Yield WebOrganizer rows from a HuggingFace dataset or a local JSONL file."""
    if jsonl is not None:
        with jsonl.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ValueError(f"Empty JSONL line in {jsonl}:{line_number}")
                yield json.loads(line)
        return
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise RuntimeError("The 'datasets' package is required for --reference-dataset") from exc
    yield from load_dataset(dataset, split=split, streaming=True)


def reference_record(
    row: dict[str, Any], label_names: Sequence[str], row_number: int
) -> DistributionRecord:
    """Convert one WebOrganizer row into a distribution record.

    WebOrganizer stores ``choice_loss`` (the answer log-probability per label,
    in taxonomy order) and ``label_permutation`` (the label shown at each
    letter). The label order is checked against ``label_names`` through the
    row's ``top_choice`` text.

    Raises:
        ValueError: If the row is malformed or its label order does not match.
    """
    try:
        doc_id = row["metadata"]["WARC-Record-ID"]
        choice_loss = row["choice_loss"]
        permutation = row["label_permutation"]
        top_choice = row["top_choice"]
        top_choice_index = row["top_choice_index"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Reference row {row_number} is missing a field: {exc}") from exc
    n = len(label_names)
    if not isinstance(doc_id, str) or not doc_id:
        raise ValueError(f"Reference row {row_number} has no WARC-Record-ID")
    if len(choice_loss) != n or sorted(permutation) != list(range(n)):
        raise ValueError(f"Reference row {row_number} does not have {n} labels")
    reference_name = top_choice.split("\n")[0].strip()
    reference_name = WEBORGANIZER_NAME_ALIASES.get(reference_name, reference_name)
    if reference_name != label_names[top_choice_index]:
        raise ValueError(
            f"Reference row {row_number}: WebOrganizer label {top_choice_index} is "
            f"{reference_name!r}, but the label file has {label_names[top_choice_index]!r}; "
            "the label files must list labels in WebOrganizer's taxonomy order"
        )
    position = {label: letter for letter, label in enumerate(permutation)}
    return DistributionRecord(
        doc_id=doc_id,
        logprobs=normalize_logprobs([float(v) for v in choice_loss]),
        letter_positions=tuple(position[i] for i in range(n)),
        num_letters=n,
    )


def load_reference(
    candidate_ids: set[str],
    label_names: Sequence[str],
    dataset: str | None,
    split: str,
    jsonl: Path | None,
) -> tuple[dict[str, DistributionRecord], int]:
    """Collect reference records for the candidate documents.

    Streaming stops as soon as every candidate document has been found.

    Returns:
        The reference records by document ID and the number of rows read.

    Raises:
        ValueError: If some candidate documents are not in the reference.
    """
    found: dict[str, DistributionRecord] = {}
    rows_read = 0
    for rows_read, row in enumerate(stream_reference_rows(dataset, split, jsonl), start=1):
        record = reference_record(row, label_names, rows_read)
        if record.doc_id in candidate_ids:
            if record.doc_id in found:
                raise ValueError(f"Duplicate reference document {record.doc_id!r}")
            found[record.doc_id] = record
            if len(found) == len(candidate_ids):
                break
    missing = candidate_ids - set(found)
    if missing:
        raise ValueError(
            f"{len(missing)} candidate documents are not in the reference after "
            f"{rows_read} rows, e.g. {sorted(missing)[:3]!r}"
        )
    return found, rows_read


def label_routing(
    records: Sequence[DistributionRecord], labels: Sequence[str], extra_labels: Sequence[str]
) -> dict[str, Any]:
    """Summarize how often the added labels were chosen, before routing them back.

    Args:
        records: Candidate records over the full label set.
        labels: Short names of the full label set, in record order.
        extra_labels: Short names of the added labels.

    Returns:
        Per added label, the share of documents where it was the top label and
        its mean probability; and the same for all added labels together.
    """
    n = len(records)
    extra_indices = [labels.index(label) for label in extra_labels]
    tops = [top_index(r.logprobs) for r in records]
    probs = [probabilities(r.logprobs) for r in records]
    return {
        "routed_labels": list(extra_labels),
        "share_top_label_routed": sum(t in extra_indices for t in tops) / n,
        "mean_routed_probability": statistics.fmean(math.fsum(p[i] for i in extra_indices) for p in probs),
        "per_label": {
            label: {
                "share_top_label": sum(t == i for t in tops) / n,
                "mean_probability": statistics.fmean(p[i] for p in probs),
            }
            for label, i in zip(extra_labels, extra_indices)
        },
    }


def evaluate(
    candidates: Sequence[DistributionRecord],
    references: Sequence[DistributionRecord],
    labels: Sequence[str],
    num_bins: int,
    candidates_full: Sequence[DistributionRecord] | None = None,
) -> dict[str, Any]:
    """Compute all candidate-vs-reference metrics for aligned records.

    Args:
        candidates: Candidate records over the reference's labels.
        references: Reference records, aligned with ``candidates``.
        labels: Short names of the reference's labels.
        num_bins: Calibration bins.
        candidates_full: Candidate records over the full label set when labels
            were routed; letter bias is measured on these, because a routed
            record can never choose the letters of the removed labels.
    """
    n = len(candidates)
    pairs = [(top_index(r.logprobs), top_index(c.logprobs)) for c, r in zip(candidates, references)]
    correct = [a == b for a, b in pairs]
    macro_f1, per_label = f1_scores(pairs, labels)
    ranked = [ranked_indices(c.logprobs) for c in candidates]
    ece, bins = expected_calibration_error(
        [math.exp(max(c.logprobs)) for c in candidates], correct, num_bins
    )
    reference_probs = [probabilities(r.logprobs) for r in references]
    candidate_shares, candidate_vector = label_shares((b for _, b in pairs), labels)
    reference_shares, reference_vector = label_shares((a for a, _ in pairs), labels)
    return {
        "documents": n,
        "agreement": {
            "top1_agreement": sum(correct) / n,
            "cohen_kappa": cohen_kappa(pairs, len(labels)),
            "macro_f1": macro_f1,
            "reference_top_in_candidate_top2": sum(a in r[:2] for (a, _), r in zip(pairs, ranked)) / n,
            "reference_top_in_candidate_top3": sum(a in r[:3] for (a, _), r in zip(pairs, ranked)) / n,
        },
        "soft": {
            "jsd_bits": summarize(
                [jensen_shannon_bits(c.logprobs, r.logprobs) for c, r in zip(candidates, references)]
            ),
            # Log loss of the reference's top label under the candidate.
            "reference_label_log_loss_bits": statistics.fmean(
                -c.logprobs[a] / math.log(2) for c, (a, _) in zip(candidates, pairs)
            ),
            # Expected log loss under the full reference distribution.
            "cross_entropy_bits": statistics.fmean(
                -math.fsum(p * v for p, v in zip(rp, c.logprobs)) / math.log(2)
                for c, rp in zip(candidates, reference_probs)
            ),
            "brier": statistics.fmean(
                math.fsum(
                    (p - (1.0 if i == a else 0.0)) ** 2
                    for i, p in enumerate(probabilities(c.logprobs))
                )
                for c, (a, _) in zip(candidates, pairs)
            ),
        },
        "calibration": {"ece": ece, "bins": bins},
        "confidence": {
            "candidate": confidence_profile(candidates),
            "reference": confidence_profile(references),
        },
        "label_shares": {
            "divergence_bits": share_divergence_bits(candidate_vector, reference_vector),
            "candidate": candidate_shares,
            "reference": reference_shares,
        },
        "per_label": per_label,
        "letter_bias": {
            "candidate": letter_bias(candidates_full if candidates_full is not None else candidates),
            "reference": letter_bias(references),
        },
        "confusions": top_confusions(pairs, labels),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Compare label distributions with WebOrganizer's annotations."
    )
    parser.add_argument("--candidate", type=Path, required=True, help="Distribution JSONL")
    reference = parser.add_mutually_exclusive_group(required=True)
    reference.add_argument(
        "--reference-dataset",
        help="WebOrganizer dataset, e.g. WebOrganizer/TopicAnnotations-Llama-3.1-405B-FP8",
    )
    reference.add_argument("--reference-jsonl", type=Path, help="Local export of the dataset")
    parser.add_argument("--reference-split", default="train")
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help=(
            "Label YAML (topics.yaml or formats.yaml): WebOrganizer's labels in "
            "taxonomy order, optionally followed by the added labels"
        ),
    )
    parser.add_argument("--num-bins", type=int, default=10, help="Calibration bins")
    parser.add_argument("--output", type=Path, help="Write JSON report to this path")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default="INFO",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Evaluate one candidate file against WebOrganizer's annotations."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.num_bins < 1:
        parser.error("--num-bins must be >= 1")
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    all_names, all_short_names = load_label_names(args.labels)
    label_names, short_names, extra_labels = weborganizer_label_subset(all_names, all_short_names)
    candidate_labels, candidate_records = read_distribution_file(args.candidate)
    if candidate_labels not in (all_short_names, short_names):
        raise ValueError(
            f"{args.candidate} labels {candidate_labels!r} match neither the labels "
            f"in {args.labels} nor WebOrganizer's subset of them"
        )
    LOGGER.info("Loaded %d candidate documents from %s", len(candidate_records), args.candidate)
    reference_records, rows_read = load_reference(
        set(candidate_records),
        label_names,
        args.reference_dataset,
        args.reference_split,
        args.reference_jsonl,
    )
    LOGGER.info("Matched all candidate documents after %d reference rows", rows_read)
    candidates_full = list(candidate_records.values())
    references = [reference_records[c.doc_id] for c in candidates_full]
    routing = None
    candidates = candidates_full
    if candidate_labels != short_names:
        # Route the added labels back to WebOrganizer's labels by renormalizing.
        keep = [candidate_labels.index(label) for label in short_names]
        candidates = [restrict_to_labels(c, keep) for c in candidates_full]
        routing = label_routing(candidates_full, candidate_labels, extra_labels)
        LOGGER.info(
            "Routed %s back to WebOrganizer's labels; they were the top label for %.1f%% of documents",
            ", ".join(extra_labels),
            100 * routing["share_top_label_routed"],
        )
    report = {
        "candidate": str(args.candidate),
        "reference": args.reference_dataset or str(args.reference_jsonl),
        "reference_rows_read": rows_read,
        "label_routing": routing,
        **evaluate(
            candidates,
            references,
            short_names,
            args.num_bins,
            candidates_full=candidates_full if routing is not None else None,
        ),
    }
    write_report(report, args.output)
    return 0


if __name__ == "__main__":
    # A streamed dataset abandoned early can crash interpreter shutdown after
    # the report is written; exceptions still propagate normally.
    exit_without_finalization(main())
