from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


@dataclass
class EvaluationSummary:
    """Aggregate ordered and unordered multilabel agreement metrics."""

    documents: int = 0
    full_match: int = 0
    first_label_match: int = 0
    first_label_and_same_set: int = 0
    set_true_positives: int = 0
    set_false_positives: int = 0
    set_false_negatives: int = 0
    jaccard_sum: float = 0.0
    ndcg_sum: float = 0.0
    first_label_mismatch_counts: dict[tuple[str, str], int] = field(
        default_factory=dict
    )
    first_label_mismatch_examples: dict[tuple[str, str], dict[str, str]] = field(
        default_factory=dict
    )

    def add(
        self,
        reference: Sequence[str],
        candidate: Sequence[str],
        doc_id: str | None = None,
        candidate_rationale: str | None = None,
    ) -> None:
        """Add one candidate/reference label sequence to the summary."""
        reference_set = set(reference)
        candidate_set = set(candidate)
        intersection = reference_set & candidate_set
        first_label_pair = (reference[0], candidate[0])

        self.documents += 1
        self.full_match += int(reference == candidate)
        self.first_label_match += int(reference[0] == candidate[0])
        self.first_label_and_same_set += int(
            reference[0] == candidate[0] and reference_set == candidate_set
        )
        if reference[0] != candidate[0]:
            self.first_label_mismatch_counts[first_label_pair] = (
                self.first_label_mismatch_counts.get(first_label_pair, 0) + 1
            )
            if doc_id is not None:
                self.first_label_mismatch_examples.setdefault(
                    first_label_pair,
                    {
                        "doc_id": doc_id,
                        "rationale": candidate_rationale or "",
                    },
                )

        self.set_true_positives += len(intersection)
        self.set_false_positives += len(candidate_set - reference_set)
        self.set_false_negatives += len(reference_set - candidate_set)
        self.jaccard_sum += len(intersection) / len(reference_set | candidate_set)
        self.ndcg_sum += _ndcg(reference, candidate)

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-serializable aggregate metrics."""
        if not self.documents:
            raise ValueError("Cannot summarize an empty evaluation")

        precision_denominator = self.set_true_positives + self.set_false_positives
        recall_denominator = self.set_true_positives + self.set_false_negatives
        precision = (
            self.set_true_positives / precision_denominator
            if precision_denominator
            else 0.0
        )
        recall = (
            self.set_true_positives / recall_denominator if recall_denominator else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        first_label_mismatches = [
            {
                "reference_label": reference_label,
                "candidate_label": candidate_label,
                "count": count,
                "example": self.first_label_mismatch_examples.get(
                    (reference_label, candidate_label)
                ),
            }
            for (reference_label, candidate_label), count in sorted(
                self.first_label_mismatch_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ]
        return {
            "documents": self.documents,
            "ordered": {
                "full_match": self.full_match / self.documents,
                "first_label_match": self.first_label_match / self.documents,
                "first_label_and_same_set": self.first_label_and_same_set
                / self.documents,
                "first_label_mismatches": first_label_mismatches,
            },
            "unordered": {
                "micro_precision": precision,
                "micro_recall": recall,
                "micro_f1": f1,
                "mean_jaccard": self.jaccard_sum / self.documents,
            },
            "rank_aware": {"mean_ndcg": self.ndcg_sum / self.documents},
        }


def _ndcg(reference: Sequence[str], candidate: Sequence[str]) -> float:
    """Compute nDCG using reference rank as graded relevance."""
    relevance = {label: len(reference) - rank for rank, label in enumerate(reference)}
    if not relevance:
        return 0.0

    def discounted_gain(labels: Sequence[str]) -> float:
        return sum(
            relevance.get(label, 0) / math.log2(rank + 2)
            for rank, label in enumerate(labels)
        )

    ideal = discounted_gain(reference)
    return discounted_gain(candidate) / ideal if ideal else 0.0


def _read_records(path: Path) -> Iterator[tuple[int, str, list[str], str]]:
    """Stream and validate result records from a JSONL file."""
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Empty JSONL line in {path}:{line_number}")
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise TypeError("record must be a JSON object")
                doc_id = record.get("doc_id")
                result = record.get("result")
                labels = result.get("labels") if isinstance(result, dict) else None
                rationale = (
                    result.get("rationale") if isinstance(result, dict) else None
                )
                if not isinstance(doc_id, str) or not doc_id:
                    raise TypeError("doc_id must be a non-empty string")
                if (
                    not isinstance(labels, list)
                    or not labels
                    or any(not isinstance(label, str) or not label for label in labels)
                    or len(labels) != len(set(labels))
                ):
                    raise TypeError("result.labels must be unique non-empty strings")
                if not isinstance(rationale, str):
                    raise TypeError("result.rationale must be a string")
            except (json.JSONDecodeError, TypeError, AttributeError) as exc:
                raise ValueError(
                    f"Invalid label record in {path}:{line_number}: {exc}"
                ) from exc
            yield line_number, doc_id, labels, rationale


def evaluate_pair(
    reference_path: Path,
    candidate_path: Path,
    max_documents: int | None = None,
) -> EvaluationSummary:
    """Compare two same-document JSONL files in streaming order."""
    reference_records = _read_records(reference_path)
    candidate_records = _read_records(candidate_path)
    summary = EvaluationSummary()

    for reference_record, candidate_record in zip(
        reference_records, candidate_records, strict=False
    ):
        reference_line, reference_id, reference_labels, _reference_rationale = (
            reference_record
        )
        candidate_line, candidate_id, candidate_labels, candidate_rationale = (
            candidate_record
        )
        if reference_id != candidate_id:
            raise ValueError(
                f"Document mismatch at pair {summary.documents + 1}: "
                f"{reference_path}:{reference_line} has {reference_id!r}, "
                f"but {candidate_path}:{candidate_line} has {candidate_id!r}"
            )
        summary.add(
            reference_labels,
            candidate_labels,
            reference_id,
            candidate_rationale,
        )
        if max_documents is not None and summary.documents >= max_documents:
            return summary

    if max_documents is None or summary.documents < max_documents:
        try:
            next(reference_records)
        except StopIteration:
            pass
        else:
            raise ValueError(
                f"{candidate_path} has fewer records than {reference_path}"
            )
        try:
            next(candidate_records)
        except StopIteration:
            pass
        else:
            raise ValueError(f"{candidate_path} has more records than {reference_path}")

    if summary.documents == 0:
        raise ValueError("The input files contain no comparable records")
    return summary


def _parse_candidate(value: str) -> tuple[str, Path]:
    """Parse a CLI candidate in the form ``language=path``."""
    language, separator, path = value.partition("=")
    if not separator or not language.strip() or not path.strip():
        raise argparse.ArgumentTypeError("candidate must use LANGUAGE=PATH")
    return language.strip(), Path(path)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Compare ordered multilingual labels against English labels."
    )
    parser.add_argument("--english", type=Path, required=True, help="English JSONL")
    parser.add_argument(
        "--candidate",
        type=_parse_candidate,
        action="append",
        required=True,
        metavar="LANGUAGE=PATH",
        help="Candidate JSONL; repeat for multiple languages",
    )
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--output", type=Path, help="Write JSON report to this path")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default="INFO",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Evaluate all requested candidate language files."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.max_documents is not None and args.max_documents < 1:
        parser.error("--max-documents must be >= 1")

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    report: dict[str, Any] = {
        "reference": str(args.english),
        "languages": {},
    }
    for language, candidate_path in args.candidate:
        LOGGER.info("Evaluating %s against %s", candidate_path, args.english)
        summary = evaluate_pair(
            args.english,
            candidate_path,
            args.max_documents,
        )
        report["languages"][language] = summary.as_dict()

    serialized = json.dumps(report, indent=2) + "\n"
    if args.output is None:
        sys.stdout.write(serialized)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
        LOGGER.info("Wrote report to %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
