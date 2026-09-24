from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from sklearn.metrics import f1_score

Metrics = dict[str, Any]


def read_llm_records(path: Path) -> Iterator[tuple[str, str, str]]:
    """Stream and validate LLM result records from a JSONL file."""
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

                yield doc_id, labels[0], rationale
            except (json.JSONDecodeError, TypeError, AttributeError) as exc:
                raise ValueError(
                    f"Invalid label record in {path}:{line_number}: {exc}"
                ) from exc


def read_classifier_records(path: Path) -> Iterator[tuple[str, str]]:
    """Stream and validate classifier result records from a JSONL file."""
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

                doc_id = record.get("warc_record_id")
                label = record.get("predicted_label")

                if not isinstance(doc_id, str) or not doc_id:
                    raise TypeError("warc_record_id must be a non-empty string")
                if not isinstance(label, str) or not label:
                    raise TypeError("predicted_label must be a non-empty string")

                yield doc_id, label
            except (json.JSONDecodeError, TypeError, AttributeError) as exc:
                raise ValueError(
                    f"Invalid label record in {path}:{line_number}: {exc}"
                ) from exc


def calculate_metrics(
    llm_records: Iterable[tuple[str, str, str]],
    classifier_records: Iterable[tuple[str, str]],
) -> Metrics:
    """Compare labels and calculate aggregate and per-class metrics.

    Classifier labels are treated as the true labels and LLM labels as predictions.
    """
    llm_dict = {doc_id: (label, rationale) for doc_id, label, rationale in llm_records}
    classifier_dict = dict(classifier_records)

    y_true: list[str] = []
    y_pred: list[str] = []

    for doc_id, (llm_label, _rationale) in llm_dict.items():
        classifier_label = classifier_dict.get(doc_id)
        if classifier_label is None:
            raise ValueError(f"Missing classifier label for doc_id: {doc_id}")

        y_true.append(classifier_label)
        y_pred.append(llm_label)

    labels = sorted(set(y_true) | set(y_pred))
    if not y_true:
        return {
            "f1_weighted": 0.0,
            "f1_macro": 0.0,
            "f1_micro": 0.0,
            "accuracy": 0.0,
            "per_class": {},
        }

    f1_weighted = f1_score(y_true, y_pred, average="weighted", labels=labels)
    f1_macro = f1_score(y_true, y_pred, average="macro", labels=labels)
    f1_micro = f1_score(y_true, y_pred, average="micro", labels=labels)
    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
    per_class: dict[str, dict[str, Any]] = {}
    for class_label in labels:
        class_f1 = f1_score(
            y_true,
            y_pred,
            labels=[class_label],
            average="macro",
            zero_division=0,
        )
        mismatch_counts: dict[str, int] = {}
        mismatch_examples: dict[str, dict[str, str]] = {}
        for (doc_id, (_, rationale)), true_label, predicted_label in zip(
            llm_dict.items(), y_true, y_pred
        ):
            if true_label != class_label or predicted_label == class_label:
                continue
            mismatch_counts[predicted_label] = (
                mismatch_counts.get(predicted_label, 0) + 1
            )
            mismatch_examples.setdefault(
                predicted_label,
                {"label": predicted_label, "doc_id": doc_id, "rationale": rationale},
            )

        ranked_misclassifications = sorted(
            mismatch_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
        misclassifications = [
            {predicted_label: count}
            for predicted_label, count in ranked_misclassifications
        ]
        examples = [
            mismatch_examples[predicted_label]
            for predicted_label, _count in ranked_misclassifications[:5]
        ]
        per_class[class_label] = {
            "f1": round(float(class_f1), 3),
            "misclassifications": misclassifications,
            "examples": examples,
        }

    return {
        "f1_weighted": round(f1_weighted, 3),
        "f1_macro": round(f1_macro, 3),
        "f1_micro": round(f1_micro, 3),
        "accuracy": round(accuracy, 3),
        "per_class": per_class,
    }


def save_comparison_results(metrics: dict[str, Metrics], output_file: Path) -> None:
    """Persist comparison results to a JSON file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare LLM and classifier results")
    parser.add_argument(
        "--llm_file_dir",
        type=Path,
        required=True,
        help="Path to the directory containing LLM JSONL results",
    )
    parser.add_argument(
        "--classifier_file",
        type=Path,
        required=True,
        help="Path to the JSONL file containing classifier predictions",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        required=True,
        help="Path to the output JSON file for comparison results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    classifier_records = dict(read_classifier_records(args.classifier_file))

    metrics: dict[str, Metrics] = {}
    for llm_file in sorted(Path(args.llm_file_dir).glob("*.jsonl")):
        llm_records = read_llm_records(llm_file)
        metrics[llm_file.name] = calculate_metrics(
            llm_records, classifier_records.items()
        )

    save_comparison_results(metrics, args.output_file)


if __name__ == "__main__":
    main()
