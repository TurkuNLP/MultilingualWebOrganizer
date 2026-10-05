from __future__ import annotations

import argparse
import csv
import html
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

MULTILINGUAL_TASKS = ("topic", "format")
MULTILINGUAL_METRICS = (
    "full_match",
    "first_label_match",
    "first_label_and_same_set",
    "micro_precision",
    "micro_recall",
    "micro_f1",
    "mean_jaccard",
    "mean_ndcg",
)
CLASSIFIER_METRICS = ("f1_weighted", "f1_macro", "f1_micro", "accuracy")


def load_evaluation(path: Path) -> dict[str, Any]:
    """Load and validate one multilingual evaluation report."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read evaluation file {path}: {exc}") from exc

    languages = report.get("languages") if isinstance(report, dict) else None
    if not isinstance(languages, dict) or not languages:
        raise ValueError(f"{path} does not contain a non-empty 'languages' object")
    return report


def extract_language_metrics(
    evaluation: dict[str, Any], model: str, task: str, language: str
) -> dict[str, Any]:
    """Extract one language's multilingual metrics from an evaluation."""
    try:
        documents = int(evaluation["documents"])
        ordered = evaluation["ordered"]
        unordered = evaluation["unordered"]
        rank_aware = evaluation["rank_aware"]
        metrics = {
            "full_match": ordered["full_match"],
            "first_label_match": ordered["first_label_match"],
            "first_label_and_same_set": ordered["first_label_and_same_set"],
            "micro_precision": unordered["micro_precision"],
            "micro_recall": unordered["micro_recall"],
            "micro_f1": unordered["micro_f1"],
            "mean_jaccard": unordered["mean_jaccard"],
            "mean_ndcg": rank_aware["mean_ndcg"],
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid metrics for {model}/{task}/{language}: {exc}"
        ) from exc
    if documents < 1:
        raise ValueError(
            f"Document count must be positive for {model}/{task}/{language}"
        )
    return {"language": language, "documents": documents, **metrics}


def aggregate_report(report: dict[str, Any], model: str, task: str) -> dict[str, Any]:
    """Aggregate per-language metrics using document counts as weights."""
    weighted_totals = {metric: 0.0 for metric in MULTILINGUAL_METRICS}
    total_documents = 0

    for language, evaluation in report["languages"].items():
        if not isinstance(evaluation, dict):
            raise ValueError(f"Invalid evaluation for language {language!r}")
        language_result = extract_language_metrics(evaluation, model, task, language)
        documents = language_result["documents"]

        total_documents += documents
        for metric in MULTILINGUAL_METRICS:
            value = language_result[metric]
            weighted_totals[metric] += float(value) * documents

    return {
        "model": model,
        "task": task,
        "documents": total_documents,
        **{
            metric: weighted_totals[metric] / total_documents
            for metric in MULTILINGUAL_METRICS
        },
    }


def collect_language_results(eval_dir: Path) -> list[dict[str, Any]]:
    """Collect per-language multilingual metrics for plotting."""
    if not eval_dir.is_dir():
        raise FileNotFoundError(f"Evaluation directory does not exist: {eval_dir}")

    results: list[dict[str, Any]] = []
    for model_dir in sorted(path for path in eval_dir.iterdir() if path.is_dir()):
        for task in MULTILINGUAL_TASKS:
            path = model_dir / f"multilingual-label-evaluation-{task}.json"
            if not path.is_file():
                continue
            report = load_evaluation(path)
            for language, evaluation in report["languages"].items():
                if not isinstance(evaluation, dict):
                    raise ValueError(f"Invalid evaluation for language {language!r}")
                results.append(
                    {
                        "model": model_dir.name,
                        "task": task,
                        **extract_language_metrics(
                            evaluation, model_dir.name, task, language
                        ),
                    }
                )
    if not results:
        raise ValueError(f"No multilingual language results found in {eval_dir}")
    return results


def collect_results(eval_dir: Path) -> list[dict[str, Any]]:
    """Aggregate only multilingual label-evaluation reports by model and task."""
    if not eval_dir.is_dir():
        raise FileNotFoundError(f"Evaluation directory does not exist: {eval_dir}")

    results: list[dict[str, Any]] = []
    for model_dir in sorted(path for path in eval_dir.iterdir() if path.is_dir()):
        report_paths = {
            task: model_dir / f"multilingual-label-evaluation-{task}.json"
            for task in MULTILINGUAL_TASKS
        }
        if not any(path.is_file() for path in report_paths.values()):
            continue
        for task, path in report_paths.items():
            if not path.is_file():
                LOGGER.warning("Skipping missing evaluation: %s", path)
                continue
            results.append(
                aggregate_report(load_evaluation(path), model_dir.name, task)
            )
    if not results:
        raise ValueError(f"No multilingual evaluation reports found in {eval_dir}")
    return results


def format_table(results: list[dict[str, Any]], metric: str) -> str:
    """Format selected multilingual metrics as a readable text table."""
    headers = ("Model", "Topic", "Format")
    values: dict[str, dict[str, str]] = {}
    for result in results:
        values.setdefault(result["model"], {})[
            result["task"]
        ] = f"{result[metric] * 100:.2f}%"

    rows = [
        (
            model,
            values[model].get("topic", "-"),
            values[model].get("format", "-"),
        )
        for model in sorted(values)
    ]
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]

    def render(row: tuple[str, ...]) -> str:
        return " | ".join(value.ljust(width) for value, width in zip(row, widths))

    separator = "-+-".join("-" * width for width in widths)
    return "\n".join([render(headers), separator, *(render(row) for row in rows)])


def write_csv(results: list[dict[str, Any]], path: Path) -> None:
    """Write all aggregated metrics to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("model", "task", "documents", *MULTILINGUAL_METRICS),
        )
        writer.writeheader()
        writer.writerows(results)


def write_markdown(
    results: list[dict[str, Any]], path: Path, metrics: Sequence[str]
) -> None:
    """Write the selected metric table and all output columns to Markdown."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Multilingual label-evaluation comparison",
        "",
        "Values are document-weighted aggregates across the languages in each report.",
        "",
        "| Model | Task | Documents | " + " | ".join(MULTILINGUAL_METRICS) + " |",
        "|---|---|---:|" + "---:|" * len(MULTILINGUAL_METRICS),
    ]
    for result in results:
        values = [f"{result[metric_name]:.6f}" for metric_name in MULTILINGUAL_METRICS]
        lines.append(
            f"| {result['model']} | {result['task']} | {result['documents']} | "
            + " | ".join(values)
            + " |"
        )
    for metric in metrics:
        lines.extend(["", f"## {metric}", "", format_table(results, metric)])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_metric_results(
    results: list[dict[str, Any]], metrics: Sequence[str], path: Path
) -> None:
    """Create a readable SVG bar plot with one row for each result."""
    task_results = {
        task: sorted(
            (result for result in results if result["task"] == task),
            key=lambda result: (result["model"], result.get("run") or ""),
        )
        for task in MULTILINGUAL_TASKS
    }
    task_results = {task: rows for task, rows in task_results.items() if rows}
    left = 300
    right = 50
    row_height = 145
    row_top = 115
    chart_top_offset = 25
    chart_height = 85
    chart_bottom_offset = chart_top_offset + chart_height
    metric_width = 115
    legend_width = len(metrics) * 150 + 40
    width = max(1400, left + metric_width * len(metrics) + right, legend_width)
    height = (
        row_top + row_height * sum(len(rows) + 1 for rows in task_results.values()) + 20
    )
    bar_width = min(58, metric_width * 0.58)
    colors = (
        "#2563eb",
        "#f97316",
        "#16a34a",
        "#9333ea",
        "#dc2626",
        "#0891b2",
        "#ca8a04",
        "#4f46e5",
    )
    metric_colors = {
        metric: colors[index % len(colors)] for index, metric in enumerate(metrics)
    }
    title = ", ".join(metric.replace("_", " ").title() for metric in metrics)
    elements = [
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{width / 2:.0f}" y="38" text-anchor="middle" '
        f'font-family="sans-serif" font-size="28" font-weight="bold">'
        f"{html.escape(title)}</text>",
    ]

    legend_start = (width - legend_width) / 2
    for index, metric in enumerate(metrics):
        x = legend_start + index * 150
        elements.extend(
            [
                f'<rect x="{x:.1f}" y="60" width="20" height="20" fill="{metric_colors[metric]}"/>',
                f'<text x="{x + 28:.1f}" y="76" font-family="sans-serif" '
                f'font-size="15">{html.escape(metric)}</text>',
            ]
        )

    current_top = row_top
    for task, rows in task_results.items():
        elements.append(
            f'<text x="{left - 25}" y="{current_top + 25}" text-anchor="end" '
            f'font-family="sans-serif" font-size="22" font-weight="bold">'
            f"{html.escape(task.title())}</text>"
        )
        current_top += row_height
        for result in rows:
            chart_top = current_top + chart_top_offset
            bottom = current_top + chart_bottom_offset
            for tick in range(0, 6):
                value = tick / 5
                y = bottom - value * chart_height
                elements.append(
                    f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" '
                    'stroke="#ddd"/> '
                    f'<text x="{left - 10}" y="{y + 5:.1f}" text-anchor="end" '
                    f'font-family="sans-serif" font-size="15">{value:.0%}</text>'
                )
            group_label = result["model"]
            if result.get("run") is not None:
                group_label += f" / {result['run']}"
            elements.append(
                f'<text x="{left - 25}" y="{current_top + 70}" text-anchor="end" '
                f'font-family="sans-serif" font-size="16">'
                f"{html.escape(group_label)}</text>"
            )
            for metric_index, metric in enumerate(metrics):
                value = result[metric]
                metric_start = left + metric_index * metric_width
                x = metric_start + (metric_width - bar_width) / 2
                y = bottom - value * chart_height
                elements.append(
                    f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" '
                    f'height="{value * chart_height:.1f}" fill="{metric_colors[metric]}"/>'
                )
                elements.append(
                    f'<text x="{x + bar_width / 2:.1f}" y="{max(y - 8, chart_top + 16):.1f}" '
                    f'text-anchor="middle" font-family="sans-serif" font-size="14">'
                    f"{value:.3f}</text>"
                )
            elements.append(
                f'<line x1="{left}" y1="{bottom + 35}" x2="{width - right}" '
                f'y2="{bottom + 35}" stroke="#bbb"/>'
            )
            current_top += row_height

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        + "".join(elements)
        + "</svg>\n",
        encoding="utf-8",
    )


def plot_language_results(
    results: list[dict[str, Any]], metrics: Sequence[str], path: Path
) -> None:
    """Plot each language and its arithmetic mean for every model and task."""
    task_models: dict[str, dict[str, list[dict[str, Any]]]] = {
        task: {} for task in MULTILINGUAL_TASKS
    }
    for result in results:
        task_models.setdefault(result["task"], {}).setdefault(
            result["model"], []
        ).append(result)
    task_models = {task: models for task, models in task_models.items() if models}

    left = 300
    right = 50
    row_height = 165
    row_top = 125
    chart_top_offset = 25
    chart_height = 100
    chart_bottom_offset = chart_top_offset + chart_height
    group_width = max(155, len(metrics) * 34 + 45)
    max_groups = max(
        (
            len({result["language"] for rows in models.values() for result in rows}) + 1
            for models in task_models.values()
        ),
        default=1,
    )
    legend_width = len(metrics) * 150 + 40
    width = max(1400, left + group_width * max_groups + right, legend_width)
    height = (
        row_top
        + row_height * sum(len(models) + 1 for models in task_models.values())
        + 20
    )
    bar_width = min(30, group_width * 0.72 / max(len(metrics), 1))
    colors = (
        "#2563eb",
        "#f97316",
        "#16a34a",
        "#9333ea",
        "#dc2626",
        "#0891b2",
        "#ca8a04",
        "#4f46e5",
    )
    metric_colors = {
        metric: colors[index % len(colors)] for index, metric in enumerate(metrics)
    }
    title = ", ".join(metric.replace("_", " ").title() for metric in metrics)
    elements = [
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{width / 2:.0f}" y="38" text-anchor="middle" '
        f'font-family="sans-serif" font-size="28" font-weight="bold">'
        f"{html.escape(title)} by language</text>",
        f'<text x="{width / 2:.0f}" y="57" text-anchor="middle" '
        'font-family="sans-serif" font-size="14">Average is the arithmetic mean across languages</text>',
    ]

    legend_start = (width - legend_width) / 2
    for index, metric in enumerate(metrics):
        x = legend_start + index * 150
        elements.extend(
            [
                f'<rect x="{x:.1f}" y="70" width="20" height="20" fill="{metric_colors[metric]}"/>',
                f'<text x="{x + 28:.1f}" y="86" font-family="sans-serif" '
                f'font-size="15">{html.escape(metric)}</text>',
            ]
        )

    current_top = row_top
    for task, models in task_models.items():
        elements.append(
            f'<text x="{left - 25}" y="{current_top + 25}" text-anchor="end" '
            f'font-family="sans-serif" font-size="22" font-weight="bold">'
            f"{html.escape(task.title())}</text>"
        )
        current_top += row_height
        for model, model_results in sorted(models.items()):
            language_results = sorted(
                model_results, key=lambda result: result["language"]
            )
            average = {
                metric: sum(float(result[metric]) for result in language_results)
                / len(language_results)
                for metric in metrics
            }
            groups: list[tuple[str, dict[str, Any], bool]] = [
                (result["language"], result, False) for result in language_results
            ]
            groups.append(("Average", average, True))
            chart_top = current_top + chart_top_offset
            bottom = current_top + chart_bottom_offset
            for tick in range(0, 6):
                value = tick / 5
                y = bottom - value * chart_height
                elements.append(
                    f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" '
                    'stroke="#ddd"/> '
                    f'<text x="{left - 10}" y="{y + 5:.1f}" text-anchor="end" '
                    f'font-family="sans-serif" font-size="15">{value:.0%}</text>'
                )
            elements.append(
                f'<text x="{left - 25}" y="{current_top + 75}" text-anchor="end" '
                f'font-family="sans-serif" font-size="16">{html.escape(model)}</text>'
            )
            for group_index, (label, group_values, is_average) in enumerate(groups):
                group_start = left + group_index * group_width
                group_offset = (group_width - len(metrics) * bar_width) / 2
                for metric_index, metric in enumerate(metrics):
                    value = float(group_values[metric])
                    x = group_start + group_offset + metric_index * bar_width
                    y = bottom - value * chart_height
                    stroke = ' stroke="#111" stroke-width="2"' if is_average else ""
                    elements.append(
                        f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" '
                        f'height="{value * chart_height:.1f}" fill="{metric_colors[metric]}"'
                        f"{stroke}/>"
                    )
                    elements.append(
                        f'<text x="{x + bar_width / 2:.1f}" '
                        f'y="{max(y - 8, chart_top + 16):.1f}" '
                        'text-anchor="middle" font-family="sans-serif" font-size="13">'
                        f"{value:.3f}</text>"
                    )
                label_x = group_start + group_width / 2
                elements.append(
                    f'<text x="{label_x:.1f}" y="{bottom + 25}" '
                    'text-anchor="middle" font-family="sans-serif" '
                    f'font-size="13" font-weight="{"bold" if is_average else "normal"}">'
                    f"{html.escape(label)}</text>"
                )
            current_top += row_height

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        + "".join(elements)
        + "</svg>\n",
        encoding="utf-8",
    )


def collect_classifier_results(eval_dir: Path) -> list[dict[str, Any]]:
    """Load classifier-versus-LLM metrics from each model directory."""
    results: list[dict[str, Any]] = []
    for model_dir in sorted(path for path in eval_dir.iterdir() if path.is_dir()):
        for task in MULTILINGUAL_TASKS:
            path = model_dir / f"classifier_llm_{task}_comparison_results.json"
            if not path.is_file():
                continue
            try:
                report = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Could not read classifier evaluation {path}: {exc}"
                ) from exc
            if not isinstance(report, dict) or not report:
                raise ValueError(f"Invalid classifier evaluation report: {path}")
            for run_name, run_metrics in report.items():
                if not isinstance(run_metrics, dict):
                    raise ValueError(
                        f"Invalid classifier metrics in {path}: {run_name}"
                    )
                try:
                    metrics = {
                        metric: float(run_metrics[metric])
                        for metric in CLASSIFIER_METRICS
                    }
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Missing classifier metrics in {path}: {run_name}: {exc}"
                    ) from exc
                results.append(
                    {"model": model_dir.name, "task": task, "run": run_name, **metrics}
                )
    return results


def plot_classifier_results(results: list[dict[str, Any]], path: Path) -> None:
    """Plot each classifier run and its arithmetic mean for every model and task."""
    language_results = [
        {
            **result,
            "language": result["run"].removesuffix(".jsonl"),
        }
        for result in results
    ]
    plot_language_results(language_results, CLASSIFIER_METRICS, path)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Compare multilingual evaluation reports for any number of models."
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=Path("results/evals"),
        help="Directory containing one subdirectory per model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/evals/multilingual-label-evaluation-comparison"),
        help="Directory for multilingual label-evaluation comparison outputs.",
    )
    parser.add_argument(
        "--metric",
        dest="metrics",
        nargs="+",
        choices=MULTILINGUAL_METRICS,
        default=["full_match"],
        help="One or more multilingual metrics displayed in the table and plot.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the model comparison and write the requested artifacts."""
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    results = collect_results(args.eval_dir)

    csv_path = args.output_dir / "multilingual_label_evaluation_summary.csv"
    markdown_path = args.output_dir / "multilingual_label_evaluation_summary.md"
    plot_path = args.output_dir / "multilingual_label_evaluation_comparison.svg"
    classifier_plot_path = args.output_dir / "classifier_llm_comparison.svg"
    write_csv(results, csv_path)
    write_markdown(results, markdown_path, args.metrics)
    language_results = collect_language_results(args.eval_dir)
    plot_language_results(language_results, args.metrics, plot_path)
    classifier_results = collect_classifier_results(args.eval_dir)
    if classifier_results:
        plot_classifier_results(classifier_results, classifier_plot_path)

    for metric in args.metrics:
        print(format_table(results, metric))
    LOGGER.info("Wrote %s", csv_path)
    LOGGER.info("Wrote %s", markdown_path)
    LOGGER.info("Wrote %s", plot_path)
    if classifier_results:
        LOGGER.info("Wrote %s", classifier_plot_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
