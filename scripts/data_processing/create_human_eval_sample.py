"""Create CSV sheets of parallel documents for human topic and format labelling.

The documents come from the output of a label distribution run, which holds each
document's full original text, so annotators label exactly the documents and text
the models saw. The sample is the first ``--sample-size`` documents in dataset order,
the same documents in every language. Each language's sheet is shuffled differently
(unless ``--keep-order``), so annotators can't match translations by position. No
model output is written to the sheets.

Outputs (in ``--output-dir``):
    <language>.csv          one sheet per language with empty label columns
    topic_labels.csv,
    format_labels.csv       the label names to record, e.g. for dropdowns
    manifest.json           seed, sources, and the sampled document IDs
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import yaml  # type: ignore

LOGGER = logging.getLogger(__name__)

# Excel truncates cells longer than this; Google Sheets allows 50,000 characters.
EXCEL_CELL_LIMIT = 32_767
LABEL_COLUMNS = ("topic_label", "topic_explanation", "format_label", "format_explanation", "notes")


def read_texts(path: Path) -> dict[str, str]:
    """Return document ID -> text from a result JSONL file, in file order.

    Raises:
        ValueError: If a record lacks an ID or text, or an ID repeats.
    """
    texts: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            doc_id, text = record.get("id"), record.get("text")
            if not isinstance(doc_id, str) or not doc_id or not isinstance(text, str) or not text.strip():
                raise ValueError(f"{path}:{line_number}: record without an ID or text")
            if doc_id in texts:
                raise ValueError(f"{path}:{line_number}: duplicate document ID {doc_id!r}")
            texts[doc_id] = text
    return texts


def write_label_list(path: Path, labels_path: Path) -> None:
    """Write the label names annotators record, with their definitions."""
    labels = yaml.safe_load(labels_path.read_text(encoding="utf-8"))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, quoting=csv.QUOTE_ALL)
        writer.writerow(["label", "definition"])
        for label in labels:
            writer.writerow([label["name"], " ".join(str(label["definition"]).split())])


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description="Create CSV sheets for human labelling of parallel documents.")
    parser.add_argument(
        "--source-run",
        type=Path,
        default=Path("results/distribution/Qwen3.5-122B-A10B-annotations/topic"),
        help="Directory with <language>-parallel.jsonl result files holding the document texts",
    )
    parser.add_argument("--languages", nargs="+", default=["english", "finnish"])
    parser.add_argument("--sample-size", type=int, default=100, help="Take the first N documents")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the per-language row order")
    parser.add_argument(
        "--keep-order",
        action="store_true",
        help="Keep dataset order in every sheet instead of shuffling rows per language",
    )
    parser.add_argument("--labels-dir", type=Path, default=Path("scripts/data_processing"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/human_evaluation"))
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Sample documents and write the labelling sheets."""
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s [%(levelname)s] %(message)s")
    if args.sample_size < 1:
        raise ValueError("--sample-size must be >= 1")

    texts = {lang: read_texts(args.source_run / f"{lang}-parallel.jsonl") for lang in args.languages}
    first = args.languages[0]
    # Parallel documents share IDs; only documents present in every language qualify.
    candidates = [doc_id for doc_id in texts[first] if all(doc_id in texts[lang] for lang in args.languages)]
    if len(candidates) < len(texts[first]):
        raise ValueError(f"{len(texts[first]) - len(candidates)} documents are missing from some languages")

    if len(candidates) < args.sample_size:
        raise ValueError(f"Only {len(candidates)} documents available; --sample-size is {args.sample_size}")
    sample = candidates[: args.sample_size]
    LOGGER.info("Taking the first %d documents in dataset order", len(sample))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for lang in args.languages:
        # A separate shuffle per language, so translations can't be matched by position.
        order = list(sample)
        if not args.keep_order:
            random.Random(f"{args.seed}:{lang}").shuffle(order)
        path = args.output_dir / f"{lang}.csv"
        too_long = 0
        # utf-8-sig adds a byte-order mark so Excel shows non-ASCII letters correctly.
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle, quoting=csv.QUOTE_ALL)
            writer.writerow(["item", "doc_id", "language", "characters", "text", *LABEL_COLUMNS])
            for item, doc_id in enumerate(order, start=1):
                text = texts[lang][doc_id]
                too_long += len(text) > EXCEL_CELL_LIMIT
                writer.writerow([item, doc_id, lang, len(text), text, *([""] * len(LABEL_COLUMNS))])
        if too_long:
            LOGGER.warning("%s: %d texts exceed Excel's %d-character cell limit; use Google Sheets or LibreOffice", path, too_long, EXCEL_CELL_LIMIT)
        LOGGER.info("Wrote %s", path)

    for kind in ("topic", "format"):
        write_label_list(args.output_dir / f"{kind}_labels.csv", args.labels_dir / f"{kind}s.yaml")

    manifest = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_run": str(args.source_run),
        "languages": args.languages,
        "selection": f"first {len(sample)} documents in dataset order",
        "row_order": "dataset order" if args.keep_order else f"shuffled per language (seed {args.seed})",
        "sample_size": len(sample),
        "sampled_ids": sample,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Wrote %s", args.output_dir / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
