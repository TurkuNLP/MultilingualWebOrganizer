"""Predict with Horizon Labs' multilingual WebOrganizer classifier on a dataset made for finetune.py.

``Horizon-Labs/multilingual-web-topic-format-base`` is mmBERT-base fine-tuned on Qwen3.8-27B
label distributions (WebOrganizer's prompt without URL or demonstrations) for about 152K pages
in 66 FineWeb-2 languages, read up to 512 tokens. It is one model with 48 sigmoid outputs:
24 ``topic_*`` and 24 ``format_*`` labels, in WebOrganizer's label order under WebOrganizer's
long label names (e.g. ``topic_electronics_hardware`` for *Hardware*).

This script runs it over a split of a dataset made by ``create_datasets.py`` or
``create_weborganizer_eval_dataset.py``, keeps the outputs of the dataset's label type, and
writes the same files as ``predict_weborganizer.py``:

- ``predictions.jsonl`` in the format of ``finetune.write_predictions``, keyed by our 24 shared
  label names. ``label_probs`` is the softmax of the label type's 24 logits: it ranks the
  labels like the model's sigmoid scores, but the model was not trained to give a
  distribution, so only the ranking (the top label) is meaningful.
- ``metrics.json`` with ``predict_weborganizer.shared_label_metrics`` (overall and per
  language), plus accuracy and macro-F1 computed as in Horizon Labs' ``webtax/metrics.py``.

**Input.** Text only (no URL), truncated to ``--max_length`` tokens (512 as in their training
and evaluation). ``--max_characters`` first cuts the text to that many characters, as their
English evaluation does (6,000).

**Reproducing their English evaluation.** Their ``webtax/build_evals.py`` draws 3,000 pages
with ``DataFrame.sample(3000, random_state=0)`` from the (single-file) test split of
WebOrganizer's Llama-3.1-405B annotations; topic and format share the same pages in the same
order. ``--sample_documents 3000 --sample_seed 0`` selects the same positions
(``numpy.random.RandomState(0).choice(n, 3000, replace=False)``, which is what pandas does),
and with ``--max_characters 6000`` the run should give their published English accuracies
(``eval/this_model.json``: topic 0.8173, format 0.688).

Example:
    python scripts/model_evaluation/predict_horizon.py --dataset_path results/datasets/topic/weborganizer-405B \\
        --split test --sample_documents 3000 --max_characters 6000 \\
        --output_dir results/models/topic/weborganizer-405B/horizon-base-test-sample3000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import datasets
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# The WebOrganizer prediction script's batching and metrics, so both comparisons are scored the
# same way; it also puts finetune.py on the path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict_weborganizer import arrow_column, predict, shared_label_metrics, softmax  # noqa: E402
from finetune import render_input, write_predictions  # noqa: E402

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL = "Horizon-Labs/multilingual-web-topic-format-base"
# The commit evaluated (2026-10-31), so reruns use the same weights.
DEFAULT_REVISION = "4ec2d09b46b89652ece638d9dee7865abcad6e69"
TEMPLATE = "{text}"
# Their output name (without the topic_/format_ prefix) for each of our 24 shared labels, in
# WebOrganizer's label order; from their webtax/run_weborganizer.py.
SLUGS = {
    "topic": {
        "Adult": "adult_content", "Art & Design": "art_design", "Software Dev.": "software_development",
        "Crime & Law": "crime_law", "Education & Jobs": "education_jobs", "Hardware": "electronics_hardware",
        "Entertainment": "entertainment", "Social Life": "social_life", "Fashion & Beauty": "fashion_beauty",
        "Finance & Business": "finance_business", "Food & Dining": "food_dining", "Games": "games", "Health": "health",
        "History": "history_geography", "Home & Hobbies": "home_hobbies", "Industrial": "industrial",
        "Literature": "literature", "Politics": "politics", "Religion": "religion",
        "Science & Tech.": "science_math_technology", "Software": "software", "Sports & Fitness": "sports_fitness",
        "Transportation": "transportation", "Travel": "travel_tourism",
    },
    "format": {
        "Academic Writing": "academic_writing", "Content Listing": "content_listing",
        "Creative Writing": "creative_writing", "Customer Support": "customer_support_page",
        "Comment Section": "discussion_forum_comment_section", "FAQ": "faqs", "Truncated": "incomplete_content",
        "Knowledge Article": "knowledge_article", "Legal Notices": "legal_notices", "Listicle": "listicle",
        "News Article": "news_article", "Nonfiction Writing": "nonfiction_writing",
        "About (Org.)": "organizational_about_page", "News (Org.)": "organizational_announcement",
        "About (Pers.)": "personal_about_page", "Personal Blog": "personal_blog", "Product Page": "product_page",
        "Q&A Forum": "q_a_forum", "Spam / Ads": "spam_ads", "Structured Data": "structured_data",
        "Documentation": "technical_writing", "Audio Transcript": "transcript_interview",
        "Tutorial": "tutorial_how_to_guide", "User Review": "user_reviews",
    },
}
# Their macro-F1 only averages labels with at least this many reference documents (webtax/metrics.py).
MACRO_F1_MIN_SUPPORT = 10


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command-line arguments.

    Args:
        argv: Arguments to parse instead of ``sys.argv``.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model name or path (default: {DEFAULT_MODEL})")
    parser.add_argument("--revision", default=DEFAULT_REVISION, help="Model commit (default: the one evaluated)")
    parser.add_argument("--dataset_path", type=Path, required=True, help="Dataset directory with labels.json")
    parser.add_argument("--split", required=True, help="Split to predict, e.g. test or validation")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_length", type=int, default=512, help="Tokens per document (their training: 512)")
    parser.add_argument("--max_characters", type=int, default=None,
                        help="Cut the text to this many characters first (their English evaluation: 6000)")
    parser.add_argument("--sample_documents", type=int, default=None,
                        help="Predict a seeded random sample of this many documents, drawn as pandas' DataFrame.sample")
    parser.add_argument("--sample_seed", type=int, default=0, help="Seed of --sample_documents (theirs: 0)")
    parser.add_argument("--max_batch_tokens", type=int, default=65536,
                        help="Padded tokens per batch (batch size x longest document)")
    parser.add_argument("--chunk_size", type=int, default=1024, help="Documents tokenized at a time")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True, help="bf16 autocast (default on)")
    parser.add_argument("--max_documents", type=int, default=None, help="Only the first N documents (for tests)")
    parser.add_argument("--device", default="cuda", help="Torch device (default cuda; cpu only for tests)")
    parser.add_argument("--cache_dir", default=None, help="Hugging Face cache directory")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs")
    args = parser.parse_args(argv)
    if args.sample_documents is not None and args.max_documents is not None:
        parser.error("--sample_documents and --max_documents are alternatives")
    for name in ("max_length", "max_characters", "sample_documents", "max_documents"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name} must be positive")
    return args


def label_columns(id2label: dict[int, str], label_type: str, shared_labels: Sequence[str]) -> list[int]:
    """Return the model's output columns for our shared labels, in our label order.

    Args:
        id2label: The model config's output names.
        label_type: ``topic`` or ``format``.
        shared_labels: Our 24 shared label names (the dataset's first 24 labels).

    Returns:
        One output index per shared label.

    Raises:
        ValueError: If the dataset's labels or the model's outputs don't match the mapping.
    """
    expected = list(SLUGS[label_type])
    if list(shared_labels) != expected:
        raise ValueError(f"The dataset's first 24 labels {list(shared_labels)} are not {expected}")
    columns = {name: index for index, name in id2label.items()}
    prefix = f"{label_type}_"
    outputs = sorted(name for name in columns if name.startswith(prefix))
    wanted = sorted(prefix + slug for slug in SLUGS[label_type].values())
    if outputs != wanted:
        raise ValueError(f"The model's {prefix}* outputs {outputs} are not {wanted}")
    return [columns[prefix + SLUGS[label_type][name]] for name in shared_labels]


def their_accuracy_and_macro_f1(predicted: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Accuracy and macro-F1 as in Horizon Labs' ``webtax/metrics.py`` (``acc_f1``).

    Macro-F1 averages the F1 of every reference label with at least ``MACRO_F1_MIN_SUPPORT``
    documents.

    Args:
        predicted: Predicted label indices.
        reference: Reference label indices.

    Returns:
        ``documents``, ``accuracy``, ``macro_f1`` and ``macro_f1_labels``.

    Raises:
        ValueError: If no label has enough reference documents.
    """
    scores = []
    for label in np.unique(reference):
        if (reference == label).sum() >= MACRO_F1_MIN_SUPPORT:
            true_positive = ((predicted == label) & (reference == label)).sum()
            false_positive = ((predicted == label) & (reference != label)).sum()
            false_negative = ((predicted != label) & (reference == label)).sum()
            scores.append(2 * true_positive / max(1, 2 * true_positive + false_positive + false_negative))
    if not scores:
        raise ValueError(f"No label has {MACRO_F1_MIN_SUPPORT} reference documents")
    return {"documents": int(len(reference)), "accuracy": float((predicted == reference).mean()),
            "macro_f1": float(np.mean(scores)), "macro_f1_labels": len(scores)}


def main(argv: Sequence[str] | None = None) -> None:
    """Predict and write the outputs.

    Args:
        argv: Arguments to parse instead of ``sys.argv``.

    Raises:
        FileExistsError: If outputs exist and ``--overwrite`` is not given.
        KeyError: If the dataset lacks the split.
        ValueError: On a label mismatch, missing weights, or a sample larger than the split.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    predictions_path = args.output_dir / "predictions.jsonl"
    metrics_path = args.output_dir / "metrics.json"
    for path in (predictions_path, metrics_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to replace it")

    label_info = json.loads((args.dataset_path / "labels.json").read_text(encoding="utf-8"))
    label_type, dataset_labels = label_info["label_type"], label_info["label_names"]
    if label_type not in SLUGS:
        raise ValueError(f"{args.dataset_path} has label type {label_type!r}")
    shared_labels = dataset_labels[:24]
    splits = datasets.load_from_disk(str(args.dataset_path))
    if args.split not in splits:
        raise KeyError(f"{args.dataset_path} has no split {args.split!r}; it has {list(splits)}")
    dataset = splits[args.split]
    sample_positions = None
    if args.sample_documents is not None:
        if args.sample_documents > len(dataset):
            raise ValueError(f"--sample_documents {args.sample_documents} > {len(dataset)} documents")
        # Equal to pandas' DataFrame.sample(n, random_state=seed) row positions.
        sample_positions = np.random.RandomState(args.sample_seed).choice(len(dataset), args.sample_documents,
                                                                          replace=False)
        dataset = dataset.select(sample_positions)
        LOGGER.info("Sampled %d documents (seed %d); first positions %s", len(dataset), args.sample_seed,
                    sample_positions[:5].tolist())
    if args.max_documents is not None:
        dataset = dataset.select(range(min(args.max_documents, len(dataset))))
    if args.max_characters is not None:
        limit = args.max_characters
        # keep_in_memory: don't write cache files into the dataset directory.
        dataset = dataset.map(lambda batch: {"text": [text[:limit] for text in batch["text"]]}, batched=True,
                              keep_in_memory=True)
    LOGGER.info("%s@%s, text input, on %s/%s (%s labels): %d documents", args.model, args.revision,
                args.dataset_path, args.split, label_type, len(dataset))
    LOGGER.info("First input:\n%s", render_input(dataset[0], TEMPLATE, decode_url=False)[:500])

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("No GPU available")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, cache_dir=args.cache_dir)
    model, loading_info = AutoModelForSequenceClassification.from_pretrained(
        args.model, revision=args.revision, cache_dir=args.cache_dir, dtype=torch.float32, output_loading_info=True
    )
    # A weight missing from the checkpoint would be randomly initialised; fail instead.
    if loading_info.get("missing_keys"):
        raise ValueError(f"Weights missing from the checkpoint: {sorted(loading_info['missing_keys'])}")
    if loading_info.get("unexpected_keys"):
        LOGGER.warning("Unused checkpoint weights: %s", sorted(loading_info["unexpected_keys"]))
    if model.config.problem_type != "multi_label_classification" or model.config.num_labels != 48:
        raise ValueError(f"Expected 48 multi-label outputs, got {model.config.num_labels} "
                         f"({model.config.problem_type})")
    columns = label_columns(model.config.id2label, label_type, shared_labels)
    LOGGER.info("Outputs for %s: %s", label_type,
                {name: model.config.id2label[column] for name, column in zip(shared_labels, columns)})
    LOGGER.info("Attention implementation: %s", model.config._attn_implementation)
    model.to(device).eval()

    started = time.time()
    # predict() reads max_length, max_batch_tokens, chunk_size and bf16 from args.
    logits = predict(model, tokenizer, dataset, TEMPLATE, args, device)[:, columns]
    elapsed = time.time() - started

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(predictions_path, dataset, logits, shared_labels)
    probs = softmax(logits)
    labels = arrow_column(dataset, "labels")
    targets = labels.flatten().to_numpy(zero_copy_only=False).astype(np.float64).reshape(len(dataset), -1)
    if targets.shape[1] != len(dataset_labels):
        raise ValueError(f"Targets have {targets.shape[1]} labels, labels.json {len(dataset_labels)}")
    # Their metrics compare the predicted and reference label on documents whose reference
    # label is one of the 24 (all of them for Llama's labels).
    target_top = targets.argmax(axis=1)
    counted = target_top < 24
    metrics: dict[str, Any] = {
        "model": args.model,
        "model_revision": args.revision,
        "label_type": label_type,
        "input": "text",
        "template": TEMPLATE,
        "dataset_path": str(args.dataset_path),
        "split": args.split,
        "sample_documents": args.sample_documents,
        "sample_seed": args.sample_seed if args.sample_documents is not None else None,
        "max_characters": args.max_characters,
        "max_length": args.max_length,
        "bf16": args.bf16,
        "seconds": elapsed,
        "documents_per_second": len(dataset) / elapsed,
        "overall": shared_label_metrics(probs, targets, shared_labels, dataset_labels),
        "their_metrics": their_accuracy_and_macro_f1(probs.argmax(axis=1)[counted], target_top[counted]),
    }
    if "language" in dataset.column_names:
        languages = np.asarray(arrow_column(dataset, "language").to_pylist())
        metrics["per_language"] = {
            language: shared_label_metrics(probs[languages == language], targets[languages == language],
                                           shared_labels, dataset_labels)
            for language in sorted(set(languages))
        }
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    overall = metrics["overall"]
    confident = overall["target_top_label_above_0.75"]
    LOGGER.info(
        "Counted %d of %d documents: accuracy %.4f (their macro-F1 %.4f); target top label > 0.75 (%d documents): "
        "accuracy %.4f, label average %.4f, worst label %.4f (%s), over %d labels",
        overall["documents_counted"], overall["documents"], overall["all"]["accuracy"],
        metrics["their_metrics"]["macro_f1"], confident["documents"], confident["accuracy"],
        confident["label_average"], confident["label_min"], confident["label_min_name"], confident["labels_counted"],
    )
    LOGGER.info("Wrote %s and %s", predictions_path, metrics_path)


if __name__ == "__main__":
    main()
