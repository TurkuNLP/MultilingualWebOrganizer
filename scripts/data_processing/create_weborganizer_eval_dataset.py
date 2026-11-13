"""Turn WebOrganizer's Llama-3.1-405B annotations into an evaluation dataset for finetune.py.

WebOrganizer (Wettig et al., 2025) trained and evaluated its classifiers on 100K English web
pages annotated by Llama-3.1-405B-Instruct-FP8, published as
``WebOrganizer/TopicAnnotations-Llama-3.1-405B-FP8`` and
``WebOrganizer/FormatAnnotations-Llama-3.1-405B-FP8`` with splits train (80K), validation (10K)
and test (10K). Its Table 7 reports the validation split.

This script writes chosen splits in the format of ``scripts/model_training/create_datasets.py``
(columns ``id``, ``language``, ``source``, ``url``, ``text``, ``labels``; ``labels.json``;
``README.md``), so that ``finetune.py --do_predict`` and the analysis notebooks treat them like
our own test data:

- ``labels`` is Llama's probability distribution over WebOrganizer's 24 labels
  (``choice_probs``, in the classifiers' label order) followed by 0 for our two extra labels,
  *Other* and *Bad example*. The label order is our label file's (``--reference_dataset``),
  whose first 24 labels must be WebOrganizer's labels in WebOrganizer's order.
- ``id`` is the WARC record ID without ``<urn:uuid:...>``, ``language`` is ``english`` and
  ``source`` is ``weborganizer_405b``.
- With ``--no_url`` the dataset has no ``url`` column, so ``finetune.py`` gives the model the
  text alone, as for our ``-nourl`` datasets.

Every row is checked: 24 finite, non-negative probabilities summing to 1, the largest at
``top_choice_index``, and the same label name (first line of ``top_choice``) for every use of
an index. Anything else raises an error.

Example (on Roihu, where the module has ``datasets``):
    python scripts/data_processing/create_weborganizer_eval_dataset.py --label_type topic \\
        --output_dir results/datasets/topic/weborganizer-405B
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
from collections import Counter
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import datasets

LOGGER = logging.getLogger(__name__)

HF_DATASETS = {
    "topic": "WebOrganizer/TopicAnnotations-Llama-3.1-405B-FP8",
    "format": "WebOrganizer/FormatAnnotations-Llama-3.1-405B-FP8",
}
# id2label of WebOrganizer/{Topic,Format}Classifier(-NoURL), checked 2026-11-13; all four agree.
WEBORGANIZER_LABELS = {
    "topic": [
        "Adult", "Art & Design", "Software Dev.", "Crime & Law", "Education & Jobs", "Hardware",
        "Entertainment", "Social Life", "Fashion & Beauty", "Finance & Business", "Food & Dining", "Games",
        "Health", "History", "Home & Hobbies", "Industrial", "Literature", "Politics", "Religion",
        "Science & Tech.", "Software", "Sports & Fitness", "Transportation", "Travel",
    ],
    "format": [
        "Academic Writing", "Content Listing", "Creative Writing", "Customer Support", "Comment Section", "FAQ",
        "Truncated", "Knowledge Article", "Legal Notices", "Listicle", "News Article", "Nonfiction Writing",
        "About (Org.)", "News (Org.)", "About (Pers.)", "Personal Blog", "Product Page", "Q&A Forum",
        "Spam / Ads", "Structured Data", "Documentation", "Audio Transcript", "Tutorial", "User Review",
    ],
}
EXTRA_LABELS = ["Other", "Bad example"]
SOURCE = "weborganizer_405b"
LANGUAGE = "english"
PROBABILITY_TOLERANCE = 1e-4


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command-line arguments.

    Args:
        argv: Arguments to parse instead of ``sys.argv``.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label_type", choices=sorted(HF_DATASETS), required=True)
    parser.add_argument("--output_dir", type=Path, required=True, help="Dataset directory to write")
    parser.add_argument(
        "--splits", nargs="+", default=["validation", "test"], help="WebOrganizer splits to include (default: validation test)"
    )
    parser.add_argument("--no_url", action="store_true", help="Leave out the url column (text-only input)")
    parser.add_argument(
        "--reference_dataset",
        type=Path,
        default=None,
        help="Dataset whose labels.json gives the label order (default: results/datasets/<label_type>/hplt-finetune "
        "next to --output_dir)",
    )
    parser.add_argument("--hf_dataset", default=None, help="Hugging Face dataset (default: by --label_type)")
    parser.add_argument("--cache_dir", default=None, help="Hugging Face datasets cache directory")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing --output_dir")
    return parser.parse_args(argv)


def read_label_info(reference_dataset: Path, label_type: str) -> dict[str, Any]:
    """Read and check the label file of the reference dataset.

    Args:
        reference_dataset: Dataset directory made by create_datasets.py.
        label_type: ``topic`` or ``format``.

    Returns:
        The reference dataset's labels.json contents.

    Raises:
        ValueError: If the labels are not WebOrganizer's 24 followed by Other and Bad example.
    """
    info = json.loads((reference_dataset / "labels.json").read_text(encoding="utf-8"))
    if info["label_type"] != label_type:
        raise ValueError(f"{reference_dataset} has {info['label_type']} labels, not {label_type}")
    expected = WEBORGANIZER_LABELS[label_type] + EXTRA_LABELS
    if info["label_names"] != expected:
        raise ValueError(f"{reference_dataset}/labels.json labels {info['label_names']} are not {expected}")
    return info


def record_id(metadata: dict[str, Any]) -> str:
    """Return the bare UUID of a WARC record ID such as ``<urn:uuid:7e67...>``.

    Raises:
        ValueError: If the record ID is missing or not in that form.
    """
    raw = metadata.get("WARC-Record-ID")
    prefix, suffix = "<urn:uuid:", ">"
    if not isinstance(raw, str) or not raw.startswith(prefix) or not raw.endswith(suffix):
        raise ValueError(f"Unexpected WARC-Record-ID {raw!r}")
    return raw[len(prefix) : -len(suffix)]


def checked_probabilities(row: dict[str, Any], where: str) -> list[float]:
    """Check a row's 24 label probabilities and return them renormalised.

    Args:
        row: A WebOrganizer annotation row.
        where: Description of the row for error messages.

    Returns:
        The probabilities, scaled to sum to exactly 1.

    Raises:
        ValueError: If the probabilities are malformed or disagree with ``top_choice_index``.
    """
    probs = row["choice_probs"]
    if len(probs) != 24:
        raise ValueError(f"{where}: {len(probs)} probabilities, expected 24")
    if not all(isinstance(p, (int, float)) and math.isfinite(p) and p >= 0 for p in probs):
        raise ValueError(f"{where}: probabilities must be finite and non-negative: {probs}")
    total = math.fsum(probs)
    if abs(total - 1) > PROBABILITY_TOLERANCE:
        raise ValueError(f"{where}: probabilities sum to {total}")
    top = max(range(24), key=probs.__getitem__)
    if top != row["top_choice_index"]:
        raise ValueError(f"{where}: largest probability at {top}, but top_choice_index is {row['top_choice_index']}")
    return [p / total for p in probs]


def generate_rows(
    hf_dataset: str, split: str, cache_dir: str | None, include_url: bool, choice_names: dict[int, str]
) -> Iterator[dict[str, Any]]:
    """Yield the dataset rows of one WebOrganizer split.

    Args:
        hf_dataset: Hugging Face dataset name.
        split: Split name.
        cache_dir: Hugging Face datasets cache directory.
        include_url: Include the url column.
        choice_names: Filled with label index -> first line of ``top_choice``, shared across splits,
            to check that an index always means the same label.

    Yields:
        Rows with id, language, source, (url,) text and labels.

    Raises:
        ValueError: On a malformed row, a missing URL or text, or an inconsistent label name.
    """
    source = datasets.load_dataset(hf_dataset, split=split, cache_dir=cache_dir)
    LOGGER.info("%s %s: %d rows", hf_dataset, split, len(source))
    for index, row in enumerate(source):
        where = f"{hf_dataset} {split} row {index}"
        probs = checked_probabilities(row, where)
        name = row["top_choice"].split("\n", 1)[0].strip()
        known = choice_names.setdefault(row["top_choice_index"], name)
        if known != name:
            raise ValueError(f"{where}: index {row['top_choice_index']} is {name!r} here but {known!r} elsewhere")
        text = row["text"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{where}: empty text")
        out = {"id": record_id(row["metadata"]), "language": LANGUAGE, "source": SOURCE}
        if include_url:
            if not isinstance(row["url"], str) or not row["url"]:
                raise ValueError(f"{where}: no URL")
            out["url"] = row["url"]
        out["text"] = text
        out["labels"] = probs + [0.0] * len(EXTRA_LABELS)
        yield out


def make_readme(args: argparse.Namespace, hf_dataset: str, sizes: dict[str, int], top_counts: dict[str, Counter]) -> str:
    """Describe how the dataset was made.

    Args:
        args: Parsed arguments.
        hf_dataset: Source Hugging Face dataset.
        sizes: Rows per split.
        top_counts: Per split, documents per Llama top label.

    Returns:
        README text in Markdown.
    """
    lines = [
        "# WebOrganizer evaluation dataset",
        "",
        "Created by `scripts/data_processing/create_weborganizer_eval_dataset.py` from "
        f"[{hf_dataset}](https://huggingface.co/datasets/{hf_dataset}) (Llama-3.1-405B-Instruct-FP8 "
        "annotations of English web pages, WebOrganizer's own classifier training and evaluation data).",
        "",
        f"- Label type: {args.label_type}; input: {'text only (no url column)' if args.no_url else 'URL and text'}",
        "- `labels`: Llama's distribution over WebOrganizer's 24 labels (`choice_probs`), then 0 for "
        "*Other* and *Bad example*; label order in `labels.json`.",
        f"- `id`: WARC record UUID; `language`: {LANGUAGE}; `source`: {SOURCE}.",
        "",
        "| Split | Documents |",
        "|---|---:|",
        *[f"| {split} | {count:,} |" for split, count in sizes.items()],
        "",
        "Documents per Llama top label:",
        "",
        "| Label | " + " | ".join(sizes) + " |",
        "|---|" + "---:|" * len(sizes),
    ]
    for label in WEBORGANIZER_LABELS[args.label_type]:
        lines.append(f"| {label} | " + " | ".join(f"{top_counts[split][label]:,}" for split in sizes) + " |")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    """Create the dataset.

    Args:
        argv: Arguments to parse instead of ``sys.argv``.

    Raises:
        FileExistsError: If the output exists and ``--overwrite`` is not given.
        ValueError: On malformed input or duplicate document IDs.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    hf_dataset = args.hf_dataset or HF_DATASETS[args.label_type]
    reference = args.reference_dataset or args.output_dir.parent / "hplt-finetune"
    label_info = read_label_info(reference, args.label_type)
    if args.output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} exists; pass --overwrite to replace it")

    columns = {name: datasets.Value("string") for name in ("id", "language", "source")}
    if not args.no_url:
        columns["url"] = datasets.Value("string")
    features = datasets.Features(
        {
            **columns,
            "text": datasets.Value("string"),
            "labels": datasets.List(datasets.Value("float32"), length=len(label_info["label_names"])),
        }
    )
    choice_names: dict[int, str] = {}
    splits = {}
    for split in args.splits:
        rows = list(generate_rows(hf_dataset, split, args.cache_dir, not args.no_url, choice_names))
        splits[split] = datasets.Dataset.from_list(rows, features=features)
    all_ids = [doc_id for split in splits.values() for doc_id in split["id"]]
    duplicates = [doc_id for doc_id, count in Counter(all_ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"{len(duplicates)} document IDs occur more than once, e.g. {duplicates[:5]}")

    names = WEBORGANIZER_LABELS[args.label_type]
    LOGGER.info("Label index -> classifier label -> name in Llama's prompt (check that they match):")
    for index in sorted(choice_names):
        LOGGER.info("  %2d %-20s %s", index, names[index], choice_names[index])
    top_counts = {
        split: Counter(names[max(range(24), key=row.__getitem__)] for row in dataset["labels"])
        for split, dataset in splits.items()
    }

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    datasets.DatasetDict(splits).save_to_disk(str(args.output_dir))
    (args.output_dir / "labels.json").write_text(json.dumps(label_info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    sizes = {split: len(dataset) for split, dataset in splits.items()}
    (args.output_dir / "README.md").write_text(make_readme(args, hf_dataset, sizes, top_counts), encoding="utf-8")
    LOGGER.info("Saved %s to %s", sizes, args.output_dir)


if __name__ == "__main__":
    main()
