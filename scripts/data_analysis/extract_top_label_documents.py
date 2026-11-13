"""Extract the documents of chosen languages whose top topic or format label is one of chosen labels.

Reads a split of the topic dataset and the matching format dataset made by
``scripts/model_training/create_datasets.py`` (``save_to_disk`` Arrow shards that hold the same
documents in the same order), and writes one JSONL file per language with every document whose
most probable topic label or most probable format label (from the LLM soft labels) is one of
``--labels``. The default picks the two labels WebOrganizer does not have, Other and Bad example.

Only the ``language`` column is read for most rows, so the script is cheap even over sshfs; the
text and labels are read only for record batches that hold one of the chosen languages.

Example:
    python scripts/data_analysis/extract_top_label_documents.py \\
        --topic_dataset results/datasets/topic/hplt-finetune \\
        --format_dataset results/datasets/format/hplt-finetune \\
        --languages english finnish \\
        --output_dir results/statistics/other_bad_examples
"""

import argparse
import json
import logging
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

logger = logging.getLogger(__name__)

LABEL_TYPES = ("topic", "format")


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--topic_dataset", type=Path, required=True, help="Topic dataset directory (has labels.json)")
    parser.add_argument("--format_dataset", type=Path, required=True, help="Format dataset with the same documents")
    parser.add_argument("--split", default="train", help="Split to read (default: train)")
    parser.add_argument("--languages", nargs="+", required=True, help="Values of the language column, e.g. english")
    parser.add_argument(
        "--labels", nargs="+", default=["Other", "Bad example"], help="Top labels to extract (default: Other, Bad example)"
    )
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for <language>.jsonl")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output files")
    return parser.parse_args()


def label_names(dataset_dir: Path) -> list[str]:
    """Read the label names of a dataset.

    Args:
        dataset_dir: Dataset directory made by create_datasets.py.

    Returns:
        The label names in the order of the ``labels`` column.
    """
    with (dataset_dir / "labels.json").open(encoding="utf-8") as f:
        return list(json.load(f)["label_names"])


def shard_paths(dataset_dir: Path, split: str) -> list[Path]:
    """List the Arrow files of a split in dataset order.

    Args:
        dataset_dir: Dataset directory made by create_datasets.py.
        split: Split name.

    Returns:
        Paths of the split's Arrow files, in the order given by its state.json.
    """
    split_dir = dataset_dir / split
    with (split_dir / "state.json").open(encoding="utf-8") as f:
        files = [entry["filename"] for entry in json.load(f)["_data_files"]]
    if not files:
        raise ValueError(f"No data files listed in {split_dir / 'state.json'}")
    return [split_dir / name for name in files]


def record_batches(path: Path) -> Iterator[pa.RecordBatch]:
    """Yield the record batches of an Arrow stream file through a memory map.

    Args:
        path: Arrow IPC stream file written by ``datasets``.

    Yields:
        The file's record batches in order.
    """
    with pa.memory_map(str(path)) as source:
        yield from pa.ipc.open_stream(source)


def top_labels(labels: pa.Array, names: list[str]) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Rank each document's two most probable labels.

    Args:
        labels: Fixed-size list array of label probabilities, one row per document.
        names: Label names in column order.

    Returns:
        For each document, the top two labels with their probabilities, and the index of the
        top label.
    """
    width = len(names)
    probs = labels.flatten().to_numpy(zero_copy_only=False).reshape(-1, width)
    if probs.shape[0] != len(labels):
        raise ValueError(f"Expected {len(labels)} rows of {width} label probabilities, got {probs.shape}")
    order = np.argsort(-probs, axis=1)[:, :2]
    ranked = [
        {
            "top": names[first],
            "top_prob": round(float(row[first]), 4),
            "second": names[second],
            "second_prob": round(float(row[second]), 4),
        }
        for row, (first, second) in zip(probs, order)
    ]
    return ranked, order[:, 0]


def extract(args: argparse.Namespace) -> None:
    """Write the matching documents of each language to ``<output_dir>/<language>.jsonl``.

    Args:
        args: Parsed command-line arguments.

    Raises:
        ValueError: If a label is unknown, or if the topic and format datasets differ in shards,
            batches or document IDs.
        FileExistsError: If an output file exists and ``--overwrite`` is not given.
    """
    datasets = {"topic": args.topic_dataset, "format": args.format_dataset}
    names = {label_type: label_names(path) for label_type, path in datasets.items()}
    wanted = {}
    for label_type in LABEL_TYPES:
        missing = [label for label in args.labels if label not in names[label_type]]
        if missing:
            raise ValueError(f"Labels {missing} are not {label_type} labels of {datasets[label_type]}")
        wanted[label_type] = np.array([names[label_type].index(label) for label in args.labels])

    topic_shards = shard_paths(args.topic_dataset, args.split)
    format_shards = shard_paths(args.format_dataset, args.split)
    if [p.name for p in topic_shards] != [p.name for p in format_shards]:
        raise ValueError("The topic and format datasets have different shard files")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, TextIO] = {}
    for language in args.languages:
        path = args.output_dir / f"{language}.jsonl"
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    language_set = pa.array(args.languages)
    seen: Counter[str] = Counter()
    written: Counter[tuple[str, str]] = Counter()
    documents: Counter[str] = Counter()
    try:
        for language in args.languages:
            outputs[language] = (args.output_dir / f"{language}.jsonl").open("w", encoding="utf-8")
        for topic_path, format_path in zip(topic_shards, format_shards):
            shard_hits = 0
            for topic_batch, format_batch in zip(record_batches(topic_path), record_batches(format_path), strict=True):
                if topic_batch.num_rows != format_batch.num_rows:
                    raise ValueError(f"Batch sizes differ in {topic_path.name}")
                mask = pc.is_in(topic_batch.column("language"), value_set=language_set)
                if not pc.any(mask).as_py():
                    continue
                topic_rows = topic_batch.filter(mask)
                format_rows = format_batch.filter(mask)
                if topic_rows.column("id") != format_rows.column("id"):
                    raise ValueError(f"Document IDs differ between the datasets in {topic_path.name}")
                ranked = {}
                matched = {}
                for label_type, rows in (("topic", topic_rows), ("format", format_rows)):
                    ranked[label_type], first = top_labels(rows.column("labels"), names[label_type])
                    matched[label_type] = np.isin(first, wanted[label_type])
                columns = topic_rows.select(["id", "language", "source", "url", "text"]).to_pylist()
                for i, row in enumerate(columns):
                    seen[row["language"]] += 1
                    matched_on = [label_type for label_type in LABEL_TYPES if matched[label_type][i]]
                    if not matched_on:
                        continue
                    record = {
                        "id": row["id"],
                        "language": row["language"],
                        "source": row["source"],
                        "matched_on": "+".join(matched_on),
                        "topic": ranked["topic"][i],
                        "format": ranked["format"][i],
                        "url": row["url"],
                        "text": row["text"],
                    }
                    outputs[row["language"]].write(json.dumps(record, ensure_ascii=False) + "\n")
                    shard_hits += 1
                    documents[row["language"]] += 1
                    for label_type in matched_on:
                        written[(row["language"], f"{label_type}: {ranked[label_type][i]['top']}")] += 1
            if shard_hits:
                logger.info("%s: %d documents written", topic_path.name, shard_hits)
    finally:
        for output in outputs.values():
            output.close()

    for language in args.languages:
        if seen[language] == 0:
            raise ValueError(f"No {args.split} documents with language {language!r}; check the language names")
        counts = ", ".join(
            f"{label} {count:,}" for (lang, label), count in sorted(written.items()) if lang == language
        )
        logger.info(
            "%s: %s of %s documents written (%s)", language, f"{documents[language]:,}", f"{seen[language]:,}", counts
        )


def main() -> None:
    """Run the extraction from the command line."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    extract(parse_args())


if __name__ == "__main__":
    main()
