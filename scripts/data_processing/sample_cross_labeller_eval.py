"""Draw a per-language sample of a dataset split for labelling by other labellers.

The cross-labeller comparison (``notebooks/horizon_comparison.ipynb``) needs the same pages
labelled by labellers other than our Qwen prompt: Horizon Labs' teacher prompt and a neutral
judge. This script draws those pages once, so every labeller sees exactly the same ones:
``--per_language`` pages from each language of a split of a dataset made by
``scripts/model_training/create_datasets.py``, uniformly at random within each language
(seeded), leaving out the pages that duplicate training pages
(``results/statistics/duplicates/<label type>/<dataset>/duplicates.tsv``, ``leak`` = 1, of both
label types).

The topic and format datasets hold the same pages in the same order; the script checks this
on the ``id`` column and reads the text from the topic dataset.

Output (``--output_dir``):
- ``sample.parquet``: ``position`` (row in the split), ``id``, ``language``, ``source``,
  ``url`` and ``text``, sorted by language and position. Horizon's ``webtax/teacher_webtax.py``
  reads its ``text`` column and passes the others through.
- ``summary.json``: settings, and eligible and sampled pages per language.

A language with fewer eligible pages than ``--per_language`` contributes all of them; the
summary and the log list such languages.

``--languages`` restricts the sample to some languages (dataset language keys, e.g.
``simplified_chinese``), and ``--exclude_samples`` leaves out the pages of earlier samples, so a
follow-up sample adds new pages instead of relabelling ones that already have labels.

Reads Arrow shards with pyarrow only (no ``datasets``), so it runs with Roihu's python-data
module on a CPU node.

Examples:
    python scripts/data_processing/sample_cross_labeller_eval.py \\
        --topic_dataset results/datasets/topic/hplt-finetune --format_dataset results/datasets/format/hplt-finetune \\
        --split test --per_language 150 --output_dir results/statistics/cross_labeller/hplt-test-150

    # 1,000 more pages in each of 10 major languages, none of them in the first sample:
    python scripts/data_processing/sample_cross_labeller_eval.py \\
        --topic_dataset results/datasets/topic/hplt-finetune --format_dataset results/datasets/format/hplt-finetune \\
        --split test --per_language 1000 --languages english russian simplified_chinese german japanese \\
        spanish french italian portuguese polish \\
        --exclude_samples results/statistics/cross_labeller/hplt-test-150/sample.parquet \\
        --output_dir results/statistics/cross_labeller/hplt-test-major10-1000
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# The Arrow shard readers of the top-label extraction script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_analysis"))
from extract_top_label_documents import record_batches, shard_paths  # noqa: E402

LOGGER = logging.getLogger(__name__)

COLUMNS = ("id", "language", "source", "url", "text")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command-line arguments.

    Args:
        argv: Arguments to parse instead of ``sys.argv``.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--topic_dataset", type=Path, required=True, help="Topic dataset directory (has labels.json)")
    parser.add_argument("--format_dataset", type=Path, required=True, help="Format dataset with the same pages")
    parser.add_argument("--split", default="test", help="Split to sample (default: test)")
    parser.add_argument("--per_language", type=int, default=150, help="Pages per language (default: 150)")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed (default: 0)")
    parser.add_argument(
        "--duplicates_dir", type=Path, default=None,
        help="Directory with <label type>/<dataset name>/duplicates.tsv (default: results/statistics/duplicates)",
    )
    parser.add_argument(
        "--languages", nargs="+", default=None,
        help="Sample only these languages (dataset language keys, e.g. simplified_chinese; default: all)",
    )
    parser.add_argument(
        "--exclude_samples", nargs="+", type=Path, default=[],
        help="sample.parquet files of earlier samples of the same split whose pages are left out",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs")
    args = parser.parse_args(argv)
    if args.per_language < 1:
        parser.error("--per_language must be positive")
    if args.languages is not None and len(set(args.languages)) != len(args.languages):
        parser.error("--languages lists a language twice")
    return args


def excluded_ids(paths: Sequence[Path], split_ids: set[str]) -> set[str]:
    """Return the page IDs of earlier samples.

    Args:
        paths: ``sample.parquet`` files written by this script.
        split_ids: IDs of the split being sampled.

    Returns:
        The IDs of all pages in the given samples.

    Raises:
        ValueError: If a sample holds pages that are not in the split (it was drawn from another
            split or dataset).
    """
    ids: set[str] = set()
    for path in paths:
        sample_ids = set(pq.read_table(path, columns=["id"]).column("id").to_pylist())
        unknown = sample_ids - split_ids
        if unknown:
            raise ValueError(f"{path}: {len(unknown)} pages are not in the split, e.g. {sorted(unknown)[:3]}")
        LOGGER.info("Leaving out the %d pages of %s", len(sample_ids), path)
        ids |= sample_ids
    return ids


def read_columns(dataset_dir: Path, split: str, columns: Sequence[str]) -> pa.Table:
    """Read some columns of a split from its Arrow shards, in dataset order.

    Args:
        dataset_dir: Dataset directory made by create_datasets.py.
        split: Split name.
        columns: Columns to keep.

    Returns:
        The columns as one table.

    Raises:
        KeyError: If a column is missing.
    """
    tables = []
    for path in shard_paths(dataset_dir, split):
        for batch in record_batches(path):
            missing = [name for name in columns if name not in batch.schema.names]
            if missing:
                raise KeyError(f"{path} has no columns {missing}")
            tables.append(pa.Table.from_batches([batch]).select(list(columns)))
    return pa.concat_tables(tables)


def leaked_ids(duplicates_dir: Path, dataset_dir: Path, split: str, documents: int) -> set[str]:
    """Return the IDs of the split's pages that duplicate training pages.

    Args:
        duplicates_dir: Root of find_duplicates.py outputs.
        dataset_dir: The dataset (its label type and name locate the outputs).
        split: Split name.
        documents: Pages in the split, checked against the duplicate search's summary.

    Returns:
        IDs with ``leak`` = 1 in the split.

    Raises:
        ValueError: If the duplicate search ran on a different number of pages.
    """
    directory = duplicates_dir / dataset_dir.parent.name / dataset_dir.name
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if summary["documents"].get(split) != documents:
        raise ValueError(f"{directory}: {summary['documents'].get(split)} {split} pages, the dataset has {documents}")
    with (directory / "duplicates.tsv").open(encoding="utf-8", newline="") as handle:
        return {row["id"] for row in csv.DictReader(handle, delimiter="\t") if row["split"] == split and row["leak"] == "1"}


def main(argv: Sequence[str] | None = None) -> None:
    """Draw the sample and write it.

    Args:
        argv: Arguments to parse instead of ``sys.argv``.

    Raises:
        FileExistsError: If outputs exist and ``--overwrite`` is not given.
        ValueError: If the topic and format datasets hold different pages, IDs repeat, a requested
            language is not in the split, or an earlier sample holds pages from elsewhere.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    sample_path = args.output_dir / "sample.parquet"
    summary_path = args.output_dir / "summary.json"
    for path in (sample_path, summary_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    duplicates_dir = args.duplicates_dir or args.topic_dataset.parents[2] / "statistics" / "duplicates"

    table = read_columns(args.topic_dataset, args.split, COLUMNS)
    ids = table.column("id").to_pylist()
    if len(set(ids)) != len(ids):
        raise ValueError(f"{args.topic_dataset}/{args.split}: duplicate page IDs")
    format_ids = read_columns(args.format_dataset, args.split, ["id"]).column("id").to_pylist()
    if format_ids != ids:
        raise ValueError(f"{args.format_dataset}/{args.split} doesn't hold the topic dataset's pages in the same order")
    leaked = set()
    for dataset in (args.topic_dataset, args.format_dataset):
        leaked |= leaked_ids(duplicates_dir, dataset, args.split, len(ids))
    unknown = leaked - set(ids)
    if unknown:
        raise ValueError(f"{len(unknown)} duplicate IDs are not in the split, e.g. {sorted(unknown)[:3]}")
    LOGGER.info("%s/%s: %d pages, %d left out as duplicates of training pages", args.topic_dataset, args.split,
                len(ids), len(leaked))

    excluded = excluded_ids(args.exclude_samples, set(ids))

    languages = np.asarray(table.column("language").to_pylist(), dtype=object)
    sampled_languages = sorted(set(languages))
    if args.languages is not None:
        unknown_languages = sorted(set(args.languages) - set(sampled_languages))
        if unknown_languages:
            raise ValueError(f"languages not in {args.topic_dataset}/{args.split}: {unknown_languages}")
        sampled_languages = sorted(args.languages)
    eligible = np.array([doc_id not in leaked and doc_id not in excluded for doc_id in ids])
    rng = np.random.default_rng(args.seed)
    chosen: list[np.ndarray] = []
    per_language: dict[str, dict[str, int]] = {}
    for language in sampled_languages:
        positions = np.flatnonzero((languages == language) & eligible)
        take = min(args.per_language, len(positions))
        chosen.append(np.sort(rng.choice(positions, size=take, replace=False)))
        per_language[language] = {"eligible": int(len(positions)), "sampled": int(take)}
    short = {language: counts["eligible"] for language, counts in per_language.items()
             if counts["eligible"] < args.per_language}
    if short:
        LOGGER.warning("%d languages have fewer than %d eligible pages and contribute all of them: %s",
                       len(short), args.per_language, short)
    positions = np.concatenate(chosen)
    sample = table.take(pa.array(positions)).add_column(0, "position", pa.array(positions, type=pa.int64()))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(sample, sample_path)
    summary: dict[str, Any] = {
        "topic_dataset": str(args.topic_dataset), "format_dataset": str(args.format_dataset), "split": args.split,
        "per_language": args.per_language, "seed": args.seed, "duplicates_dir": str(duplicates_dir),
        "split_pages": len(ids), "left_out_as_duplicates": len(leaked),
        "languages_requested": args.languages, "exclude_samples": [str(path) for path in args.exclude_samples],
        "left_out_from_earlier_samples": len(excluded), "sampled_pages": int(len(positions)),
        "languages": len(per_language), "languages_short": short,
        "sources": dict(Counter(sample.column("source").to_pylist())), "per_language_counts": per_language,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    LOGGER.info("Wrote %d pages in %d languages to %s", len(positions), len(per_language), sample_path)


if __name__ == "__main__":
    main()
