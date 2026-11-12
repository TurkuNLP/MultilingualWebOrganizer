"""Count documents and mmBERT tokens per language, and write them to a CSV file.

The documents are read either from a dataset made by ``create_datasets.py``
(``--dataset``: exactly the classifier's training data) or from label distribution
results (``--input``, with the same per-file caps as ``create_datasets.py``, so the counts
can be made before the dataset is built). In both cases the language is the one
``create_datasets.py`` assigns: the result file's name up to the first hyphen.

Tokens are counted on the classifier's input, built exactly as ``finetune.py`` builds it
(``finetune.render_input``): by default the percent-decoded URL, a blank line, and the
text, or the text alone with ``--template '{text}'``. Without ``--max_length`` the count
is of the whole input without special tokens, i.e. how much text there is; with
``--max_length 8192`` it is what the model sees in training (with the two special tokens,
cut at 8,192 tokens).

The CSV has one row per language, in the order of the language file
(``data/qwen_label_languages.txt``, lines ``<Language>: <code>``), with the columns
``Language``, ``Language code``, ``Documents`` and ``Tokens``. A language is matched to the
file by its name in lower case with spaces as underscores (``Norwegian Bokmål`` is
``norwegian_bokmål``); a language in the data without a line in the file stops the
script, and languages of the file without data are left out with a warning.

Example (from the repository root, on a Roihu GPU node or in hpc/roihu/count_language_tokens.sh):
    python scripts/model_training/count_language_tokens.py --output results/statistics/hplt-random.csv \\
        --input results/distribution/Qwen3.8-27B-hplt-annotations/{clean,noisy,poor}_random/topic \\
        --max_per_file 25000 25000 10000
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import multiprocessing
from collections import Counter
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import datasets

# Sibling modules: the script's directory is on sys.path. finetune.render_input builds the
# classifier input, so the counts match training; create_datasets reads results the same
# way the dataset is built from them.
from create_datasets import find_input_files, input_file_limits, language_of, read_lines
from finetune import TEXT_TEMPLATE, URL_TEMPLATE, render_input

LOGGER = logging.getLogger(__name__)

DEFAULT_TOKENIZER = "jhu-clsp/mmBERT-base"
BATCH_SIZE = 1000

# Per-process state of the worker pool, set by _init_worker.
_TOKENIZER: Any = None
_SETTINGS: dict[str, Any] = {}


def read_language_codes(path: Path) -> dict[str, tuple[str, str]]:
    """Return file-name key -> (language name, language code) from ``<Language>: <code>`` lines.

    Raises:
        ValueError: If a line is malformed or two languages map to the same key.
    """
    languages: dict[str, tuple[str, str]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        if ":" not in line:
            raise ValueError(f"{path}:{line_number}: expected '<Language>: <code>', got {line!r}")
        name, code = (part.strip() for part in line.split(":", 1))
        if not name or not code or " " in code:
            raise ValueError(f"{path}:{line_number}: expected one language and one code, got {line!r}")
        key = name.lower().replace(" ", "_")
        if key in languages:
            raise ValueError(f"{path}:{line_number}: {name!r} repeats another language's name")
        languages[key] = (name, code)
    return languages


def _init_worker(tokenizer_name: str, template: str, decode_url: bool, max_length: int | None) -> None:
    """Load the tokenizer once per worker process."""
    global _TOKENIZER, _SETTINGS
    from transformers import AutoTokenizer

    _TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_name)
    _SETTINGS = {"template": template, "decode_url": decode_url, "max_length": max_length}


def count_tokens(texts: list[str]) -> int:
    """Return the total number of tokens of the inputs, as configured in the worker."""
    max_length = _SETTINGS["max_length"]
    if max_length is None:
        encoded = _TOKENIZER(texts, add_special_tokens=False, return_attention_mask=False)
    else:
        # As finetune.tokenize_split: special tokens included, cut at max_length.
        encoded = _TOKENIZER(texts, truncation=True, max_length=max_length, return_attention_mask=False,
                             return_token_type_ids=False)
    return sum(len(ids) for ids in encoded["input_ids"])


def count_file(task: tuple[str, int | None]) -> tuple[str, int, int]:
    """Count the documents and tokens of the first ``limit`` records of a result file.

    Returns:
        The language, the number of documents, and the number of tokens.
    """
    path_name, limit = task
    path = Path(path_name)
    documents = tokens = 0
    batch: list[str] = []
    for line_number, line in read_lines(path, limit):
        if not line.strip():
            raise ValueError(f"{path}:{line_number}: empty line")
        record = json.loads(line)
        batch.append(render_input(record, _SETTINGS["template"], _SETTINGS["decode_url"]))
        documents += 1
        if len(batch) == BATCH_SIZE:
            tokens += count_tokens(batch)
            batch = []
    if batch:
        tokens += count_tokens(batch)
    if limit is not None and documents < limit:
        LOGGER.warning("%s has %d documents, fewer than the cap of %d; counting all of them", path, documents, limit)
    return language_of(path), documents, tokens


def count_dataset_rows(task: tuple[str, str, int, int]) -> tuple[Counter[str], Counter[str]]:
    """Count documents and tokens per language in a slice of a saved dataset split.

    Returns:
        Documents per language and tokens per language.
    """
    dataset_path, split, start, stop = task
    dataset = datasets.load_from_disk(dataset_path)[split].select(range(start, stop))
    documents: Counter[str] = Counter()
    tokens: Counter[str] = Counter()
    for batch in dataset.iter(batch_size=BATCH_SIZE):
        rows = [dict(zip(batch, values)) for values in zip(*batch.values())]
        by_language: dict[str, list[str]] = {}
        for row in rows:
            by_language.setdefault(row["language"], []).append(
                render_input(row, _SETTINGS["template"], _SETTINGS["decode_url"]))
        for language, texts in by_language.items():
            documents[language] += len(texts)
            tokens[language] += count_tokens(texts)
    return documents, tokens


def dataset_tasks(dataset_path: Path, splits: Sequence[str] | None, workers: int) -> Iterator[tuple[str, str, int, int]]:
    """Yield (dataset, split, start, stop) slices that spread a dataset over the workers.

    Raises:
        KeyError: If a requested split is not in the dataset.
    """
    dataset = datasets.load_from_disk(str(dataset_path))
    for split in splits or list(dataset):
        if split not in dataset:
            raise KeyError(f"{dataset_path} has no split {split!r}; it has {list(dataset)}")
        rows = len(dataset[split])
        step = max(BATCH_SIZE, -(-rows // (workers * 4)))
        for start in range(0, rows, step):
            yield str(dataset_path), split, start, min(start + step, rows)


def write_csv(path: Path, documents: Counter[str], tokens: Counter[str], languages: dict[str, tuple[str, str]]) -> None:
    """Write one row per language, in language-file order.

    Raises:
        ValueError: If a language in the data has no line in the language file.
    """
    unknown = sorted(set(documents) - set(languages))
    if unknown:
        raise ValueError(f"Languages without a line in the language file: {unknown}")
    absent = [name for key, (name, _) in languages.items() if key not in documents]
    if absent:
        LOGGER.warning("%d languages of the language file have no documents and are left out: %s",
                       len(absent), ", ".join(absent))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Language", "Language code", "Documents", "Tokens"])
        for key, (name, code) in languages.items():
            if key in documents:
                writer.writerow([name, code, documents[key], tokens[key]])


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description="Count documents and mmBERT tokens per language into a CSV file.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", type=Path, help="Dataset directory made by create_datasets.py")
    source.add_argument("--input", type=Path, nargs="+",
                        help="Label distribution result JSONL files, or directories of them")
    parser.add_argument("--splits", nargs="+", help="With --dataset: the splits to count (default: all)")
    parser.add_argument("--max_per_file", type=int, nargs="+",
                        help="With --input: documents taken from each file, one value or one per --input "
                             "path (as in create_datasets.py)")
    parser.add_argument("--output", type=Path, required=True, help="CSV file to write")
    parser.add_argument("--languages_file", type=Path, default=Path("data/qwen_label_languages.txt"),
                        help="Lines '<Language>: <code>' giving the names, codes and row order")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--template",
                        help=f"Classifier input template (default: {URL_TEMPLATE!r}, or {TEXT_TEMPLATE!r} "
                             "for a dataset without URLs)")
    parser.add_argument("--decode_url", choices=("true", "false"), default="true",
                        help="Percent-decode URLs as finetune.py does by default")
    parser.add_argument("--max_length", type=int,
                        help="Count what the model sees: special tokens included, cut at this many tokens "
                             "(e.g. 8192). Default: the whole input without special tokens")
    parser.add_argument("--workers", type=int, default=max(1, (multiprocessing.cpu_count() or 2) - 1))
    parser.add_argument("--log_level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Count documents and tokens per language and write the CSV."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s [%(levelname)s] %(message)s")
    if args.max_per_file is not None and args.input is None:
        parser.error("--max_per_file needs --input")
    if args.splits is not None and args.dataset is None:
        parser.error("--splits needs --dataset")
    if args.max_length is not None and args.max_length < 2:
        parser.error("--max_length must be at least 2")
    if args.workers < 1:
        parser.error("--workers must be positive")

    languages = read_language_codes(args.languages_file)
    if args.dataset is not None:
        dataset = datasets.load_from_disk(str(args.dataset))
        has_url = {split: "url" in dataset[split].column_names for split in dataset}
        template = args.template or (URL_TEMPLATE if all(has_url.values()) else TEXT_TEMPLATE)
        tasks: list[Any] = list(dataset_tasks(args.dataset, args.splits, args.workers))
    else:
        template = args.template or URL_TEMPLATE
        files = find_input_files(args.input)
        limits = input_file_limits(args.input, args.max_per_file)
        tasks = [(str(path), limits[str(path)]) for path in files]
    LOGGER.info("Counting %d tasks with %d workers; template %r, decode_url %s, max_length %s",
                len(tasks), args.workers, template, args.decode_url, args.max_length)

    documents: Counter[str] = Counter()
    tokens: Counter[str] = Counter()
    init_args = (args.tokenizer, template, args.decode_url == "true", args.max_length)
    with multiprocessing.Pool(args.workers, initializer=_init_worker, initargs=init_args) as pool:
        if args.dataset is not None:
            for docs, toks in pool.imap_unordered(count_dataset_rows, tasks):
                documents.update(docs)
                tokens.update(toks)
        else:
            for done, (language, docs, toks) in enumerate(pool.imap_unordered(count_file, tasks), start=1):
                documents[language] += docs
                tokens[language] += toks
                if done % 50 == 0 or done == len(tasks):
                    LOGGER.info("%d of %d files counted", done, len(tasks))

    write_csv(args.output, documents, tokens, languages)
    LOGGER.info("%d languages, %d documents, %d tokens -> %s", len(documents), sum(documents.values()),
                sum(tokens.values()), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
