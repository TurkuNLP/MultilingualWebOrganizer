"""Label an HPLT 4.0 file with the fine-tuned topic and format classifiers.

Reads one HPLT 4.0 file (``<shard>_<part>.jsonl.zst``: one JSON document per line, with the
document ID in ``id``, the URL in ``u`` and the text in ``text``) and runs the topic and format
classifiers trained by ``finetune.py`` on every document. The two models share mmBERT's
tokenizer and their input, so each document is read and tokenized once for both.

**Input.** The model input is built as in training, with ``finetune.render_input`` and the
template and URL decoding saved in the models' ``config.json`` (``input_template``
``"{url}\\n\\n{text}"``, ``decode_url`` true): the percent-decoded URL, a blank line and the
text, truncated to ``--max_length`` tokens (8,192, as in training), run with bf16 autocast on
fp32 weights (as ``finetune.py --bf16``). Only models whose input has the URL are accepted, not
the ``-nourl`` ones. A malformed line, or a document without an ID, URL or text, stops the run
with an error.

**Output.** ``<output_root>/<language>/<shard>_<part>.jsonl.zst``, one line per input document,
in input order, with only the document's ID, URL and text (as in the input: the URL not
percent-decoded, the text not truncated) and the predictions; HPLT's other fields are left out::

    {"id": "...", "url": "...", "text": "...", "topic": "Politics", "topic_probs": [...],
     "format": "News Article", "format_probs": [...]}

``topic`` and ``format`` are the most probable labels, and ``topic_probs`` and ``format_probs``
the full probability distributions over all labels, rounded to ``--decimals`` decimals, in the
label order of ``<output_root>/metadata.json`` (which also records the models and settings). Every file under one output root is made with the same models and
settings: a run with others stops with an error, so use a new output root for them.

**Resuming.** Results are appended to ``<shard>_<part>.jsonl.partial`` as they are made, and
the file is compressed under its final name (and checked to hold one line per document) only
once every document is labelled. A rerun skips a finished file and continues a partial one,
after checking that its IDs are those of the input's first documents; a line cut off by a
killed job is dropped and labelled again. ``--max_documents`` stops early for tests and keeps
the partial file, which a later full run continues.

**Locking.** While a file is labelled, ``<shard>_<part>.lock`` next to its output holds the
owner (SLURM job ID, host, process ID), so a second run on the same file stops with an error
instead of writing into the same partial file. The lock is removed when the run ends, also on
an error or SIGTERM (as at a SLURM time limit). A run killed outright (SIGKILL, node failure)
leaves its lock behind; the launchers in ``hpc/lumi/`` remove locks whose job is no longer in
the queue, and by hand it is safe to remove one once its job has ended.

Example::

    python scripts/data_processing/classify_hplt.py \\
        --input /appl/local/openeurollm/training/catalogue/hplt/4.0/clean/fin_Latn/10_1.jsonl.zst \\
        --output_root results/classifier_predictions/mmbert-1/hplt-4.0-clean \\
        --topic_model results/models/topic/hplt-finetune/mmbert-topic-1 \\
        --format_model results/models/format/hplt-finetune/mmbert-format-1
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import queue
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import FrameType
from typing import Any, TypeVar

import numpy as np
import torch
import transformers
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from transformers.utils.versions import require_version

# The models were saved by transformers 5: their tokenizer config names a transformers 5 class,
# and their config.json keeps mmBERT's RoPE settings under the transformers 5 key
# "rope_parameters", which transformers 4 ignores (it would run with the wrong local RoPE base).
require_version("transformers>=5", "the classifiers were saved by transformers 5 and need it to load correctly")

# HPLT file reading from the sampler (a sibling module), and training's input rendering, so
# documents reach the models exactly as in fine-tuning.
from sample_hplt_by_quality import open_zst, stdlib_zstd, zstandard  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model_training"))
from finetune import render_input, resolve_attn_implementation, template_fields  # noqa: E402

LOGGER = logging.getLogger(__name__)

# HPLT 4.0 file names, e.g. 10_1.jsonl.zst.
INPUT_NAME = re.compile(r"^(?P<shard>\d+)_(?P<part>\d+)\.jsonl\.zst$")
INPUT_SUFFIX = ".jsonl.zst"
# The HPLT 4.0 fields read, by the name the input template gives them.
HPLT_FIELDS = {"id": "id", "url": "u", "text": "text"}
# The models must take the URL and the text (not the -nourl models).
REQUIRED_TEMPLATE_FIELDS = frozenset({"url", "text"})
CLASSIFIER_NAMES = ("topic", "format")
# Fields of each output line, in order.
OUTPUT_FIELDS = ("id", "url", "text", "topic", "topic_probs", "format", "format_probs")
# Tokenizer files that must be identical in both model directories to tokenize once for both.
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")

Item = TypeVar("Item")


@dataclass(frozen=True)
class Classifier:
    """A loaded classifier.

    Attributes:
        name: ``topic`` or ``format``; also the output field names.
        path: The model directory.
        model: The model, in eval mode on the device.
        labels: Label names in the model's output order.
    """

    name: str
    path: Path
    model: torch.nn.Module
    labels: list[str]


@dataclass(frozen=True)
class Document:
    """One HPLT document: the fields kept in the output, and its model input.

    Attributes:
        doc_id: HPLT document ID (``id``).
        url: The URL as in the input (``u``).
        text: The text as in the input (``text``).
        model_input: The classifier input built from them (``finetune.render_input``).
    """

    doc_id: str
    url: str
    text: str
    model_input: str


@dataclass(frozen=True)
class Chunk:
    """Consecutive input documents, tokenized.

    Attributes:
        documents: The documents, in input order.
        input_ids: Token IDs of each document's model input, truncated to the maximum length.
    """

    documents: list[Document]
    input_ids: list[list[int]]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command-line arguments.

    Args:
        argv: Arguments to parse instead of ``sys.argv``.

    Returns:
        The parsed arguments.

    Raises:
        SystemExit: If an argument is invalid.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="HPLT 4.0 file, <language>/<shard>_<part>.jsonl.zst")
    parser.add_argument(
        "--output_root", type=Path, required=True, help="Output directory; results go to <output_root>/<language>/"
    )
    parser.add_argument("--topic_model", type=Path, required=True, help="Topic classifier directory (finetune.py output)")
    parser.add_argument("--format_model", type=Path, required=True, help="Format classifier directory (finetune.py output)")
    parser.add_argument("--max_length", type=int, default=8192, help="Tokens per document (training: 8192)")
    parser.add_argument(
        "--max_batch_tokens", type=int, default=65536, help="Padded tokens per batch (batch size x longest document)"
    )
    parser.add_argument("--max_batch_size", type=int, default=512, help="Documents per batch at most")
    parser.add_argument(
        "--chunk_size", type=int, default=4096, help="Documents tokenized at a time and sorted by length into batches"
    )
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True, help="bf16 autocast (default on)")
    parser.add_argument(
        "--attn_implementation",
        default="auto",
        choices=["auto", "flash_attention_2", "sdpa", "eager"],
        help="auto: flash_attention_2 if the flash-attn package is installed, else sdpa (as in finetune.py)",
    )
    parser.add_argument("--decimals", type=int, default=4, help="Decimals the probabilities are rounded to")
    parser.add_argument(
        "--compress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compress finished files with zstandard (.jsonl.zst; default) or leave them as .jsonl",
    )
    parser.add_argument("--compression_level", type=int, default=3, help="zstandard compression level")
    parser.add_argument(
        "--max_documents",
        type=int,
        default=None,
        help="Stop once the partial file holds this many documents (for tests); a later run continues it",
    )
    parser.add_argument("--log_every_seconds", type=float, default=300.0, help="Seconds between progress messages")
    args = parser.parse_args(argv)

    positive = ("max_length", "max_batch_tokens", "max_batch_size", "chunk_size", "compression_level", "log_every_seconds")
    for name in positive:
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if args.max_batch_tokens < args.max_length:
        parser.error("--max_batch_tokens must be at least --max_length, or a full-length document fits no batch")
    if args.decimals < 1:
        parser.error("--decimals must be at least 1")
    if args.max_documents is not None and args.max_documents <= 0:
        parser.error("--max_documents must be positive")
    if INPUT_NAME.match(args.input.name) is None:
        parser.error(f"--input {args.input} is not named <shard>_<part>{INPUT_SUFFIX}")
    return args


def compression_backend() -> str:
    """Return the name of the zstandard compressor that ``compress_file`` will use.

    Raises:
        RuntimeError: If none is available.
    """
    if stdlib_zstd is not None:
        return "compression.zstd (standard library)"
    if zstandard is not None:
        return "zstandard package"
    if shutil.which("zstd") is not None:
        return "zstd command-line tool"
    raise RuntimeError(
        "No zstandard compressor (Python >= 3.14, the zstandard package or the zstd tool); "
        "run with --no-compress to write uncompressed .jsonl files"
    )


def compress_file(source: Path, target: Path, level: int) -> None:
    """Compress a file with zstandard.

    Args:
        source: The file to compress.
        target: The compressed file to write (replaced if it exists).
        level: zstandard compression level.

    Raises:
        RuntimeError: If no compressor is available.
        subprocess.CalledProcessError: If the zstd command-line tool fails.
    """
    if stdlib_zstd is not None:
        with source.open("rb") as reader, stdlib_zstd.open(target, "wb", level=level) as writer:
            shutil.copyfileobj(reader, writer, length=1 << 20)
    elif zstandard is not None:
        compressor = zstandard.ZstdCompressor(level=level, threads=-1)
        with source.open("rb") as reader, target.open("wb") as writer:
            compressor.copy_stream(reader, writer)
    elif shutil.which("zstd") is not None:
        command = ["zstd", "--quiet", "--force", f"-{level}", "-T0", "-o", str(target), "--", str(source)]
        subprocess.run(command, check=True)
    else:
        raise RuntimeError("No zstandard compressor; run with --no-compress")


def count_lines(path: Path, compressed: bool) -> int:
    """Count the lines of a file, decompressing it first if ``compressed``.

    Raises:
        ValueError: If the file does not end with a newline.
    """
    lines = 0
    last = b"\n"
    with open_zst(path) if compressed else path.open("rb") as handle:
        while block := handle.read(1 << 22):
            lines += block.count(b"\n")
            last = block[-1:]
    if last != b"\n":
        raise ValueError(f"{path} does not end with a newline")
    return lines


def load_classifier(name: str, path: Path, device: torch.device, attn_implementation: str) -> tuple[Classifier, Any]:
    """Load a classifier saved by ``finetune.py``, in fp32 and eval mode on the device.

    Args:
        name: ``topic`` or ``format``.
        path: The model directory.
        device: The device to put it on.
        attn_implementation: Attention implementation to load it with.

    Returns:
        The classifier and its config.

    Raises:
        ValueError: If the config lacks the input template or URL decoding saved by
            ``finetune.py``, or its labels are not numbered 0..N-1.
    """
    config = AutoConfig.from_pretrained(path)
    for setting in ("input_template", "decode_url"):
        if getattr(config, setting, None) is None:
            raise ValueError(f"{path}/config.json has no {setting!r}; was the model trained with finetune.py?")
    if sorted(config.id2label) != list(range(config.num_labels)):
        raise ValueError(f"{path}/config.json: id2label keys {sorted(config.id2label)} are not 0..{config.num_labels - 1}")
    # Weights stay in fp32 as in training; --bf16 runs the forward pass in mixed precision.
    model = AutoModelForSequenceClassification.from_pretrained(
        path, config=config, dtype=torch.float32, attn_implementation=attn_implementation
    )
    model.to(device).eval()
    labels = [config.id2label[index] for index in range(config.num_labels)]
    LOGGER.info("%s classifier: %s, %d labels", name, path, len(labels))
    return Classifier(name=name, path=path, model=model, labels=labels), config


def shared_input(configs: dict[str, Any], model_paths: dict[str, Path], max_length: int) -> tuple[str, bool]:
    """Check that the classifiers take the same input, with the URL, and return it.

    Args:
        configs: Model config per classifier name.
        model_paths: Model directory per classifier name.
        max_length: Tokens per document.

    Returns:
        The input template and whether URLs are percent-decoded.

    Raises:
        ValueError: If the models' inputs or tokenizers differ, the input lacks the URL or the
            text, or a model's context is shorter than ``max_length``.
    """
    settings = {name: (config.input_template, bool(config.decode_url)) for name, config in configs.items()}
    if len(set(settings.values())) != 1:
        raise ValueError(f"The classifiers take different inputs (template, decode_url): {settings}")
    template, decode_url = next(iter(settings.values()))
    fields = template_fields(template)
    if fields != REQUIRED_TEMPLATE_FIELDS:
        raise ValueError(
            f"The classifiers' input template {template!r} uses {sorted(fields)}, not {sorted(REQUIRED_TEMPLATE_FIELDS)}; "
            "this script is for the models that take the URL and the text"
        )
    for name, config in configs.items():
        if config.max_position_embeddings < max_length:
            raise ValueError(f"--max_length {max_length} exceeds the {name} model's {config.max_position_embeddings} positions")
    first, *others = model_paths.values()
    for other in others:
        for filename in TOKENIZER_FILES:
            if (first / filename).read_bytes() != (other / filename).read_bytes():
                raise ValueError(f"{first / filename} and {other / filename} differ; the models need the same tokenizer")
    return template, decode_url


def ensure_metadata(path: Path, metadata: dict[str, Any]) -> None:
    """Write the output root's metadata, or check that the existing one matches.

    Args:
        path: The metadata file.
        metadata: The models and settings of this run.

    Raises:
        ValueError: If the output root was made with other models or settings.
    """
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        # Written under a unique name and renamed, so tasks starting together never see half a file.
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    existing = json.loads(path.read_text(encoding="utf-8"))
    if existing != metadata:
        differing = sorted(key for key in existing.keys() | metadata.keys() if existing.get(key) != metadata.get(key))
        raise ValueError(
            f"{path.parent} was labelled with other models or settings (differing: {differing}); "
            f"use another --output_root.\nExisting: {existing}\nThis run: {metadata}"
        )


def read_documents(path: Path, template: str, decode_url: bool) -> Iterator[Document]:
    """Yield every document of an HPLT file with its model input, in file order.

    Args:
        path: The ``.jsonl.zst`` file.
        template: Input template for ``finetune.render_input``.
        decode_url: Percent-decode the URL in the model input, as in training.

    Yields:
        The documents.

    Raises:
        ValueError: On a blank or malformed line, or a document without an ID, URL or text.
    """
    with open_zst(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            where = f"{path}:{line_number}"
            if not line.strip():
                raise ValueError(f"{where}: blank line")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{where}: invalid JSON: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"{where}: expected a JSON object, got {type(record).__name__}")
            row = {}
            for name, field in HPLT_FIELDS.items():
                value = record.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{where}: field {field!r} is {value!r}; expected a non-empty string")
                row[name] = value
            yield Document(
                doc_id=row["id"], url=row["url"], text=row["text"], model_input=render_input(row, template, decode_url)
            )


def drop_cut_off_line(path: Path) -> None:
    """Truncate a file after its last newline, dropping a line cut off by a killed job.

    Raises:
        OSError: If the file cannot be read or truncated.
    """
    size = path.stat().st_size
    if size == 0:
        return
    with path.open("r+b") as handle:
        handle.seek(size - 1)
        if handle.read(1) == b"\n":
            return
        keep = 0
        end = size
        while end > 0:
            start = max(0, end - (1 << 16))
            handle.seek(start)
            newline = handle.read(end - start).rfind(b"\n")
            if newline >= 0:
                keep = start + newline + 1
                break
            end = start
        LOGGER.warning("%s ends in a cut-off line (%d bytes); dropping it to label its document again", path, size - keep)
        handle.truncate(keep)


def resume(partial: Path, documents: Iterator[Document], classifiers: Sequence[Classifier]) -> int:
    """Check an existing partial output against the input and skip the documents it holds.

    Reads the partial file and the input side by side, so ``documents`` is left at the first
    document still to label.

    Args:
        partial: The partial output file; missing if no earlier run started this input.
        documents: The input's documents, from the start.
        classifiers: The classifiers, to check each line's fields.

    Returns:
        The number of documents already labelled.

    Raises:
        ValueError: If a line is malformed, its ID or URL differs from the input document at the
            same position, or the partial file has more lines than the input has documents.
    """
    if not partial.exists():
        return 0
    drop_cut_off_line(partial)
    done = 0
    with partial.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            where = f"{partial}:{line_number}"
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{where}: invalid JSON: {error}") from error
            if not isinstance(record, dict) or tuple(record) != OUTPUT_FIELDS:
                fields = list(record) if isinstance(record, dict) else type(record).__name__
                raise ValueError(f"{where}: fields {fields}, expected {list(OUTPUT_FIELDS)}")
            for classifier in classifiers:
                label = record.get(classifier.name)
                probs = record.get(f"{classifier.name}_probs")
                if label not in classifier.labels or not isinstance(probs, list) or len(probs) != len(classifier.labels):
                    raise ValueError(f"{where}: malformed {classifier.name} result; was it written by another model?")
            document = next(documents, None)
            if document is None:
                raise ValueError(f"{partial} has more lines than the input has documents")
            if (record["id"], record["url"]) != (document.doc_id, document.url):
                raise ValueError(
                    f"{where}: ID and URL {record['id']!r}, {record['url']!r}, but input document {line_number} "
                    f"has {document.doc_id!r}, {document.url!r}"
                )
            done += 1
    LOGGER.info("Resuming after %d documents already in %s", done, partial)
    return done


def tokenized_chunks(
    documents: Iterator[Document], tokenizer: Any, chunk_size: int, max_length: int, limit: int | None
) -> Iterator[Chunk]:
    """Tokenize documents in chunks of consecutive documents.

    Args:
        documents: The documents.
        tokenizer: The models' tokenizer.
        chunk_size: Documents per chunk.
        max_length: Tokens per document; longer inputs are truncated, as in training.
        limit: Stop after this many documents, or None for all.

    Yields:
        Chunks, in input order.
    """
    remaining = limit
    while remaining is None or remaining > 0:
        size = chunk_size if remaining is None else min(chunk_size, remaining)
        batch = list(itertools.islice(documents, size))
        if not batch:
            return
        if remaining is not None:
            remaining -= len(batch)
        encoded = tokenizer(
            [document.model_input for document in batch],
            truncation=True,
            max_length=max_length,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        yield Chunk(documents=batch, input_ids=encoded["input_ids"])


def prefetch(items: Iterator[Item], depth: int) -> Iterator[Item]:
    """Run an iterator in a background thread, keeping up to ``depth`` items ready.

    Reading, parsing and tokenizing the next chunk then overlaps with the GPU work on the
    current one. An exception in the thread is raised here.

    Args:
        items: The iterator to run.
        depth: Items to keep ready.

    Yields:
        The iterator's items, in order.
    """
    slots: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=depth)
    stop = threading.Event()

    def put(entry: tuple[str, Any]) -> bool:
        while not stop.is_set():
            try:
                slots.put(entry, timeout=1.0)
                return True
            except queue.Full:
                continue
        return False

    def work() -> None:
        try:
            for item in items:
                if not put(("item", item)):
                    return
        except BaseException as error:  # noqa: BLE001 - re-raised in the consuming thread
            put(("error", error))
            return
        put(("done", None))

    thread = threading.Thread(target=work, name="reader", daemon=True)
    thread.start()
    try:
        while True:
            kind, value = slots.get()
            if kind == "error":
                raise value
            if kind == "done":
                thread.join()
                return
            yield value
    finally:
        stop.set()


def batches(lengths: Sequence[int], max_batch_tokens: int, max_batch_size: int) -> list[list[int]]:
    """Group positions (longest first) into batches whose padded size fits the token budget.

    Args:
        lengths: Token count per document.
        max_batch_tokens: Maximum of batch size times the longest document in the batch.
        max_batch_size: Maximum documents per batch.

    Returns:
        Lists of positions into ``lengths``.
    """
    order = sorted(range(len(lengths)), key=lambda position: -lengths[position])
    groups: list[list[int]] = []
    current: list[int] = []
    for position in order:
        longest = lengths[current[0]] if current else lengths[position]
        if current and (longest * (len(current) + 1) > max_batch_tokens or len(current) == max_batch_size):
            groups.append(current)
            current = []
        current.append(position)
    if current:
        groups.append(current)
    return groups


@torch.inference_mode()
def predict(
    classifiers: Sequence[Classifier], chunk: Chunk, pad_token_id: int, device: torch.device, args: argparse.Namespace
) -> dict[str, np.ndarray]:
    """Compute every classifier's label probabilities for a chunk.

    Args:
        classifiers: The classifiers.
        chunk: The tokenized documents.
        pad_token_id: The tokenizer's padding token.
        device: The models' device.
        args: Parsed arguments (batching and bf16).

    Returns:
        Probabilities per classifier name, one float32 row per document in chunk order.

    Raises:
        ValueError: If a probability is not finite.
    """
    lengths = [len(ids) for ids in chunk.input_ids]
    probs = {c.name: np.full((len(lengths), len(c.labels)), np.nan, dtype=np.float32) for c in classifiers}
    for group in batches(lengths, args.max_batch_tokens, args.max_batch_size):
        longest = lengths[group[0]]
        # Right padding, as in training (the tokenizer's padding side).
        input_ids = np.full((len(group), longest), pad_token_id, dtype=np.int64)
        attention_mask = np.zeros((len(group), longest), dtype=np.int64)
        for row, position in enumerate(group):
            input_ids[row, : lengths[position]] = chunk.input_ids[position]
            attention_mask[row, : lengths[position]] = 1
        batch = {
            "input_ids": torch.from_numpy(input_ids).to(device),
            "attention_mask": torch.from_numpy(attention_mask).to(device),
        }
        for classifier in classifiers:
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.bf16):
                logits = classifier.model(**batch).logits
            probs[classifier.name][group] = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    for name, values in probs.items():
        bad = ~np.isfinite(values).all(axis=1)
        if bad.any():
            examples = [chunk.documents[i].doc_id for i in np.flatnonzero(bad)[:5]]
            raise ValueError(f"{name}: non-finite probabilities for {int(bad.sum())} documents, e.g. {examples}")
    return probs


def output_lines(chunk: Chunk, probs: dict[str, np.ndarray], classifiers: Sequence[Classifier], decimals: int) -> str:
    """Format a chunk's results as JSON lines, in input order.

    Args:
        chunk: The documents.
        probs: Probabilities per classifier name.
        classifiers: The classifiers.
        decimals: Decimals the probabilities are rounded to.

    Returns:
        One line per document, each ending in a newline.
    """
    # The top label comes from the unrounded probabilities. Rounding in float64 keeps the JSON
    # short (0.0012, not float32's 0.0012000000569...).
    top = {c.name: probs[c.name].argmax(axis=1) for c in classifiers}
    rounded = {c.name: np.round(probs[c.name].astype(np.float64), decimals).tolist() for c in classifiers}
    lines = []
    for row, document in enumerate(chunk.documents):
        record: dict[str, Any] = {"id": document.doc_id, "url": document.url, "text": document.text}
        for classifier in classifiers:
            record[classifier.name] = classifier.labels[top[classifier.name][row]]
            record[f"{classifier.name}_probs"] = rounded[classifier.name][row]
        lines.append(json.dumps(record, ensure_ascii=False) + "\n")
    return "".join(lines)


def finish(partial: Path, final: Path, documents: int, args: argparse.Namespace) -> None:
    """Move a complete partial file to its final name, compressing it if asked.

    The final file is checked to hold one line per document before the partial file is removed.

    Args:
        partial: The complete partial output.
        final: The final output path.
        documents: The number of documents in the input.
        args: Parsed arguments (compression).

    Raises:
        ValueError: If the output does not hold one line per document.
    """
    if args.compress:
        temporary = final.with_name(f"{final.name}.tmp")
        LOGGER.info("Compressing %s with %s", partial, compression_backend())
        compress_file(partial, temporary, args.compression_level)
        lines = count_lines(temporary, compressed=True)
        if lines != documents:
            raise ValueError(f"{temporary} holds {lines} lines for {documents} documents; {partial} is kept")
        temporary.replace(final)
        partial.unlink()
    else:
        lines = count_lines(partial, compressed=False)
        if lines != documents:
            raise ValueError(f"{partial} holds {lines} lines for {documents} documents")
        partial.replace(final)
    LOGGER.info("Wrote %s (%d documents)", final, documents)


@contextmanager
def file_lock(path: Path) -> Iterator[None]:
    """Hold a lock file while labelling one input, so no two runs write the same output.

    The lock is created atomically (``O_EXCL``, which Lustre supports) and holds
    ``key=value`` lines (``slurm_job_id``, ``host``, ``pid``, ``started``) that the
    launchers read to remove locks of jobs that are no longer in the queue.

    Args:
        path: The lock file.

    Yields:
        Nothing; the lock is held until the block ends.

    Raises:
        RuntimeError: If the lock is already held.
    """
    owner = {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "none"),
        "host": socket.gethostname(),
        "pid": str(os.getpid()),
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o664)
    except FileExistsError:
        holder = path.read_text(encoding="utf-8").strip().replace("\n", ", ")
        raise RuntimeError(
            f"{path} exists ({holder}): another run is labelling this file. If that job has ended, "
            "remove the lock and run again"
        ) from None
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write("".join(f"{key}={value}\n" for key, value in owner.items()))
        yield
    finally:
        path.unlink(missing_ok=True)


def exit_on_sigterm(signum: int, frame: FrameType | None) -> None:
    """Turn SIGTERM (sent by SLURM at the time limit) into SystemExit, so locks are released.

    Raises:
        SystemExit: Always, with the shell's exit code for the signal.
    """
    LOGGER.warning("Received signal %d; stopping. The partial file is kept and a later run continues it", signum)
    raise SystemExit(128 + signum)


def main(argv: Sequence[str] | None = None) -> None:
    """Label one HPLT file with the topic and format classifiers.

    Args:
        argv: Arguments to parse instead of ``sys.argv``.

    Raises:
        RuntimeError: If there is no GPU or no compressor, or another run holds the file's lock.
        ValueError: On malformed input, mismatched models, or an output root made with other settings.
    """
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    signal.signal(signal.SIGTERM, exit_on_sigterm)

    language = args.input.parent.name
    stem = args.input.name.removesuffix(INPUT_SUFFIX)
    output_dir = args.output_root / language
    final = output_dir / f"{stem}{'.jsonl.zst' if args.compress else '.jsonl'}"
    partial = output_dir / f"{stem}.jsonl.partial"
    if final.exists() and not partial.exists():
        LOGGER.info("%s is already labelled: %s", args.input, final)
        return
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    output_dir.mkdir(parents=True, exist_ok=True)
    with file_lock(output_dir / f"{stem}.lock"):
        # Checked again under the lock: another run may have finished the file meanwhile.
        if final.exists():
            if partial.exists():
                raise ValueError(f"Both {final} and {partial} exist; check them and remove the stale one")
            LOGGER.info("%s is already labelled: %s", args.input, final)
            return
        label_file(args, final, partial)


def label_file(args: argparse.Namespace, final: Path, partial: Path) -> None:
    """Label one HPLT file, resuming its partial output, and write the final output when complete.

    Args:
        args: Parsed arguments.
        final: The final output path.
        partial: The partial output path.

    Raises:
        RuntimeError: If there is no GPU or no compressor.
        ValueError: On malformed input, mismatched models, or an output root made with other settings.
    """
    if args.compress:
        LOGGER.info("Compressing finished files with %s", compression_backend())
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU available")

    device = torch.device("cuda")
    attn_implementation = resolve_attn_implementation(args.attn_implementation)
    LOGGER.info(
        "torch %s, transformers %s, device %s, attention %s",
        torch.__version__, transformers.__version__, torch.cuda.get_device_name(device), attn_implementation,
    )
    model_paths = {"topic": args.topic_model, "format": args.format_model}
    classifiers: list[Classifier] = []
    configs: dict[str, Any] = {}
    for name in CLASSIFIER_NAMES:
        classifier, configs[name] = load_classifier(name, model_paths[name], device, attn_implementation)
        classifiers.append(classifier)
    template, decode_url = shared_input(configs, model_paths, args.max_length)
    tokenizer = AutoTokenizer.from_pretrained(args.topic_model)
    if tokenizer.pad_token_id is None or tokenizer.padding_side != "right":
        raise ValueError(f"Expected a right-padding tokenizer with a pad token, as in training: {tokenizer}")

    ensure_metadata(
        args.output_root / "metadata.json",
        {
            "hplt_fields": HPLT_FIELDS,
            "output_fields": list(OUTPUT_FIELDS),
            "input_template": template,
            "decode_url": decode_url,
            "max_length": args.max_length,
            "bf16": args.bf16,
            "decimals": args.decimals,
            "compress": args.compress,
            "classifiers": {c.name: {"model": str(c.path.resolve()), "labels": c.labels} for c in classifiers},
        },
    )

    LOGGER.info("Labelling %s into %s", args.input, final)
    documents = read_documents(args.input, template, decode_url)
    done = resume(partial, documents, classifiers)
    limit = None if args.max_documents is None else max(0, args.max_documents - done)

    started = time.monotonic()
    last_log = started
    new_documents = 0
    new_tokens = 0
    with partial.open("a", encoding="utf-8") as output:
        chunks = tokenized_chunks(documents, tokenizer, args.chunk_size, args.max_length, limit)
        for chunk in prefetch(chunks, depth=2):
            probs = predict(classifiers, chunk, tokenizer.pad_token_id, device, args)
            # One write and flush per chunk: a killed job loses at most the chunk being written.
            output.write(output_lines(chunk, probs, classifiers, args.decimals))
            output.flush()
            new_documents += len(chunk.documents)
            new_tokens += sum(len(ids) for ids in chunk.input_ids)
            now = time.monotonic()
            if now - last_log >= args.log_every_seconds:
                elapsed = now - started
                LOGGER.info(
                    "%d documents labelled (%d this run): %.1f documents and %.0f tokens per second",
                    done + new_documents, new_documents, new_documents / elapsed, new_tokens / elapsed,
                )
                last_log = now
    done += new_documents
    elapsed = time.monotonic() - started
    if new_documents:
        LOGGER.info(
            "This run: %d documents in %.0f s, %.1f documents and %.0f tokens per second (%.0f tokens per document)",
            new_documents, elapsed, new_documents / elapsed, new_tokens / elapsed, new_tokens / new_documents,
        )

    # The reader thread has finished, so the input can be read here: is anything left?
    if next(documents, None) is not None:
        LOGGER.info("Stopped at --max_documents %d; %s is kept and a later run continues it", args.max_documents, partial)
        return
    if done == 0:
        raise ValueError(f"{args.input} has no documents")
    finish(partial, final, done, args)


if __name__ == "__main__":
    main()
