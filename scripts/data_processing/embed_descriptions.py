"""Embed the English page descriptions of ``describe_documents.py`` with Qwen3-Embedding.

Each chosen field of the descriptions (by default genre, topic, problems and description) is
embedded separately, so the clustering can use the field that suits the question: genre for
format Other, problems for Bad example, and so on. Qwen3-Embedding takes an instruction before
the text; each field has its own (``FIELD_INSTRUCTIONS``).

The model's embeddings are cut to their first ``--dimensions`` values and normalised to unit
length again. Qwen3-Embedding is trained so that such a prefix is itself an embedding
(Matryoshka representation learning), and doing the cut here keeps it independent of vLLM's
support for it.

Every description file must cover exactly the documents of its input file; a partial or extra
file stops the script, so an unfinished description job can't go unnoticed.

Outputs in ``--output-dir``:

- ``index.tsv``: language and document ID of each embedding row, in row order;
- ``<field>.npy``: float16 array of shape (documents, dimensions) per field;
- ``metadata.json``: the model, dimensions, instructions and document counts.

Example:
    python scripts/data_processing/embed_descriptions.py \\
        --descriptions-dir results/other_bad_descriptions/Qwen3.8-27B \\
        --input-dir results/statistics/other_bad_examples \\
        --languages thai swahili \\
        --output-dir results/other_bad_descriptions/Qwen3.8-27B/embeddings/Qwen3-Embedding-8B
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)

FIELD_INSTRUCTIONS = {
    "genre": "Given the genre of a web page, find web pages of the same genre",
    "topic": "Given the topic of a web page, find web pages on the same topic",
    "problems": "Given the quality problems of a web page, find web pages with the same problems",
    "description": "Given a description of a web page, find web pages of the same kind",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command-line arguments.

    Args:
        argv: Arguments, or ``None`` for the command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--descriptions-dir", type=Path, required=True, help="Has <language>.jsonl descriptions")
    parser.add_argument("--input-dir", type=Path, required=True, help="Has the described <language>.jsonl inputs")
    parser.add_argument("--languages", nargs="+", required=True, help="Languages to embed, e.g. thai swahili")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--fields", nargs="+", default=list(FIELD_INSTRUCTIONS), choices=list(FIELD_INSTRUCTIONS), help="Fields to embed"
    )
    parser.add_argument("--model-name", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--dimensions", type=int, default=1024, help="Leading dimensions kept (default 1024)")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=4096, help="Longest instruction plus text, in tokens")
    parser.add_argument("--batch-size", type=int, default=4096, help="Texts per vLLM call")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = parser.parse_args(argv)
    if len(set(args.languages)) != len(args.languages) or len(set(args.fields)) != len(args.fields):
        parser.error("--languages and --fields must not repeat")
    if args.dimensions < 1 or args.batch_size < 1:
        parser.error("--dimensions and --batch-size must be >= 1")
    return args


def read_ids(path: Path) -> Iterator[str]:
    """Yield the ``id`` of every line of a JSONL file.

    Args:
        path: JSONL file.

    Yields:
        The IDs in file order.

    Raises:
        ValueError: On an empty line or a record without a string ``id``.
    """
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Empty line in {path}:{line_number}")
            record_id = json.loads(line).get("id")
            if not isinstance(record_id, str) or not record_id:
                raise ValueError(f"No id in {path}:{line_number}")
            yield record_id


def load_descriptions(
    descriptions_dir: Path, input_dir: Path, languages: Sequence[str], fields: Sequence[str]
) -> tuple[list[tuple[str, str]], dict[str, list[str]]]:
    """Read the chosen fields of every description, checking each file is complete.

    Args:
        descriptions_dir: Directory of description files.
        input_dir: Directory of the input files that were described.
        languages: Languages to read.
        fields: Description fields to return.

    Returns:
        The (language, id) of each row, and the text of each field in the same row order.

    Raises:
        ValueError: If a description file doesn't cover exactly its input's documents, an ID
            repeats, or a field is missing or empty.
    """
    index: list[tuple[str, str]] = []
    texts: dict[str, list[str]] = {field: [] for field in fields}
    for language in languages:
        description_path = descriptions_dir / f"{language}.jsonl"
        expected = set(read_ids(input_dir / f"{language}.jsonl"))
        seen: set[str] = set()
        with description_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                record = json.loads(line)
                record_id = record["id"]
                if record_id in seen:
                    raise ValueError(f"Repeated id {record_id!r} in {description_path}:{line_number}")
                if record.get("language") != language:
                    raise ValueError(f"{description_path}:{line_number} has language {record.get('language')!r}")
                seen.add(record_id)
                index.append((language, record_id))
                for field in fields:
                    value = record.get(field)
                    if not isinstance(value, str) or not value.strip():
                        raise ValueError(f"Empty or missing {field!r} in {description_path}:{line_number}")
                    texts[field].append(value.strip())
        if seen != expected:
            raise ValueError(
                f"{description_path} doesn't match {input_dir / f'{language}.jsonl'}: "
                f"{len(expected - seen)} input documents not described, {len(seen - expected)} unknown IDs"
            )
        LOGGER.info("%s: %d descriptions", language, len(seen))
    return index, texts


def embed_texts(llm: object, texts: Sequence[str], instruction: str, dimensions: int, batch_size: int) -> np.ndarray:
    """Embed texts with an instruction, keeping the leading dimensions.

    Args:
        llm: vLLM engine started with ``runner="pooling"``.
        texts: Texts to embed.
        instruction: Qwen3-Embedding task instruction.
        dimensions: Leading dimensions kept.
        batch_size: Texts per vLLM call.

    Returns:
        A float16 array of unit-length rows, one per text.

    Raises:
        ValueError: If the model's embeddings are shorter than ``dimensions`` or a row is zero.
    """
    rows = np.empty((len(texts), dimensions), dtype=np.float16)
    for start in range(0, len(texts), batch_size):
        batch = [f"Instruct: {instruction}\nQuery:{text}" for text in texts[start : start + batch_size]]
        outputs = llm.embed(batch, use_tqdm=False)  # type: ignore[attr-defined]
        if len(outputs) != len(batch):
            raise RuntimeError(f"vLLM returned {len(outputs)} embeddings for {len(batch)} texts")
        full = np.asarray([output.outputs.embedding for output in outputs], dtype=np.float32)
        if full.ndim != 2 or full.shape[1] < dimensions:
            raise ValueError(f"Embeddings have shape {full.shape}; can't keep {dimensions} dimensions")
        kept = full[:, :dimensions]
        norms = np.linalg.norm(kept, axis=1, keepdims=True)
        if not np.all(np.isfinite(norms)) or np.any(norms == 0):
            raise ValueError("An embedding is zero or not finite")
        rows[start : start + len(batch)] = kept / norms
        LOGGER.info("Embedded %d of %d", start + len(batch), len(texts))
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    """Embed the description fields of the chosen languages.

    Args:
        argv: Arguments, or ``None`` for the command line.

    Returns:
        The process exit code.
    """
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s [%(levelname)s] %(message)s")
    outputs = [args.output_dir / name for name in ("index.tsv", "metadata.json")]
    outputs += [args.output_dir / f"{field}.npy" for field in args.fields]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Outputs exist (use --overwrite): {[str(path) for path in existing]}")

    index, texts = load_descriptions(args.descriptions_dir, args.input_dir, args.languages, args.fields)
    LOGGER.info("%d descriptions in %d languages", len(index), len(args.languages))

    from vllm import LLM  # type: ignore

    llm = LLM(
        model=args.model_name,
        runner="pooling",
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "index.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["row", "language", "id"])
        writer.writerows((row, language, record_id) for row, (language, record_id) in enumerate(index))
    for field in args.fields:
        start = time.time()
        array = embed_texts(llm, texts[field], FIELD_INSTRUCTIONS[field], args.dimensions, args.batch_size)
        np.save(args.output_dir / f"{field}.npy", array)
        LOGGER.info("%s: %s in %.0f s", field, array.shape, time.time() - start)
    metadata = {
        "model": args.model_name,
        "dimensions": args.dimensions,
        "instructions": {field: FIELD_INSTRUCTIONS[field] for field in args.fields},
        "documents": {language: sum(1 for row in index if row[0] == language) for language in args.languages},
        "descriptions_dir": str(args.descriptions_dir),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
