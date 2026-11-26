"""Select the documents of one classifier output file for a training mixture.

Reads one file of ``classify_hplt.py``'s output (``<language>/<bucket>_<part>.jsonl.zst``)
and keeps each document with the selection probability of its topic x format pair, from
the mixture of ``compute_mixture.py``. The decision is a seeded hash of the document ID, so
the selection is reproducible and does not depend on the order or the split of the files.

Writes, all through temporary files that are renamed when complete:

* ``--output``: plain JSONL with ``id`` and ``text`` of the selected documents, in input
  order, for Megatron's ``preprocess_data.py`` (which reads the ``text`` key);
* ``--labels``: id, topic and format of each selected document, in the same
  order, so token counts of the tokenized data can be assigned to pairs;
* ``--summary``: counts of input and selected documents and characters, per pair and in
  total, with the mixture file's SHA-256.

Stops with an error on a malformed line, a missing field, a label the mixture does not
know, or a selection probability above 1 (repetition is not supported).

Example::

    python scripts/data_processing/select_mixture_documents.py \\
        --mixture results/mixtures/hplt-4.0-clean/deu_Latn/both.json \\
        --input results/classifier_predictions/mmbert-1/hplt-4.0-clean/deu_Latn/9_17.jsonl.zst \\
        --output /scratch/.../work/9_17.jsonl \\
        --labels data/megatron/.../deu_Latn/9_17.labels.tsv \\
        --summary data/megatron/.../deu_Latn/9_17.selection.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# The zstandard reader of the HPLT sampler, so the input is read like the other outputs.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sample_hplt_by_quality import open_zst, zstd_backend  # noqa: E402

LOGGER = logging.getLogger(__name__)

REQUIRED_FIELDS = ("id", "text", "topic", "format")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command-line arguments.

    Args:
        argv: Arguments to parse, ``sys.argv[1:]`` if None.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mixture", type=Path, required=True, help="JSON output of compute_mixture.py")
    parser.add_argument("--input", type=Path, required=True, help="One classify_hplt.py output file (.jsonl.zst)")
    parser.add_argument("--output", type=Path, required=True, help="Plain JSONL file of the selected documents")
    parser.add_argument("--labels", type=Path, required=True,
                        help="TSV file of id, topic and format of the selected documents")
    parser.add_argument("--summary", type=Path, required=True, help="JSON file of the selection counts")
    parser.add_argument("--seed", type=int, default=42, help="Seed of the selection hash")
    parser.add_argument("--log_every_seconds", type=float, default=300, help="Progress log interval")
    return parser.parse_args(argv)


def load_probabilities(path: Path) -> dict[tuple[str, str], float]:
    """Load the selection probability of every topic x format pair.

    Args:
        path: JSON output of compute_mixture.py.

    Returns:
        Probability per (topic, format).

    Raises:
        ValueError: If a probability is outside [0, 1].
    """
    mixture = json.loads(path.read_text(encoding="utf-8"))
    probabilities = {}
    for pair in mixture["pairs"]:
        p = pair["selection_probability"]
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"{path}: {pair['topic']} x {pair['format']} has selection probability {p}; "
                             "only selection without repetition (at most 1) is supported")
        probabilities[(pair["topic"], pair["format"])] = p
    return probabilities


def uniform(doc_id: str, key: bytes) -> float:
    """Map a document ID to a pseudo-random number in [0, 1), fixed by the key.

    Args:
        doc_id: Document ID.
        key: Hash key, from the seed.

    Returns:
        The number.
    """
    digest = hashlib.blake2b(doc_id.encode("utf-8"), digest_size=8, key=key).digest()
    return int.from_bytes(digest, "big") / 2**64


def sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file.

    Args:
        path: File to hash.

    Returns:
        The digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: Sequence[str] | None = None) -> None:
    """Select the documents of one file and write them.

    Args:
        argv: Command-line arguments, ``sys.argv[1:]`` if None.

    Raises:
        ValueError: On a malformed line, a missing field, an unknown pair or a tab in an ID.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    probabilities = load_probabilities(args.mixture)
    key = str(args.seed).encode("utf-8")
    labels_path = args.labels
    for path in (args.output, labels_path, args.summary):
        path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = args.output.with_name(args.output.name + ".tmp")
    tmp_labels = labels_path.with_name(labels_path.name + ".tmp")
    LOGGER.info("Selecting from %s (%s) with %s, seed %d", args.input, zstd_backend(), args.mixture, args.seed)

    pairs: dict[str, list[int]] = {}  # "topic\tformat" -> input docs, input chars, selected docs, selected chars
    totals = [0, 0, 0, 0]
    start = last_log = time.monotonic()
    with open_zst(args.input) as source, tmp_output.open("w", encoding="utf-8") as out, \
            tmp_labels.open("w", encoding="utf-8") as labels_out:
        for line_number, line in enumerate(source, start=1):
            try:
                doc: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{args.input}:{line_number}: malformed JSON: {error}") from error
            missing = [field for field in REQUIRED_FIELDS if not isinstance(doc.get(field), str)]
            if missing:
                raise ValueError(f"{args.input}:{line_number}: missing or non-string fields {missing}")
            pair = (doc["topic"], doc["format"])
            if pair not in probabilities:
                raise ValueError(f"{args.input}:{line_number}: pair {pair} is not in the mixture")
            if "\t" in doc["id"] or "\n" in doc["id"]:
                raise ValueError(f"{args.input}:{line_number}: tab or newline in ID {doc['id']!r}")
            counts = pairs.setdefault(f"{pair[0]}\t{pair[1]}", [0, 0, 0, 0])
            characters = len(doc["text"])
            counts[0] += 1
            counts[1] += characters
            if uniform(doc["id"], key) < probabilities[pair]:
                out.write(json.dumps({"id": doc["id"], "text": doc["text"]}, ensure_ascii=False) + "\n")
                labels_out.write(f"{doc['id']}\t{pair[0]}\t{pair[1]}\n")
                counts[2] += 1
                counts[3] += characters
            now = time.monotonic()
            if now - last_log >= args.log_every_seconds:
                LOGGER.info("%d documents read, %.0f per second", line_number, line_number / (now - start))
                last_log = now

    for counts in pairs.values():
        for i, value in enumerate(counts):
            totals[i] += value
    summary = {
        "input": str(args.input),
        "mixture": str(args.mixture),
        "mixture_sha256": sha256(args.mixture),
        "seed": args.seed,
        "documents": totals[0],
        "characters": totals[1],
        "selected_documents": totals[2],
        "selected_characters": totals[3],
        "pairs": {name: dict(zip(("documents", "characters", "selected_documents", "selected_characters"), counts))
                  for name, counts in sorted(pairs.items())},
    }
    tmp_summary = args.summary.with_name(args.summary.name + ".tmp")
    tmp_summary.write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp_output.replace(args.output)
    tmp_labels.replace(labels_path)
    # The summary last: its presence means the selection is complete.
    tmp_summary.replace(args.summary)
    LOGGER.info("Selected %d of %d documents (%.2f%% of the characters) in %.0f s; wrote %s",
                totals[2], totals[0], 100 * totals[3] / max(totals[1], 1), time.monotonic() - start, args.output)


if __name__ == "__main__":
    main()
