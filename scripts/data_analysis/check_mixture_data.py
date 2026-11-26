"""Check the tokenized training data of a mixture against the planned mixture.

Reads every Megatron index file of ``build_mixture_data.sh``'s output for one language
(``<megatron_dir>/<name>_text_document.idx``) with the matching ``<selection_dir>/<name>.
labels.tsv`` (id, topic and format of each document, in the same order), and counts the
tokens of every topic x format pair. Logs the topic and format token shares next to the
shares planned by ``compute_mixture.py`` (which were in characters), and the total.

Stops with an error if a file is missing, if an index file has another number of documents
than its labels file, or if the training split (``--train_fraction`` of the documents, as in
Megatron's ``--split``) holds fewer tokens than ``--train_tokens``: Megatron would then
repeat data.

Example::

    python scripts/data_analysis/check_mixture_data.py \\
        --mixture results/mixtures/hplt-4.0-clean/deu_Latn/both.json \\
        --megatron_dir data/megatron/oellm-v1-256k/hplt-4.0-clean-both/deu_Latn \\
        --selection_dir data/megatron/oellm-v1-256k/hplt-4.0-clean-both/selection/deu_Latn \\
        --output results/mixtures/hplt-4.0-clean/deu_Latn/both.tokens.json
"""

from __future__ import annotations

import argparse
import json
import logging
import struct
from array import array
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

LOGGER = logging.getLogger(__name__)

MAGIC = b"MMIDIDX\x00\x00"
SUFFIX = "_text_document.idx"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command-line arguments.

    Args:
        argv: Arguments to parse, ``sys.argv[1:]`` if None.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mixture", type=Path, required=True, help="JSON output of compute_mixture.py")
    parser.add_argument("--megatron_dir", type=Path, required=True, help="Directory of the .bin and .idx files")
    parser.add_argument("--selection_dir", type=Path, required=True, help="Directory of the .labels.tsv files")
    parser.add_argument("--output", type=Path, required=True, help="JSON file to write the token counts to")
    parser.add_argument("--train_tokens", type=float, default=100e9, help="Tokens the model trains on")
    parser.add_argument("--train_fraction", type=float, default=0.99,
                        help="Fraction of the documents in Megatron's training split")
    return parser.parse_args(argv)


def sequence_lengths(path: Path) -> array:
    """Read the sequence lengths of a Megatron index file.

    Args:
        path: ``.idx`` file (format version 1).

    Returns:
        The length in tokens of every sequence, in order.

    Raises:
        ValueError: If the file is not a version 1 Megatron index file or is cut off.
    """
    with path.open("rb") as fh:
        header = fh.read(len(MAGIC) + 8 + 1 + 8 + 8)
        if header[:len(MAGIC)] != MAGIC:
            raise ValueError(f"{path}: not a Megatron index file")
        version, _dtype, count, _documents = struct.unpack("<QBQQ", header[len(MAGIC):])
        if version != 1:
            raise ValueError(f"{path}: index format version {version}, expected 1")
        lengths = array("i")
        data = fh.read(4 * count)
        if len(data) != 4 * count:
            raise ValueError(f"{path}: cut off in the sequence lengths")
        lengths.frombytes(data)
    if lengths.itemsize != 4:
        raise RuntimeError("array('i') is not 32-bit on this platform")
    return lengths


def main(argv: Sequence[str] | None = None) -> None:
    """Count the tokens per pair and compare them with the plan.

    Args:
        argv: Command-line arguments, ``sys.argv[1:]`` if None.

    Raises:
        ValueError: If the files do not match each other or the data is too small.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    mixture = json.loads(args.mixture.read_text(encoding="utf-8"))
    index_files = sorted(args.megatron_dir.glob(f"*{SUFFIX}"))
    if not index_files:
        raise ValueError(f"No *{SUFFIX} files in {args.megatron_dir}")

    tokens: dict[tuple[str, str], int] = defaultdict(int)
    documents = 0
    for index_file in index_files:
        name = index_file.name[:-len(SUFFIX)]
        labels_file = args.selection_dir / f"{name}.labels.tsv"
        lengths = sequence_lengths(index_file)
        count = 0
        with labels_file.open(encoding="utf-8") as fh:
            for count, line in enumerate(fh, start=1):
                if count > len(lengths):
                    break
                _doc_id, topic, fmt = line.rstrip("\n").split("\t")
                tokens[(topic, fmt)] += lengths[count - 1]
        if count != len(lengths):
            found = f"more than {len(lengths)}" if count > len(lengths) else str(count)
            raise ValueError(f"{index_file} has {len(lengths)} documents but {labels_file} has {found}")
        documents += len(lengths)
    total = sum(tokens.values())

    planned = {(p["topic"], p["format"]): p["selected_characters"] for p in mixture["pairs"]}
    planned_total = sum(planned.values())
    LOGGER.info("%d files, %d documents, %d tokens (planned %.4g)", len(index_files), documents, total,
                mixture["settings"]["train_tokens"] * mixture["settings"]["margin"])
    for kind, position in (("topic", 0), ("format", 1)):
        LOGGER.info("%-20s %9s %9s", kind, "planned", "tokens")
        for label in mixture["labels"][kind]:
            got = sum(v for k, v in tokens.items() if k[position] == label) / total
            plan = sum(v for k, v in planned.items() if k[position] == label) / planned_total
            LOGGER.info("%-20s %8.2f%% %8.2f%%", label, 100 * plan, 100 * got)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "files": len(index_files),
        "documents": documents,
        "tokens": total,
        "pairs": [{"topic": t, "format": f, "tokens": n} for (t, f), n in sorted(tokens.items())],
    }, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    train_tokens = total * args.train_fraction
    if train_tokens < args.train_tokens:
        raise ValueError(f"The training split holds about {train_tokens:.4g} tokens, fewer than "
                         f"{args.train_tokens:.4g}: Megatron would repeat data")
    LOGGER.info("Training split about %.4g tokens, %.1f%% above the %.4g trained on; wrote %s",
                train_tokens, 100 * (train_tokens / args.train_tokens - 1), args.train_tokens, args.output)


if __name__ == "__main__":
    main()
