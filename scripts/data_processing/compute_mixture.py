"""Compute the WebOrganizer topic x format training mixture for one language.

Turns the label proportions of one language (``label_proportions.py``) and a WebOrganizer
domain mixture (Table 9 of Wettig et al. 2025, e.g. ``configs/mixtures/weborganizer_both.json``)
into the amount of text to select from every topic x format pair, and the probability with
which each document of a pair is selected (``select_mixture_documents.py``).

The mixture, by pair (top topic x top format of each document):

* **Bad example** (as topic or format): dropped, i.e. used as a quality filter.
* **Other** (as topic or format, without *Bad example*): fixed at the pair's natural share
  of the full corpus, since the WebOrganizer mixture has no weight for it.
* **The 24 x 24 WebOrganizer pairs** fill the rest of the budget, in proportion to the
  product of the topic and format weights (the paper's Section 4.2). A pair can give at
  most ``--max_epochs`` times its text; the budget of a pair that hits this cap goes to the
  other pairs, in proportion to their weights (water-filling), as in the paper.

Text is measured in characters. The token budget is converted to characters with one
``--chars_per_token`` ratio of the training tokenizer, so the ratio only sets the size of the
selection and the caps, not the shares. The selection is ``--margin`` times the training
budget, because Megatron holds out part of the data for validation (``--split 99,1,0``) and
the ratio is an estimate; training then uses a random subset of the selection.

Writes a JSON file with the settings, the per-pair selection and the topic and format
shares (natural, WebOrganizer target, realized), and one row per pair to
``<output stem>.pairs.csv``. Stops with an error if the mixture's labels do not match the
classifiers' labels or if the corpus cannot fill the budget.

Example::

    python scripts/data_processing/compute_mixture.py \\
        --proportions results/statistics/classifier_label_proportions/hplt-4.0-clean/deu_Latn.json \\
        --mixture configs/mixtures/weborganizer_both.json \\
        --chars_per_token 4.43 \\
        --output results/mixtures/hplt-4.0-clean/deu_Latn/both.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

KINDS = ("topic", "format")
OTHER = "Other"
BAD = "Bad example"


@dataclass
class Pair:
    """The selection of one topic x format pair.

    Attributes:
        topic: Top topic label.
        format: Top format label.
        group: ``bad`` (dropped), ``other`` (natural share) or ``weborganizer`` (mixture).
        documents: Documents of the pair in the corpus.
        characters: Characters of the pair in the corpus.
        weight: Normalized WebOrganizer weight (product of topic and format weight) for
            ``weborganizer`` pairs, the natural share for ``other`` pairs, 0 for ``bad``.
        selected_characters: Characters to select.
        selection_probability: Probability with which each document is selected (may be
            above 1 only with ``--max_epochs`` above 1).
        capped: Whether the pair gives all the text it may (``--max_epochs`` times its text).
    """

    topic: str
    format: str
    group: str
    documents: int
    characters: int
    weight: float
    selected_characters: float = 0.0
    selection_probability: float = 0.0
    capped: bool = False


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command-line arguments.

    Args:
        argv: Arguments to parse, ``sys.argv[1:]`` if None.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--proportions", type=Path, required=True, help="JSON output of label_proportions.py")
    parser.add_argument("--mixture", type=Path, required=True,
                        help="JSON with 'topic' and 'format' weights of the 24 WebOrganizer labels each")
    parser.add_argument("--output", type=Path, required=True, help="JSON file to write the selection to")
    parser.add_argument("--train_tokens", type=float, default=100e9, help="Tokens the model trains on")
    parser.add_argument("--chars_per_token", type=float, required=True,
                        help="Characters per token of the training tokenizer on this language's text")
    parser.add_argument("--margin", type=float, default=1.05,
                        help="Select this many times the training budget (validation split, ratio error)")
    parser.add_argument("--max_epochs", type=float, default=1.0,
                        help="At most this many times a pair's text (1: no repetition)")
    return parser.parse_args(argv)


def load_mixture(path: Path, labels: dict[str, list[str]]) -> dict[str, dict[str, float]]:
    """Load the WebOrganizer weights and check them against the classifiers' labels.

    Args:
        path: JSON file with a ``topic`` and a ``format`` object of label weights.
        labels: The classifiers' labels per kind, from the label proportions.

    Returns:
        The weights per kind, normalized to sum to 1.

    Raises:
        ValueError: If the labels differ from the classifiers' labels without *Other* and
            *Bad example*, or if a weight is negative or the weights do not sum to about 100.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    mixture = {}
    for kind in KINDS:
        weights = data[kind]
        expected = set(labels[kind]) - {OTHER, BAD}
        if set(weights) != expected:
            raise ValueError(f"{path}: {kind} labels differ from the classifiers' labels: "
                             f"missing {sorted(expected - set(weights))}, unknown {sorted(set(weights) - expected)}")
        if any(w < 0 for w in weights.values()):
            raise ValueError(f"{path}: negative {kind} weight")
        total = sum(weights.values())
        # The printed percentages are rounded to 0.1, so they need not sum to exactly 100.
        if not 98.0 <= total <= 102.0:
            raise ValueError(f"{path}: {kind} weights sum to {total}, not about 100")
        mixture[kind] = {label: w / total for label, w in weights.items()}
    return mixture


def build_pairs(proportions: dict[str, Any], mixture: dict[str, dict[str, float]]) -> list[Pair]:
    """Group the corpus's pairs and give each its weight.

    Args:
        proportions: Output of label_proportions.py.
        mixture: Normalized WebOrganizer weights per kind.

    Returns:
        One entry per topic x format pair, in label order.
    """
    labels = proportions["labels"]
    total_characters = proportions["total"]["characters"]
    characters = proportions["total"]["pairs"]["characters"]
    documents = proportions["total"]["pairs"]["documents"]
    pairs = []
    for i, topic in enumerate(labels["topic"]):
        for j, fmt in enumerate(labels["format"]):
            if BAD in (topic, fmt):
                group, weight = "bad", 0.0
            elif OTHER in (topic, fmt):
                group, weight = "other", characters[i][j] / total_characters
            else:
                group, weight = "weborganizer", mixture["topic"][topic] * mixture["format"][fmt]
            pairs.append(Pair(topic, fmt, group, documents[i][j], characters[i][j], weight))
    return pairs


def water_fill(pairs: list[Pair], budget: float, max_epochs: float) -> None:
    """Share a budget between pairs in proportion to their weights, capping each pair.

    A pair gets ``scale * weight`` characters, but at most ``max_epochs`` times its text;
    ``scale`` is chosen so that the pairs together get the budget.

    Args:
        pairs: Pairs to fill; their ``selected_characters`` and ``capped`` are set.
        budget: Characters to share.
        max_epochs: Cap of each pair, as a multiple of its text.

    Raises:
        ValueError: If the pairs cannot fill the budget even when all are capped.
    """
    fillable = [p for p in pairs if p.weight > 0]
    capacity = sum(p.characters * max_epochs for p in fillable)
    if capacity < budget:
        raise ValueError(f"The pairs hold {capacity:.4g} characters at {max_epochs} epochs, "
                         f"less than the budget of {budget:.4g}")
    remaining_budget = budget
    remaining_weight = sum(p.weight for p in fillable)
    # Pairs in the order in which a growing scale reaches their cap.
    order = sorted(fillable, key=lambda p: p.characters * max_epochs / p.weight)
    for pair in order:
        cap = pair.characters * max_epochs
        if cap < remaining_budget / remaining_weight * pair.weight:
            pair.selected_characters = cap
            pair.capped = True
            remaining_budget -= cap
            remaining_weight -= pair.weight
        else:
            break
    scale = remaining_budget / remaining_weight
    for pair in fillable:
        if not pair.capped:
            pair.selected_characters = scale * pair.weight


def shares(pairs: list[Pair], labels: dict[str, list[str]], mixture: dict[str, dict[str, float]],
           total_characters: int) -> dict[str, dict[str, dict[str, float]]]:
    """Compute the natural, target and realized share of every topic and format.

    Args:
        pairs: Pairs with their selection.
        labels: The classifiers' labels per kind.
        mixture: Normalized WebOrganizer weights per kind.
        total_characters: Characters of the full corpus.

    Returns:
        Per kind and label: ``natural`` (share of the corpus), ``selected`` (share of the
        selection), ``weborganizer_target`` and ``weborganizer_realized`` (shares within the
        24 x 24 WebOrganizer pairs, comparable to each other; absent for Other and Bad example).
    """
    selected_total = sum(p.selected_characters for p in pairs)
    block_total = sum(p.selected_characters for p in pairs if p.group == "weborganizer")
    result: dict[str, dict[str, dict[str, float]]] = {}
    for kind in KINDS:
        result[kind] = {}
        for label in labels[kind]:
            members = [p for p in pairs if getattr(p, kind) == label]
            entry = {
                "natural": sum(p.characters for p in members) / total_characters,
                "selected": sum(p.selected_characters for p in members) / selected_total,
            }
            if label in mixture[kind]:
                entry["weborganizer_target"] = mixture[kind][label]
                entry["weborganizer_realized"] = sum(
                    p.selected_characters for p in members if p.group == "weborganizer") / block_total
            result[kind][label] = entry
    return result


def write_pairs_csv(path: Path, pairs: list[Pair], chars_per_token: float) -> None:
    """Write one row per pair.

    Args:
        path: CSV file to write.
        pairs: Pairs with their selection.
        chars_per_token: Ratio used to estimate tokens.
    """
    selected_total = sum(p.selected_characters for p in pairs)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["topic", "format", "group", "documents", "characters", "selected_characters",
                         "selected_tokens_estimate", "selected_share", "selection_probability", "capped"])
        for p in pairs:
            writer.writerow([p.topic, p.format, p.group, p.documents, p.characters, round(p.selected_characters),
                             round(p.selected_characters / chars_per_token), f"{p.selected_characters / selected_total:.8f}",
                             f"{p.selection_probability:.8f}", int(p.capped)])


def log_shares(kind: str, kind_shares: dict[str, dict[str, float]]) -> None:
    """Log the shares of one kind as a table, largest WebOrganizer target first.

    Args:
        kind: ``topic`` or ``format``.
        kind_shares: Shares per label, from :func:`shares`.
    """
    LOGGER.info("%-20s %8s %8s %8s %8s", kind, "natural", "target", "realized", "selected")
    order = sorted(kind_shares, key=lambda label: -kind_shares[label].get("weborganizer_target", -1))
    for label in order:
        s = kind_shares[label]
        target = f"{s['weborganizer_target']:.2%}" if "weborganizer_target" in s else "-"
        realized = f"{s['weborganizer_realized']:.2%}" if "weborganizer_realized" in s else "-"
        LOGGER.info("%-20s %8.2f%% %8s %8s %8.2f%%", label, 100 * s["natural"], target, realized, 100 * s["selected"])


def main(argv: Sequence[str] | None = None) -> None:
    """Compute the mixture and write it.

    Args:
        argv: Command-line arguments, ``sys.argv[1:]`` if None.

    Raises:
        ValueError: If an argument is out of range or the Other pairs cannot give their share.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    if args.chars_per_token <= 0 or args.margin < 1 or args.max_epochs <= 0 or args.train_tokens <= 0:
        raise ValueError("--chars_per_token, --train_tokens and --max_epochs must be positive and --margin at least 1")

    proportions = json.loads(args.proportions.read_text(encoding="utf-8"))
    labels = proportions["labels"]
    mixture = load_mixture(args.mixture, labels)
    pairs = build_pairs(proportions, mixture)
    total_characters = proportions["total"]["characters"]
    budget = args.train_tokens * args.margin * args.chars_per_token

    # Other pairs: their natural share of the selection, the rest goes to the WebOrganizer pairs.
    others = [p for p in pairs if p.group == "other"]
    for pair in others:
        pair.selected_characters = pair.weight * budget
        if pair.selected_characters > pair.characters * args.max_epochs:
            raise ValueError(f"{pair.topic} x {pair.format}: its natural share needs more than {args.max_epochs} epochs")
    block = [p for p in pairs if p.group == "weborganizer"]
    block_weight = sum(p.weight for p in block)
    for pair in block:
        pair.weight /= block_weight
    water_fill(block, budget - sum(p.selected_characters for p in others), args.max_epochs)
    for pair in pairs:
        pair.selection_probability = pair.selected_characters / pair.characters if pair.characters else 0.0
    selected = sum(p.selected_characters for p in pairs)
    if not math.isclose(selected, budget, rel_tol=1e-9):
        raise RuntimeError(f"Selected {selected} characters instead of the budget of {budget}")

    label_shares = shares(pairs, labels, mixture, total_characters)
    capped = [p for p in pairs if p.capped]
    bad_share = sum(p.characters for p in pairs if p.group == "bad") / total_characters
    other_share = sum(p.weight for p in others)
    distance = {kind: 0.5 * sum(abs(s["weborganizer_realized"] - s["weborganizer_target"])
                                for s in label_shares[kind].values() if "weborganizer_target" in s)
                for kind in KINDS}

    LOGGER.info("Corpus: %d documents, %.4g characters (~%.4g tokens at %.2f characters per token)",
                proportions["total"]["documents"], total_characters, total_characters / args.chars_per_token,
                args.chars_per_token)
    LOGGER.info("Selection: %.4g characters (~%.4g tokens, %.1f%% of the corpus), expected %d documents",
                budget, budget / args.chars_per_token, 100 * budget / total_characters,
                round(sum(p.documents * min(p.selection_probability, 1.0) for p in pairs)))
    LOGGER.info("Bad example pairs dropped: %.2f%% of the corpus; Other pairs fixed at %.2f%%",
                100 * bad_share, 100 * other_share)
    LOGGER.info("%d WebOrganizer pairs capped at %g epochs, holding %.1f%% of the selection",
                len(capped), args.max_epochs, 100 * sum(p.selected_characters for p in capped) / budget)
    for kind in KINDS:
        LOGGER.info("Total variation distance from the WebOrganizer %s target: %.1f%%", kind, 100 * distance[kind])
        log_shares(kind, label_shares[kind])

    result = {
        "settings": {
            "proportions": str(args.proportions),
            "mixture": str(args.mixture),
            "mixture_source": json.loads(args.mixture.read_text(encoding="utf-8")).get("source"),
            "train_tokens": args.train_tokens,
            "chars_per_token": args.chars_per_token,
            "margin": args.margin,
            "max_epochs": args.max_epochs,
            "selected_characters": budget,
            "variant": "Bad example dropped; Other pairs at their natural share of the full corpus; "
                       "WebOrganizer pairs by product of topic and format weights, water-filled under the cap",
        },
        "summary": {
            "corpus_characters": total_characters,
            "corpus_documents": proportions["total"]["documents"],
            "bad_example_share": bad_share,
            "other_share": other_share,
            "capped_pairs": len(capped),
            "capped_share_of_selection": sum(p.selected_characters for p in capped) / budget,
            "total_variation_distance": distance,
        },
        "labels": labels,
        "shares": label_shares,
        "pairs": [asdict(p) for p in pairs],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name(args.output.name + ".tmp")
    tmp.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(args.output)
    write_pairs_csv(args.output.with_suffix(".pairs.csv"), pairs, args.chars_per_token)
    LOGGER.info("Wrote %s and %s", args.output, args.output.with_suffix(".pairs.csv"))


if __name__ == "__main__":
    main()
