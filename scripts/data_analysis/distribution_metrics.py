"""Shared loading and metrics for evaluating label probability distributions.

Used by ``evaluate_distribution_reference.py`` (agreement with WebOrganizer's
labels) and ``evaluate_multilingual_distributions.py`` (consistency across
parallel languages). Distributions are handled as log-probabilities in a fixed
label order, and divergences are reported in bits.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import statistics
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

LOGGER = logging.getLogger(__name__)

# Allowed deviation of a distribution's total probability from one.
PROBABILITY_SUM_TOLERANCE = 1e-3


@dataclass(frozen=True)
class DistributionRecord:
    """One document's label distribution.

    Attributes:
        doc_id: Document ID.
        logprobs: Log-probability per label, in the file's label order.
        letter_positions: Letter position (0 = ``A``) at which each label was
            shown to the model, in the same order as ``logprobs``.
        num_letters: Number of letters shown to the model. Equals the number of
            labels, unless the record was restricted to fewer labels with
            :func:`restrict_to_labels`.
    """

    doc_id: str
    logprobs: tuple[float, ...]
    letter_positions: tuple[int, ...]
    num_letters: int


def _logsumexp(values: Sequence[float]) -> float:
    """Return ``log(sum(exp(values)))`` computed in a numerically stable way."""
    max_value = max(values)
    return max_value + math.log(math.fsum(math.exp(v - max_value) for v in values))


def normalize_logprobs(values: Sequence[float]) -> tuple[float, ...]:
    """Normalize log-scores into log-probabilities that sum to one."""
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError(f"log-scores must be non-empty and finite; got {values!r}")
    log_normalizer = _logsumexp(values)
    return tuple(v - log_normalizer for v in values)


def read_distribution_file(
    path: Path,
) -> tuple[list[str], dict[str, DistributionRecord]]:
    """Stream and validate an output file of ``generate_label_distributions.py``.

    Args:
        path: Distribution JSONL file.

    Returns:
        The label names in file order and the records keyed by document ID,
        in file order.

    Raises:
        ValueError: If a record is malformed, labels differ between records,
            probabilities do not sum to one, or a document ID repeats.
    """
    if not path.is_file():
        raise FileNotFoundError(path)
    labels: list[str] | None = None
    records: dict[str, DistributionRecord] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            location = f"{path}:{line_number}"
            if not line.strip():
                raise ValueError(f"Empty JSONL line in {location}")
            try:
                raw = json.loads(line)
                doc_id = raw["id"]
                logprob_map = raw["label_logprobs"]
                label_order = raw["label_order"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"Invalid distribution record in {location}: {exc}") from exc
            if not isinstance(doc_id, str) or not doc_id:
                raise ValueError(f"Invalid id in {location}")
            if not isinstance(logprob_map, dict) or not isinstance(label_order, list):
                raise ValueError(f"Invalid label fields in {location}")
            if labels is None:
                labels = list(logprob_map)
            if list(logprob_map) != labels or sorted(label_order) != sorted(labels):
                raise ValueError(f"Labels in {location} differ from the first record")
            logprobs = tuple(float(logprob_map[label]) for label in labels)
            if not all(math.isfinite(v) for v in logprobs):
                raise ValueError(f"Non-finite log-probability in {location}")
            if abs(math.expm1(_logsumexp(logprobs))) > PROBABILITY_SUM_TOLERANCE:
                raise ValueError(f"Probabilities in {location} do not sum to one")
            if doc_id in records:
                raise ValueError(f"Duplicate document ID {doc_id!r} in {location}")
            position = {label: index for index, label in enumerate(label_order)}
            records[doc_id] = DistributionRecord(
                doc_id=doc_id,
                logprobs=logprobs,
                letter_positions=tuple(position[label] for label in labels),
                num_letters=len(labels),
            )
    if labels is None:
        raise ValueError(f"{path} contains no records")
    return labels, records


def restrict_to_labels(record: DistributionRecord, keep: Sequence[int]) -> DistributionRecord:
    """Return the record's distribution over a subset of labels, renormalized.

    Used to route labels that a reference taxonomy lacks (e.g. ``Other`` and
    ``Bad example``, which WebOrganizer doesn't have) back to the remaining
    labels: their probability is removed and the rest is rescaled to sum to one.
    The letter positions and the number of shown letters are kept.

    Args:
        record: Distribution over the full label set.
        keep: Indices of the labels to keep, in the order wanted.

    Returns:
        A record with one log-probability per kept label.
    """
    if not keep or len(set(keep)) != len(keep):
        raise ValueError("keep must be a non-empty list of distinct label indices")
    return DistributionRecord(
        doc_id=record.doc_id,
        logprobs=normalize_logprobs([record.logprobs[i] for i in keep]),
        letter_positions=tuple(record.letter_positions[i] for i in keep),
        num_letters=record.num_letters,
    )


def probabilities(logprobs: Sequence[float]) -> list[float]:
    """Convert log-probabilities to probabilities."""
    return [math.exp(v) for v in logprobs]


def top_index(logprobs: Sequence[float]) -> int:
    """Return the index of the most probable label (first one on ties)."""
    return max(range(len(logprobs)), key=logprobs.__getitem__)


def ranked_indices(logprobs: Sequence[float]) -> list[int]:
    """Return label indices from most to least probable."""
    return sorted(range(len(logprobs)), key=logprobs.__getitem__, reverse=True)


def entropy_bits(logprobs: Sequence[float]) -> float:
    """Return the Shannon entropy of a distribution in bits."""
    return -math.fsum(math.exp(v) * v for v in logprobs) / math.log(2)


def _entropy_of_probs_bits(probs: Sequence[float]) -> float:
    return -math.fsum(p * math.log2(p) for p in probs if p > 0)


def jensen_shannon_bits(first: Sequence[float], second: Sequence[float]) -> float:
    """Return the Jensen-Shannon divergence of two log-distributions, in bits.

    The value is between 0 (identical) and 1 (disjoint). Unlike KL divergence it
    stays finite when one distribution gives near-zero probability to a label.
    """
    return generalized_jensen_shannon_bits([first, second])


def generalized_jensen_shannon_bits(distributions: Sequence[Sequence[float]]) -> float:
    """Return the Jensen-Shannon divergence of several log-distributions, in bits.

    Computed as the entropy of the mean distribution minus the mean entropy. It
    is zero only if all distributions are identical.
    """
    if len(distributions) < 2:
        raise ValueError("At least two distributions are required")
    prob_lists = [probabilities(d) for d in distributions]
    mean = [math.fsum(column) / len(prob_lists) for column in zip(*prob_lists)]
    mean_entropy = math.fsum(_entropy_of_probs_bits(p) for p in prob_lists) / len(prob_lists)
    return max(0.0, _entropy_of_probs_bits(mean) - mean_entropy)


def summarize(values: Sequence[float]) -> dict[str, float]:
    """Return the mean, median, and 10th/90th percentiles of values."""
    if not values:
        raise ValueError("Cannot summarize an empty sequence")
    ordered = sorted(values)

    def quantile(q: float) -> float:
        return ordered[min(len(ordered) - 1, int(q * len(ordered)))]

    return {
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p10": quantile(0.10),
        "p90": quantile(0.90),
    }


def cohen_kappa(pairs: Sequence[tuple[int, int]], num_labels: int) -> float:
    """Return Cohen's kappa for pairs of (reference, candidate) label indices."""
    n = len(pairs)
    if n == 0:
        raise ValueError("Cannot compute kappa without pairs")
    observed = sum(a == b for a, b in pairs) / n
    first = Counter(a for a, _ in pairs)
    second = Counter(b for _, b in pairs)
    expected = sum(first[i] * second[i] for i in range(num_labels)) / n**2
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def f1_scores(
    pairs: Sequence[tuple[int, int]], labels: Sequence[str]
) -> tuple[float, dict[str, dict[str, float]]]:
    """Return macro F1 and per-label precision, recall, F1, and support.

    The first element of each pair is treated as the reference label.
    Labels that never occur as reference or prediction are left out of the
    macro average.
    """
    per_label: dict[str, dict[str, float]] = {}
    f1_values = []
    for index, name in enumerate(labels):
        tp = sum(a == index and b == index for a, b in pairs)
        fp = sum(a != index and b == index for a, b in pairs)
        fn = sum(a == index and b != index for a, b in pairs)
        if tp + fp + fn == 0:
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[name] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}
        f1_values.append(f1)
    return statistics.fmean(f1_values), per_label


def expected_calibration_error(
    confidences: Sequence[float], correct: Sequence[bool], num_bins: int = 10
) -> tuple[float, list[dict[str, float]]]:
    """Return the expected calibration error and per-bin statistics.

    Args:
        confidences: Top-label probability per document.
        correct: Whether the top label matched the reference.
        num_bins: Number of equal-width confidence bins.

    Returns:
        The count-weighted mean absolute gap between confidence and accuracy,
        and each non-empty bin's range, count, mean confidence, and accuracy.
    """
    if len(confidences) != len(correct) or not confidences:
        raise ValueError("confidences and correct must be equally long and non-empty")
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(num_bins)]
    for confidence, is_correct in zip(confidences, correct):
        bins[min(num_bins - 1, int(confidence * num_bins))].append((confidence, is_correct))
    ece = 0.0
    details = []
    for index, members in enumerate(bins):
        if not members:
            continue
        mean_confidence = statistics.fmean(c for c, _ in members)
        accuracy = sum(k for _, k in members) / len(members)
        ece += len(members) / len(confidences) * abs(mean_confidence - accuracy)
        details.append(
            {
                "low": index / num_bins,
                "high": (index + 1) / num_bins,
                "count": len(members),
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
            }
        )
    return ece, details


def spearman(first: Sequence[float], second: Sequence[float]) -> float:
    """Return Spearman's rank correlation, using average ranks for ties."""
    if len(first) != len(second) or len(first) < 2:
        raise ValueError("Spearman correlation needs two equally long sequences")

    def ranks(values: Sequence[float]) -> list[float]:
        order = sorted(range(len(values)), key=values.__getitem__)
        result = [0.0] * len(values)
        start = 0
        while start < len(order):
            end = start
            while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
                end += 1
            for position in range(start, end + 1):
                result[order[position]] = (start + end) / 2
            start = end + 1
        return result

    rank_a, rank_b = ranks(first), ranks(second)
    mean_a, mean_b = statistics.fmean(rank_a), statistics.fmean(rank_b)
    covariance = math.fsum((a - mean_a) * (b - mean_b) for a, b in zip(rank_a, rank_b))
    variance_a = math.fsum((a - mean_a) ** 2 for a in rank_a)
    variance_b = math.fsum((b - mean_b) ** 2 for b in rank_b)
    if variance_a == 0 or variance_b == 0:
        return float("nan")
    return covariance / math.sqrt(variance_a * variance_b)


def letter_bias(records: Iterable[DistributionRecord]) -> dict[str, float]:
    """Return statistics on how the shown letter relates to the chosen label.

    Labels are shuffled per document, so without bias every letter is equally
    likely to hold the top label. ``winner_chi2`` and ``runner_up_chi2`` test
    that for the top and second label (df = letters - 1; the 5% critical value
    is 35.2 for 24 letters and 37.7 for 26). ``winner_mean_position_z`` is
    negative when early letters are favoured. Use records over the full label
    set: a restricted record can never choose the letters of removed labels.
    """
    records = list(records)
    if not records:
        raise ValueError("Cannot compute letter bias without records")
    num_letters = records[0].num_letters
    if any(r.num_letters != num_letters for r in records):
        raise ValueError("All records must have the same number of shown letters")
    n = len(records)
    expected = n / num_letters
    winners = Counter()
    runners_up = Counter()
    for record in records:
        ranked = ranked_indices(record.logprobs)
        winners[record.letter_positions[ranked[0]]] += 1
        runners_up[record.letter_positions[ranked[1]]] += 1

    def chi2(counts: Counter) -> float:
        return math.fsum((counts[j] - expected) ** 2 / expected for j in range(num_letters))

    mean_position = math.fsum(j * c for j, c in winners.items()) / n
    unbiased_mean = (num_letters - 1) / 2
    standard_error = math.sqrt((num_letters**2 - 1) / 12 / n)
    return {
        "documents": n,
        "winner_chi2": chi2(winners),
        "runner_up_chi2": chi2(runners_up),
        "winner_mean_position": mean_position,
        "winner_mean_position_z": (mean_position - unbiased_mean) / standard_error,
    }


def confidence_profile(records: Iterable[DistributionRecord]) -> dict[str, Any]:
    """Return how peaked the distributions are."""
    records = list(records)
    top_probs = [math.exp(max(r.logprobs)) for r in records]
    return {
        "top_prob": summarize(top_probs),
        "share_top_prob_ge_0.99": sum(p >= 0.99 for p in top_probs) / len(top_probs),
        "share_top_prob_ge_0.999999": sum(p >= 0.999999 for p in top_probs) / len(top_probs),
        "entropy_bits": summarize([entropy_bits(r.logprobs) for r in records]),
    }


def label_shares(
    tops: Iterable[int], labels: Sequence[str]
) -> tuple[dict[str, float], list[float]]:
    """Return the share of documents whose top label is each label."""
    counts = Counter(tops)
    n = sum(counts.values())
    shares = [counts[i] / n for i in range(len(labels))]
    return dict(zip(labels, shares)), shares


def share_divergence_bits(first: Sequence[float], second: Sequence[float]) -> float:
    """Return the Jensen-Shannon divergence between two label-share vectors."""
    mean = [(a + b) / 2 for a, b in zip(first, second)]
    return max(
        0.0,
        _entropy_of_probs_bits(mean)
        - (_entropy_of_probs_bits(first) + _entropy_of_probs_bits(second)) / 2,
    )


def top_confusions(
    pairs: Sequence[tuple[int, int]], labels: Sequence[str], limit: int = 15
) -> list[dict[str, Any]]:
    """Return the most frequent (reference, candidate) top-label disagreements."""
    counts = Counter((a, b) for a, b in pairs if a != b)
    return [
        {"reference": labels[a], "candidate": labels[b], "count": count}
        for (a, b), count in counts.most_common(limit)
    ]


def random_pair_jsd_bits(
    first: Sequence[Sequence[float]], second: Sequence[Sequence[float]], seed: int
) -> float:
    """Return the mean JSD between non-matching documents of two aligned lists.

    Pairs each document of ``first`` with a different, randomly chosen document
    of ``second`` (a derangement), giving the divergence expected when two
    distributions are unrelated. Parallel-document JSD relative to this value
    measures consistency independent of how peaked the distributions are.
    """
    n = len(first)
    if n != len(second) or n < 2:
        raise ValueError("Random pairing needs two aligned lists of at least 2 documents")
    permutation = list(range(n))
    randomizer = random.Random(seed)
    randomizer.shuffle(permutation)
    # Rotate by one within the shuffled order: no document is paired with itself.
    partner = [0] * n
    for position, index in enumerate(permutation):
        partner[index] = permutation[(position + 1) % n]
    return statistics.fmean(
        jensen_shannon_bits(first[i], second[partner[i]]) for i in range(n)
    )


def write_report(report: dict[str, Any], output: Path | None) -> None:
    """Write a JSON report to a file, or to stdout when no path is given."""
    serialized = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if output is None:
        sys.stdout.write(serialized)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized, encoding="utf-8")
        LOGGER.info("Wrote report to %s", output)


def exit_without_finalization(exit_code: int) -> NoReturn:
    """Flush output and exit, skipping interpreter finalization.

    Abandoning a streamed HuggingFace dataset early leaves read-ahead threads
    running, which can abort the interpreter during shutdown (exit code 134)
    after the report has been written. Mirrors the helper in
    ``scripts/data_processing/labeling_common.py``.
    """
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
