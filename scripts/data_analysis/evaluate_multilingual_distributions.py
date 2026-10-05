"""Measure how consistently a model labels parallel documents across languages.

Inputs are outputs of ``generate_label_distributions.py`` for the same parallel
documents in several languages. Each language is compared with English, and
the report adds consistency across all languages at once, a pairwise matrix,
and optionally a noise floor from a second English run with a different seed.

Parallel documents should be labelled with a different seed per language (as
``array_generate_distribution.sh`` does). Otherwise every language sees the same
letter order, and a model's letter preferences inflate consistency.
"""

from __future__ import annotations

import argparse
import logging
import math
import statistics
from collections.abc import Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

try:
    from .distribution_metrics import (
        DistributionRecord,
        cohen_kappa,
        confidence_profile,
        generalized_jensen_shannon_bits,
        jensen_shannon_bits,
        label_shares,
        letter_bias,
        random_pair_jsd_bits,
        ranked_indices,
        read_distribution_file,
        share_divergence_bits,
        spearman,
        summarize,
        top_confusions,
        top_index,
        write_report,
    )
except ImportError:
    from distribution_metrics import (
        DistributionRecord,
        cohen_kappa,
        confidence_profile,
        generalized_jensen_shannon_bits,
        jensen_shannon_bits,
        label_shares,
        letter_bias,
        random_pair_jsd_bits,
        ranked_indices,
        read_distribution_file,
        share_divergence_bits,
        spearman,
        summarize,
        top_confusions,
        top_index,
        write_report,
    )

LOGGER = logging.getLogger(__name__)


def shared_ids(
    reference_ids: Sequence[str], candidate_ids: Sequence[str], name: str
) -> list[str]:
    """Return the documents present in both files, in reference order.

    Runs over the same parallel data stream the same documents in the same
    order, possibly with different ``max_documents``, so one file's documents
    must be a subset of the other's.

    Raises:
        ValueError: If neither file's documents are a subset of the other's.
    """
    reference_set, candidate_set = set(reference_ids), set(candidate_ids)
    if not (candidate_set <= reference_set or reference_set <= candidate_set):
        only_candidate = len(candidate_set - reference_set)
        only_reference = len(reference_set - candidate_set)
        raise ValueError(
            f"{name}: documents do not match English ({only_candidate} only in "
            f"{name}, {only_reference} only in English); the files must cover "
            "the same parallel documents"
        )
    return [doc_id for doc_id in reference_ids if doc_id in candidate_set]


def compare(
    reference: Sequence[DistributionRecord],
    candidate: Sequence[DistributionRecord],
    labels: Sequence[str],
    seed: int,
) -> dict[str, Any]:
    """Compare aligned English and candidate-language distributions."""
    n = len(reference)
    pairs = [(top_index(r.logprobs), top_index(c.logprobs)) for r, c in zip(reference, candidate)]
    ranked = [ranked_indices(c.logprobs) for c in candidate]
    jsd = [jensen_shannon_bits(r.logprobs, c.logprobs) for r, c in zip(reference, candidate)]
    random_jsd = random_pair_jsd_bits(
        [r.logprobs for r in reference], [c.logprobs for c in candidate], seed
    )
    mean_jsd = statistics.fmean(jsd)
    _, reference_vector = label_shares((a for a, _ in pairs), labels)
    candidate_shares, candidate_vector = label_shares((b for _, b in pairs), labels)
    return {
        "documents": n,
        "agreement": {
            "top1_agreement": sum(a == b for a, b in pairs) / n,
            "cohen_kappa": cohen_kappa(pairs, len(labels)),
            "english_top_in_top2": sum(a in r[:2] for (a, _), r in zip(pairs, ranked)) / n,
        },
        "soft": {
            "jsd_bits": summarize(jsd),
            "random_pair_jsd_bits": random_jsd,
            # 1 = parallel documents as similar as identical inputs; 0 = no more
            # similar than unrelated documents. Does not reward flat distributions.
            "normalized_consistency": 1 - mean_jsd / random_jsd if random_jsd > 0 else float("nan"),
            "english_label_log_loss_bits": statistics.fmean(
                -c.logprobs[a] / math.log(2) for c, (a, _) in zip(candidate, pairs)
            ),
        },
        "confidence": {
            "english": confidence_profile(reference),
            "language": confidence_profile(candidate),
            "top_prob_spearman": spearman(
                [max(r.logprobs) for r in reference], [max(c.logprobs) for c in candidate]
            ),
        },
        "label_shares": {
            "divergence_bits": share_divergence_bits(reference_vector, candidate_vector),
            "language": candidate_shares,
        },
        "letter_bias": letter_bias(candidate),
        "top_label_flips": top_confusions(pairs, labels),
    }


def _parse_candidate(value: str) -> tuple[str, Path]:
    """Parse a CLI candidate in the form ``language=path``."""
    language, separator, path = value.partition("=")
    if not separator or not language.strip() or not path.strip():
        raise argparse.ArgumentTypeError("candidate must use LANGUAGE=PATH")
    return language.strip(), Path(path)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Measure label distribution consistency across parallel languages."
    )
    parser.add_argument("--english", type=Path, required=True, help="English JSONL")
    parser.add_argument(
        "--candidate",
        type=_parse_candidate,
        action="append",
        required=True,
        metavar="LANGUAGE=PATH",
        help="Candidate JSONL; repeat for multiple languages",
    )
    parser.add_argument(
        "--english-rerun",
        type=Path,
        help="Second English run with a different seed, for the noise floor",
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        help="Use only the first N English documents",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for the random-pair baseline"
    )
    parser.add_argument("--output", type=Path, help="Write JSON report to this path")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default="INFO",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Evaluate all candidate languages against English."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.max_documents is not None and args.max_documents < 2:
        parser.error("--max-documents must be >= 2")
    languages = [language for language, _ in args.candidate]
    lowered = [language.lower() for language in languages]
    if len(set(lowered)) != len(lowered) or "english" in lowered:
        parser.error("candidate languages must be unique and must not include English")
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    labels, english = read_distribution_file(args.english)
    english_ids = list(english)[: args.max_documents]
    runs: dict[str, dict[str, DistributionRecord]] = {}
    for language, path in args.candidate:
        candidate_labels, records = read_distribution_file(path)
        if candidate_labels != labels:
            raise ValueError(f"{path} has different labels than {args.english}")
        runs[language] = records

    report: dict[str, Any] = {
        "reference": str(args.english),
        "english_documents": len(english_ids),
        "languages": {},
    }
    for language, records in runs.items():
        ids = shared_ids(english_ids, list(records), language)
        if len(ids) < 2:
            raise ValueError(f"{language}: fewer than 2 documents shared with English")
        LOGGER.info("Evaluating %s on %d documents", language, len(ids))
        report["languages"][language] = compare(
            [english[i] for i in ids], [records[i] for i in ids], labels, args.seed
        )

    # Consistency across all languages at once, on documents present everywhere.
    common = [i for i in english_ids if all(i in records for records in runs.values())]
    if len(common) >= 2:
        all_runs = {"english": english, **runs}
        report["all_languages"] = {
            "languages": list(all_runs),
            "documents": len(common),
            "generalized_jsd_bits": summarize(
                [
                    generalized_jensen_shannon_bits([r[i].logprobs for r in all_runs.values()])
                    for i in common
                ]
            ),
            "unanimous_top1": sum(
                len({top_index(r[i].logprobs) for r in all_runs.values()}) == 1 for i in common
            )
            / len(common),
        }
        pairwise = {}
        for (name_a, run_a), (name_b, run_b) in combinations(all_runs.items(), 2):
            pairwise[f"{name_a}|{name_b}"] = {
                "top1_agreement": sum(
                    top_index(run_a[i].logprobs) == top_index(run_b[i].logprobs) for i in common
                )
                / len(common),
                "mean_jsd_bits": statistics.fmean(
                    jensen_shannon_bits(run_a[i].logprobs, run_b[i].logprobs) for i in common
                ),
            }
        report["pairwise"] = pairwise
    else:
        LOGGER.warning("Fewer than 2 documents are shared by all languages; skipping all-language metrics")

    if args.english_rerun is not None:
        rerun_labels, rerun = read_distribution_file(args.english_rerun)
        if rerun_labels != labels:
            raise ValueError(f"{args.english_rerun} has different labels than {args.english}")
        ids = shared_ids(english_ids, list(rerun), "english rerun")
        # Same documents and language, different letter order: agreement no
        # language can be expected to exceed.
        report["noise_floor"] = compare(
            [english[i] for i in ids], [rerun[i] for i in ids], labels, args.seed
        )

    write_report(report, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
