"""Rank the trials of a hyperparameter search made with ``finetune.py --hp_search_space``.

Reads ``<search_dir>/trial-*/trial.json``, ranks the completed trials by their best
validation score, writes ``<search_dir>/summary.tsv``, and prints the ranking and the
best trial's settings as ``finetune.py`` flags for the final training run
(``hpc/roihu/train_classifier.sh``). Running and failed trials are listed separately.
"""

from __future__ import annotations

import argparse
import csv
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import hp_search  # sibling module: the script's directory is on sys.path

LOGGER = logging.getLogger(__name__)
# Fields that must be the same in every trial of one search.
SEARCH_FIELDS = ("search_space", "search_seed", "dataset_path", "metric", "greater_is_better",
                 "max_train_samples", "max_eval_samples")


def load_trials(search_dir: Path) -> list[dict[str, Any]]:
    """Read every trial record of a search.

    Raises:
        FileNotFoundError: If the directory has no trial records.
        ValueError: If trials come from different searches (space, seed, data, metric).
    """
    trials = []
    for trial_dir in sorted(search_dir.glob("trial-*")):
        record = hp_search.read_trial(trial_dir)
        if record is not None:
            trials.append(record)
    if not trials:
        raise FileNotFoundError(f"No {hp_search.TRIAL_FILE} files under {search_dir}/trial-*")
    for name in SEARCH_FIELDS:
        values = {repr(trial.get(name)) for trial in trials}
        if len(values) > 1:
            raise ValueError(f"Trials in {search_dir} differ in {name}: {sorted(values)}; they aren't one search")
    return trials


def as_flags(params: dict[str, Any]) -> str:
    """Format hyperparameters as finetune.py command-line flags."""
    return " ".join(f"--{name} {value}" for name, value in params.items())


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description="Rank the trials of a classifier hyperparameter search.")
    parser.add_argument("search_dir", type=Path, help="Output directory of the search (holds trial-*/)")
    parser.add_argument("--top", type=int, default=10, help="Number of trials to print")
    parser.add_argument("--log_level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Rank the trials, write summary.tsv, and print the best settings."""
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s [%(levelname)s] %(message)s")

    trials = load_trials(args.search_dir)
    complete = [trial for trial in trials if trial["status"] == "complete"]
    unfinished = [trial for trial in trials if trial["status"] != "complete"]
    for trial in unfinished:
        LOGGER.warning("Trial %d is %s", trial["trial"], trial["status"])
    if not complete:
        raise ValueError(f"No completed trials in {args.search_dir}")

    greater_is_better = bool(complete[0]["greater_is_better"])
    objective = complete[0]["objective"]
    complete.sort(key=lambda trial: trial["best_value"], reverse=greater_is_better)
    param_names = list(dict.fromkeys(name for trial in complete for name in trial["params"]))

    summary_path = args.search_dir / "summary.tsv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["rank", "trial", objective, "best_epoch", "train_runtime_s", *param_names])
        for rank, trial in enumerate(complete, start=1):
            writer.writerow([rank, trial["trial"], trial["best_value"], trial["best_epoch"], trial["train_runtime"],
                             *(trial["params"].get(name) for name in param_names)])
    LOGGER.info("Wrote %s", summary_path)

    direction = "higher" if greater_is_better else "lower"
    print(f"{len(complete)} completed trials, {len(unfinished)} unfinished; {objective} ({direction} is better)")
    print(f"{'rank':>4} {'trial':>5} {objective:>24} {'epoch':>5}  parameters")
    for rank, trial in enumerate(complete[: args.top], start=1):
        params = ", ".join(f"{name}={value}" for name, value in trial["params"].items())
        print(f"{rank:>4} {trial['trial']:>5} {trial['best_value']:>24.5f} {trial['best_epoch']:>5}  {params}")
    best = complete[0]
    print(f"\nBest trial {best['trial']} (best at epoch {best['best_epoch']}). Flags for the final training run:")
    print(as_flags(best["params"]))
    if best.get("max_train_samples"):
        print(f"(The trials trained on {best['max_train_samples']} documents; the final run uses all of them.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
