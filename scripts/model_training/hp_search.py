"""Seeded random hyperparameter search for ``finetune.py``, one trial per SLURM array task.

A search is a search-space file, a search seed, and an output directory. Trial ``i``
draws its hyperparameters from a random generator seeded with the search seed, the
trial index, and the parameter name, so every array task computes its own trial
independently: no shared database, no coordination between jobs, and rerunning a
trial reproduces it exactly. Each trial trains on the training split, evaluates on
the validation split, and writes ``trial-<i>/trial.json`` with its parameters and its
best validation score; ``summarize_hp_search.py`` ranks the trials.

Random search is used instead of a sequential optimizer (e.g. Optuna's TPE) because it
needs no extra packages (the Roihu python-vllm module has no Optuna) and its trials
run in parallel; with a handful of parameters, a few dozen random trials find settings
close to what model-based optimizers find (Bergstra & Bengio, 2012).

Only the standard library and PyYAML are used, so this module can be tested without
torch.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml  # type: ignore

DISTRIBUTIONS = ("uniform", "log_uniform", "int_uniform", "choice")
# Training arguments a trial may set. They are applied before the Trainer is built and
# need no derived settings recomputed; anything else raises instead of being ignored.
SEARCHABLE_ARGUMENTS = frozenset(
    {
        "learning_rate",
        "weight_decay",
        "warmup_steps",
        "num_train_epochs",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "max_grad_norm",
        "adam_beta1",
        "adam_beta2",
        "label_temperature",
        "classifier_pooling",
    }
)
TRIAL_FILE = "trial.json"


@dataclass(frozen=True)
class SearchDimension:
    """One hyperparameter and the distribution it is drawn from.

    Attributes:
        name: Training argument the value is assigned to.
        distribution: ``uniform``, ``log_uniform``, ``int_uniform`` (inclusive
            bounds), or ``choice``.
        low: Lower bound for the continuous and integer distributions.
        high: Upper bound for the continuous and integer distributions.
        values: Options for ``choice``.
    """

    name: str
    distribution: str
    low: float | None = None
    high: float | None = None
    values: tuple[Any, ...] = ()

    def sample(self, rng: random.Random) -> Any:
        """Draw one value."""
        if self.distribution == "choice":
            return rng.choice(self.values)
        assert self.low is not None and self.high is not None
        if self.distribution == "uniform":
            return round(rng.uniform(self.low, self.high), 6)
        if self.distribution == "log_uniform":
            value = math.exp(rng.uniform(math.log(self.low), math.log(self.high)))
            return float(f"{value:.3g}")  # three significant digits keep the values readable
        return rng.randint(int(self.low), int(self.high))


def load_search_space(path: Path) -> list[SearchDimension]:
    """Read and validate a search-space YAML file.

    The file has a ``parameters`` mapping from training-argument name to a spec, e.g.
    ``learning_rate: {distribution: log_uniform, low: 1.0e-5, high: 1.0e-4}`` or
    ``classifier_pooling: {distribution: choice, values: [cls, mean]}``.

    Raises:
        ValueError: If the file is malformed, names an argument that can't be
            searched, or has invalid bounds.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("parameters"), dict) or not raw["parameters"]:
        raise ValueError(f"{path}: expected a non-empty 'parameters' mapping")
    space = []
    for name, spec in raw["parameters"].items():
        where = f"{path}: parameter {name!r}"
        if name not in SEARCHABLE_ARGUMENTS:
            raise ValueError(f"{where} can't be searched; searchable: {sorted(SEARCHABLE_ARGUMENTS)}")
        if not isinstance(spec, dict) or spec.get("distribution") not in DISTRIBUTIONS:
            raise ValueError(f"{where} needs a 'distribution', one of {DISTRIBUTIONS}")
        distribution = spec["distribution"]
        if distribution == "choice":
            values = spec.get("values")
            if not isinstance(values, list) or not values or set(spec) != {"distribution", "values"}:
                raise ValueError(f"{where}: 'choice' needs exactly a non-empty 'values' list")
            space.append(SearchDimension(name, distribution, values=tuple(values)))
            continue
        if set(spec) != {"distribution", "low", "high"}:
            raise ValueError(f"{where}: {distribution!r} needs exactly 'low' and 'high'")
        low, high = spec["low"], spec["high"]
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (low, high)) or not low < high:
            raise ValueError(f"{where}: needs numbers with low < high, got {low!r}, {high!r}")
        if distribution == "log_uniform" and low <= 0:
            raise ValueError(f"{where}: 'log_uniform' needs low > 0")
        if distribution == "int_uniform" and not (float(low).is_integer() and float(high).is_integer()):
            raise ValueError(f"{where}: 'int_uniform' needs integer bounds")
        space.append(SearchDimension(name, distribution, float(low), float(high)))
    return space


def sample_trial(space: list[SearchDimension], search_seed: int, trial: int) -> dict[str, Any]:
    """Return the hyperparameters of one trial.

    Each parameter has its own generator, seeded with the search seed, the trial
    index, and the parameter name, so adding or removing a parameter doesn't change
    the values the others get.

    Raises:
        ValueError: If the trial index is negative.
    """
    if trial < 0:
        raise ValueError(f"Trial index must be non-negative, got {trial}")
    return {dim.name: dim.sample(random.Random(f"{search_seed}:{trial}:{dim.name}")) for dim in space}


def apply_trial_params(args: Any, params: dict[str, Any]) -> None:
    """Set a trial's hyperparameters on a training-argument dataclass.

    Values are cast to the type of the current value (e.g. 3 epochs to ``3.0``) and
    checked against the field's ``choices``. A fraction is never truncated to an
    integer: ``warmup_steps`` defaults to the integer 0 but is declared ``float``, and
    a value like 0.05 must stay 0.05.

    Raises:
        ValueError: If a parameter can't be searched, a value isn't allowed, or a
            fraction is given for an integer-only argument.
    """
    fields = {f.name: f for f in dataclasses.fields(args)}
    for name, value in params.items():
        if name not in SEARCHABLE_ARGUMENTS or name not in fields:
            raise ValueError(f"Hyperparameter {name!r} can't be set by a trial")
        current = getattr(args, name)
        if isinstance(current, int) and not isinstance(current, bool) and isinstance(value, float) \
                and not value.is_integer():
            if "float" not in str(fields[name].type):
                raise ValueError(f"Hyperparameter {name!r} takes integers, got {value!r}")
        elif current is not None and not isinstance(current, bool) and not isinstance(value, str):
            value = type(current)(value)
        choices = fields[name].metadata.get("choices")
        if choices is not None and value not in choices:
            raise ValueError(f"Hyperparameter {name!r} = {value!r} is not one of {list(choices)}")
        setattr(args, name, value)


def best_evaluation(
    log_history: list[dict[str, Any]], metric: str, greater_is_better: bool
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    """Find the best evaluation of a training run.

    Args:
        log_history: ``TrainerState.log_history``.
        metric: Metric name, with or without the ``eval_`` prefix.
        greater_is_better: Direction of the metric.

    Returns:
        The prefixed metric name, the best evaluation log entry, and all evaluation
        log entries in order.

    Raises:
        ValueError: If no evaluation logged the metric, or a logged value is NaN.
    """
    key = metric if metric.startswith("eval_") else f"eval_{metric}"
    evaluations = [entry for entry in log_history if key in entry]
    if not evaluations:
        raise ValueError(f"No evaluation logged {key!r}; is eval_strategy set?")
    if any(math.isnan(entry[key]) for entry in evaluations):
        raise ValueError(f"An evaluation logged NaN for {key!r}")
    pick = max if greater_is_better else min
    return key, pick(evaluations, key=lambda entry: entry[key]), evaluations


def read_trial(trial_dir: Path) -> dict[str, Any] | None:
    """Return a trial's record, or None if the trial hasn't started."""
    path = trial_dir / TRIAL_FILE
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_trial(trial_dir: Path, record: dict[str, Any]) -> None:
    """Write a trial's record atomically, so a crash never leaves a partial file."""
    trial_dir.mkdir(parents=True, exist_ok=True)
    record = {**record, "updated": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    temporary = trial_dir / f".{TRIAL_FILE}.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(trial_dir / TRIAL_FILE)


def trial_dir_name(trial: int) -> str:
    """Directory name of a trial inside the search directory."""
    return f"trial-{trial:03d}"
