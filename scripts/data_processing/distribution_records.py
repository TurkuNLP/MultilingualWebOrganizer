"""Output records of the label distribution pipelines.

Shared by ``generate_label_distributions.py`` (chat models scored through vLLM)
and ``generate_label_clef.py`` (the Clef decision model), so both write the same
JSONL format and the evaluation and training scripts read either without
changes. This module has no model dependencies (no vLLM, no torch).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from pathlib import Path

from pydantic import Field, model_validator  # type: ignore

try:
    from .labeling_common import (
        JsonlResultStore,
        LabelDefinition,
        PROMPT_ORDER_MODES,
        SavedRecordBase,
    )
except ImportError:
    from labeling_common import (
        JsonlResultStore,
        LabelDefinition,
        PROMPT_ORDER_MODES,
        SavedRecordBase,
    )

# Allowed deviation of a probability distribution's total mass from one.
PROBABILITY_SUM_TOLERANCE = 1e-3
# Ordering scheme of models that can't reorder the labels per document: every
# document sees the labels in the same order and no demonstrations. It is not a
# ``--prompt-order`` choice of the vLLM pipeline, only a value stored in records.
FIXED_PROMPT_ORDER = "fixed"
RECORD_PROMPT_ORDERS = (*PROMPT_ORDER_MODES, FIXED_PROMPT_ORDER)


class SavedDistributionRecord(SavedRecordBase):
    """Flat JSONL output record with the document and its label distribution.

    Attributes:
        id: Document ID.
        text: Original, untruncated document text.
        label: Short name of the most probable label.
        label_prob: Probability of ``label``.
        label_probs: Probability per label short name, in label-file order.
        label_logprobs: Log-probability per label short name, normalized over
            the label set.
        label_order: Label short names in the order they were presented to the
            model, i.e. as answers ``A``, ``B``, ``C``, ...
        example_order: Indices into the examples file in the order the
            demonstrations were presented to the model.
        prompt_order: Ordering scheme, ``"random"`` or ``"rotated"``, or
            ``"fixed"`` for models that present the labels in the same order
            for every document (see ``FIXED_PROMPT_ORDER``). Records written
            before this field existed used ``"random"``.
    """

    text: str = ""
    label: str = Field(min_length=1)
    label_prob: float
    label_probs: dict[str, float] = Field(min_length=1)
    label_logprobs: dict[str, float] = Field(min_length=1)
    label_order: list[str] = Field(min_length=1)
    example_order: list[int]
    prompt_order: str = "random"

    @model_validator(mode="after")
    def validate_distribution(self) -> "SavedDistributionRecord":
        """Check that the stored distribution is complete and consistent."""
        label_names = set(self.label_probs)
        if set(self.label_logprobs) != label_names:
            raise ValueError("label_probs and label_logprobs must have the same keys")
        if len(self.label_order) != len(set(self.label_order)) or set(
            self.label_order
        ) != label_names:
            raise ValueError("label_order must be a permutation of the label names")
        if sorted(self.example_order) != list(range(len(self.example_order))):
            raise ValueError("example_order must be a permutation of example indices")
        if self.prompt_order not in RECORD_PROMPT_ORDERS:
            raise ValueError(f"Unknown prompt_order {self.prompt_order!r}")
        for name, probability in self.label_probs.items():
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError(f"Invalid probability {probability!r} for {name!r}")
            logprob = self.label_logprobs[name]
            if not math.isfinite(logprob):
                raise ValueError(f"Invalid log-probability {logprob!r} for {name!r}")
            if not math.isclose(
                math.exp(logprob), probability, rel_tol=1e-9, abs_tol=1e-12
            ):
                raise ValueError(
                    f"Probability {probability!r} for {name!r} does not match "
                    f"log-probability {logprob!r}"
                )
        total = math.fsum(self.label_probs.values())
        if abs(total - 1.0) > PROBABILITY_SUM_TOLERANCE:
            raise ValueError(f"label_probs must sum to 1; got {total!r}")
        if self.label not in self.label_probs:
            raise ValueError(f"label {self.label!r} is missing from label_probs")
        if self.label_prob != self.label_probs[self.label]:
            raise ValueError("label_prob must equal label_probs[label]")
        if self.label_prob != max(self.label_probs.values()):
            raise ValueError("label must be the most probable label")
        return self


def logsumexp(values: Sequence[float]) -> float:
    """Return ``log(sum(exp(values)))`` computed in a numerically stable way."""
    max_value = max(values)
    return max_value + math.log(math.fsum(math.exp(v - max_value) for v in values))


def compute_label_distribution(answer_logprobs: Sequence[float]) -> list[float]:
    """Normalize per-answer log-probabilities over the answer set.

    This is the answer scoring from the original WebOrganizer::

        scores = choice_loss - np.max(choice_loss)
        scores = scores - np.log(np.exp(scores).sum())
        probs = np.exp(scores)

    In the template, ``choice_loss`` is the mean log-probability of each
    answer's tokens. Every answer here is a single token, so that mean is the
    token's log-probability itself.

    Args:
        answer_logprobs: Log-probability of each answer.

    Returns:
        Log-probabilities normalized to sum to one in probability space, in the
        same order as ``answer_logprobs``.

    Raises:
        ValueError: If the input is empty or contains non-finite values.
    """
    if not answer_logprobs:
        raise ValueError("answer_logprobs must not be empty")
    if not all(math.isfinite(value) for value in answer_logprobs):
        raise ValueError(f"answer_logprobs must be finite; got {answer_logprobs!r}")
    max_score = max(answer_logprobs)
    scores = [score - max_score for score in answer_logprobs]
    log_normalizer = math.log(math.fsum(math.exp(score) for score in scores))
    return [score - log_normalizer for score in scores]


def distribution_record_validator(
    label_names: set[str], prompt_order: str
) -> Callable[[SavedDistributionRecord], None]:
    """Build a store check that records match the current labels and ordering.

    Args:
        label_names: Label short names of the current labeling schema.
        prompt_order: Ordering scheme of the current run.

    Returns:
        A function raising ``ValueError`` for records with other labels or
        another ordering scheme, so one output file never mixes the two.
    """

    def validate(record: SavedDistributionRecord) -> None:
        record_labels = set(record.label_probs)
        if record_labels != label_names:
            raise ValueError(
                f"Result for {record.id!r} has labels {sorted(record_labels)!r}; "
                f"expected {sorted(label_names)!r}"
            )
        if record.prompt_order != prompt_order:
            raise ValueError(
                f"Result for {record.id!r} used prompt_order="
                f"{record.prompt_order!r}, but this run uses {prompt_order!r}; "
                "use the matching --prompt-order or a new output path"
            )

    return validate


def open_result_store(
    output_path: Path, labels: Sequence[LabelDefinition], prompt_order: str
) -> JsonlResultStore[SavedDistributionRecord]:
    """Open the distribution result store for the given labels and ordering."""
    return JsonlResultStore(
        output_path,
        SavedDistributionRecord,
        distribution_record_validator(
            {label.short_name for label in labels}, prompt_order
        ),
    )
