"""Fine-tune a classifier on soft topic or format labels from the label distribution runs.

Adapted from WebOrganizer's classifier training script (``weborganizer_classifier.py``):
the model is trained with cross-entropy against each document's label distribution,
not against a single hard label. The input is a dataset made by ``create_datasets.py``
(``train``/``validation``/``test`` splits with a ``labels`` probability vector per
document, and ``labels.json`` with the label names).

Changes from the WebOrganizer script:

* Targets mmBERT (``jhu-clsp/mmBERT-base``, a ModernBERT model) and transformers 5;
  works with any ``AutoModelForSequenceClassification`` architecture.
* The model is loaded in fp32; use ``--bf16`` for mixed precision. The loss is computed
  in fp32 on fp32 labels.
* Texts are tokenized once before training (cached next to the dataset), and training
  batches are grouped by length to cut padding at the 8,192-token context.
* Soft labels are validated (length, finite, non-negative, summing to 1); malformed
  labels raise an error instead of being trained on.
* Evaluation adds KL and Jensen-Shannon divergence and the Brier score to the accuracy
  metrics; labels without examples are reported in a warning instead of producing NaN.
* Test predictions are written to ``predictions.jsonl`` (document ID, language, label
  probabilities), with metrics per language.
* The label temperature applies to the training loss only; evaluation always compares
  with the original label distributions, so runs with different temperatures are
  comparable.
* Hyperparameter search: with ``--hp_search_space`` and ``--hp_trial``, the run is one
  trial of a seeded random search (see ``hp_search.py``), trained on the training
  split and scored on the validation split; ``summarize_hp_search.py`` ranks trials.

Usage (all ``TrainingArguments`` are available as command-line options)::

    python finetune.py --dataset_path results/datasets/topic/dataset1 \\
        --output_dir results/models/topic/dataset1 --do_train --do_eval --do_predict \\
        --bf16 --eval_strategy epoch --save_strategy epoch

    # Trial 7 of a search; writes <output_dir>/trial-007/trial.json
    python finetune.py --dataset_path ... --output_dir results/models/topic/dataset1/hp_search/search1 \\
        --hp_search_space configs/roihu/classifier/hp_space.yaml --hp_trial 7 --do_train \\
        --bf16 --eval_strategy epoch --save_strategy no \\
        --metric_for_best_model js_divergence_bits --greater_is_better false
"""

import dataclasses
import functools
import json
import logging
import math
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import datasets
import numpy as np
import torch
import transformers
from torch.utils.data import Sampler, SequentialSampler
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    PrinterCallback,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import is_flash_attn_2_available
from transformers.utils.versions import require_version

import hp_search  # sibling module: the script's directory is on sys.path

# ModernBERT, the architecture of mmBERT, was added in transformers 4.48.
require_version("transformers>=4.48", "mmBERT (ModernBERT) needs transformers>=4.48")

logger = logging.getLogger(__name__)

# The distribution pipeline accepts label probabilities that sum to 1 within this tolerance.
PROBABILITY_SUM_TOLERANCE = 1e-3
# Columns kept next to the token IDs, so predictions can be matched to documents.
ID_COLUMNS = ("id", "language")
# Metrics reported per language for the test set; the per-label accuracies are left out.
SUMMARY_METRICS = (
    "accuracy",
    "accuracy_label_average",
    "kl_divergence_bits",
    "js_divergence_bits",
    "brier_score",
)

# Metrics where lower values are better; a wrong direction would keep the worst checkpoint.
LOWER_IS_BETTER_MARKERS = ("loss", "divergence", "brier")

# Length-grouped batching below relies on the transformers 5 name of this option;
# fail here rather than silently train without it on an older version.
if "train_sampling_strategy" not in {f.name for f in dataclasses.fields(TrainingArguments)}:
    raise ImportError(
        f"transformers {transformers.__version__} has no TrainingArguments.train_sampling_strategy; "
        "this script is written for transformers 5 (Roihu's python-vllm/0.19.1 module has 5.8)"
    )


@dataclass
class ScriptArguments(TrainingArguments):
    """Training arguments plus the model, data, and soft-label options."""

    model_name: str = field(
        default="jhu-clsp/mmBERT-base",
        metadata={"help": "The model checkpoint for weights initialization."},
    )
    trust_remote_code: bool = field(default=False, metadata={"help": "Trust remote code"})
    cache_dir: str | None = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )

    dataset_path: str | None = field(
        default=None,
        metadata={"help": "Dataset directory written by create_datasets.py (required)"},
    )
    train_split: str = field(default="train", metadata={"help": "Split used for training"})
    validation_split: str = field(default="validation", metadata={"help": "Split used for evaluation"})
    test_split: str = field(default="test", metadata={"help": "Split used for --do_predict"})

    max_length: int = field(default=8192, metadata={"help": "Max length of input sequences"})

    label_field: list[str] = field(
        default_factory=lambda: ["labels"],
        metadata={"help": "Field name for labels. Supply back-off labels with multiple"},
    )

    template: str = field(default="{text}", metadata={"help": "Template for the text input"})

    label_temperature: float = field(
        default=1.0,
        metadata={
            "help": "Temperature applied to soft labels in the training loss (>1 flattens, <1 sharpens); "
            "evaluation uses the original labels"
        },
    )

    freeze_encoder: bool = field(default=False, metadata={"help": "Freeze the encoder"})

    classifier_pooling: str | None = field(
        default=None,
        metadata={
            "help": "Pooling of the classification head for models that support it (ModernBERT/mmBERT): "
            "'cls' or 'mean'. Default: the model's own setting ('cls' for mmBERT).",
            "choices": ["cls", "mean"],
        },
    )
    attn_implementation: str = field(
        default="auto",
        metadata={
            "help": "Attention implementation: 'auto' uses flash_attention_2 when the flash-attn package "
            "is installed, otherwise sdpa.",
            "choices": ["auto", "flash_attention_2", "sdpa", "eager"],
        },
    )
    preprocessing_num_workers: int | None = field(
        default=None, metadata={"help": "Processes used to tokenize the datasets"}
    )
    max_train_samples: int | None = field(
        default=None,
        metadata={"help": "Train on a seeded random subset of this many documents (e.g. for cheaper trials)"},
    )
    max_eval_samples: int | None = field(
        default=None,
        metadata={"help": "Evaluate on a seeded random subset of this many documents"},
    )

    hp_search_space: str | None = field(
        default=None,
        metadata={"help": "Search-space YAML file; makes this run one hyperparameter-search trial"},
    )
    hp_trial: int | None = field(
        default=None, metadata={"help": "Trial index (e.g. the SLURM array task ID); needs --hp_search_space"}
    )
    hp_search_seed: int = field(
        default=0, metadata={"help": "Seed of the search: the same seed and trial give the same hyperparameters"}
    )

    # Defaults changed from TrainingArguments. Batches of similar length waste far less
    # compute on padding; the sampler reads the precomputed "length" column, which the
    # Trainer would drop if remove_unused_columns were True (the collator picks the
    # columns it needs itself).
    train_sampling_strategy: str = field(
        default="group_by_length",
        metadata={
            "help": "Sampler for training: 'random', 'sequential', or 'group_by_length' (default here).",
            "choices": ["random", "sequential", "group_by_length"],
        },
    )
    remove_unused_columns: bool = field(
        default=False,
        metadata={"help": "Keep False: the length-grouped sampler needs the 'length' column."},
    )
    # True counts padding too, but batches are length-grouped. (HfArgumentParser parses
    # this option as a boolean, so the string modes can't be given as defaults or flags.)
    include_num_input_tokens_seen: str | bool = field(
        default=True,
        metadata={"help": "Count input tokens for the throughput log (default here: true)."},
    )

    def __post_init__(self) -> None:
        """Validate the script-specific arguments."""
        super().__post_init__()
        if not self.dataset_path:
            raise ValueError("--dataset_path is required")
        if not self.label_temperature > 0:
            raise ValueError("--label_temperature must be positive")
        if self.max_length < 1:
            raise ValueError("--max_length must be positive")
        if not self.label_field:
            raise ValueError("--label_field needs at least one field name")
        for name in ("max_train_samples", "max_eval_samples"):
            if getattr(self, name) is not None and getattr(self, name) < 1:
                raise ValueError(f"--{name} must be positive")
        if (self.hp_search_space is None) != (self.hp_trial is None):
            raise ValueError("--hp_search_space and --hp_trial must be given together")
        metric = self.metric_for_best_model or ""
        if self.greater_is_better and any(marker in metric for marker in LOWER_IS_BETTER_MARKERS):
            raise ValueError(f"--metric_for_best_model {metric} is lower-is-better; pass --greater_is_better false")


class LogCallback(TrainerCallback):
    """Log progress, remaining time, token throughput, and free GPU memory."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.start_time: float | None = None
        self.last_log_time: float | None = None
        self.is_training = False

        self.first_step_of_run = 0

    def on_step_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> None:
        """Start the clocks at the first step of this run (also after resuming)."""
        if state.is_world_process_zero and self.last_log_time is None:
            self.last_step = 0
            self.start_time = time.time()
            self.last_log_time = self.start_time
            self.first_step_of_run = state.global_step
            self.last_tokens_seen = state.num_input_tokens_seen

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        """Log the Trainer's metrics, with progress information during training."""
        if logs is None:
            return
        _ = logs.pop("total_flos", None)

        if state.is_world_process_zero:
            if self.is_training and self.start_time is not None and self.last_log_time is not None:
                current_time = time.time()
                time_diff = current_time - self.last_log_time

                self.last_log_time = current_time
                steps_completed = max(state.global_step, 1)

                steps_since_first = max(1, state.global_step - self.first_step_of_run)
                self.last_step = state.global_step

                tokens_seen_since_last = state.num_input_tokens_seen - self.last_tokens_seen
                self.last_tokens_seen = state.num_input_tokens_seen

                remaining_steps = state.max_steps - steps_completed
                pct_completed = (steps_completed / state.max_steps) * 100
                time_since_start = current_time - self.start_time
                remaining_time = (time_since_start / steps_since_first) * remaining_steps

                update = {
                    "completed": f"{pct_completed:.2f}% ({steps_completed:_} / {state.max_steps:_})",
                    "remaining time": self.format_duration(remaining_time),
                    "throughput": f"{tokens_seen_since_last / max(time_diff, 1e-9):.2f} tokens/s",
                }
                if torch.cuda.is_available():
                    gpu_mem_free, _ = torch.cuda.mem_get_info(device=args.device)
                    update["gpu_mem_free"] = f"{gpu_mem_free / 1024 / 1024:.0f}MB"

                logger.info(str({**logs, **update}))
            else:
                logger.info(str(logs))

    def on_step_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> None:
        """Mark that the following logs come from training."""
        if state.is_world_process_zero:
            self.is_training = True

    def on_prediction_step(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> None:
        """Mark that the following logs come from evaluation."""
        if state.is_world_process_zero:
            self.is_training = False

    @staticmethod
    def format_duration(seconds: float) -> str:
        """Format seconds as H:MM:SS."""
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours)}:{int(minutes):02}:{int(seconds):02}"


def get_labels(item: dict[str, Any], label_fields: Sequence[str]) -> Any:
    """Return the first label field of an item that is present and not None.

    Raises:
        KeyError: If the item has none of the label fields.
    """
    for label_field in label_fields:
        if item.get(label_field) is not None:
            return item[label_field]
    raise KeyError(f"Item has none of the label fields {list(label_fields)}; its fields are {sorted(item)}")


def soft_label_matrix(rows: Sequence[Any], num_labels: int) -> np.ndarray:
    """Validate soft labels and renormalize them.

    Args:
        rows: One probability vector per example (lists or arrays).
        num_labels: Expected length of every vector.

    Returns:
        A float32 array of shape ``(len(rows), num_labels)`` whose rows sum to 1.

    Raises:
        ValueError: If a vector has the wrong length, a value is negative or not
            finite, or a vector doesn't sum to 1.
    """
    lengths = {len(row) for row in rows}
    if lengths != {num_labels}:
        raise ValueError(f"Expected soft labels of length {num_labels}, got lengths {sorted(lengths)}")
    labels = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(labels).all():
        raise ValueError("Soft labels contain NaN or infinite values")
    if (labels < 0).any():
        raise ValueError("Soft labels contain negative values")
    sums = labels.sum(axis=-1)
    bad = np.flatnonzero(np.abs(sums - 1.0) > PROBABILITY_SUM_TOLERANCE)
    if bad.size:
        raise ValueError(f"{bad.size} soft label rows don't sum to 1 (e.g. row {bad[0]} sums to {sums[bad[0]]})")
    labels = labels / labels.sum(axis=-1, keepdims=True)
    return labels.astype(np.float32)


def temper_labels(labels: torch.Tensor, temperature: float) -> torch.Tensor:
    """Raise label distributions to the power ``1 / temperature`` and renormalize.

    Temperatures above 1 flatten the distributions, below 1 sharpen them.
    """
    labels = labels ** (1 / temperature)
    return labels / labels.sum(dim=-1, keepdim=True)


class DataCollator:
    """Pad pre-tokenized inputs and stack their validated soft labels."""

    def __init__(self, args: ScriptArguments, tokenizer: PreTrainedTokenizerBase, num_labels: int) -> None:
        self.args = args
        self.num_labels = num_labels
        if tokenizer.pad_token_id is None:
            raise ValueError("The tokenizer has no pad token")
        self.pad_token_id = tokenizer.pad_token_id
        self.padding_side = tokenizer.padding_side

    @torch.no_grad()
    def __call__(self, items: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Build a batch of ``input_ids``, ``attention_mask``, and fp32 ``labels``."""
        max_length = max(len(item["input_ids"]) for item in items)
        input_ids = torch.full((len(items), max_length), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(items), max_length), dtype=torch.long)
        for row, item in enumerate(items):
            ids = torch.as_tensor(item["input_ids"], dtype=torch.long)
            span = slice(0, len(ids)) if self.padding_side == "right" else slice(max_length - len(ids), max_length)
            input_ids[row, span] = ids
            attention_mask[row, span] = 1

        labels = soft_label_matrix([get_labels(item, self.args.label_field) for item in items], self.num_labels)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": torch.from_numpy(labels)}


def soft_label_metrics(logits: np.ndarray, targets: np.ndarray, label_names: Sequence[str]) -> dict[str, float]:
    """Compare predicted and target label distributions.

    Accuracy compares the predicted top label with the target's top label, overall and
    for examples whose target top label has probability above 0.5 and 0.75 (the
    confident subsets). Per-label accuracy is the recall of each target top label;
    labels that are never the target's top label are reported in a warning and left
    out of the label average and minimum. KL divergence (target to prediction) and
    Jensen-Shannon divergence are in bits; the Brier score is the summed squared
    difference between the two distributions, averaged over examples (0 is perfect,
    at most 2).

    Args:
        logits: Model logits, shape ``(examples, labels)``.
        targets: Target probabilities, same shape, rows summing to 1.
        label_names: Label names in column order, used in the metric names.

    Raises:
        ValueError: If the shapes don't match or there are no examples.
    """
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if logits.ndim != 2 or logits.shape != targets.shape or logits.shape[1] != len(label_names):
        raise ValueError(f"Logits {logits.shape} and targets {targets.shape} don't match {len(label_names)} labels")
    if not len(logits):
        raise ValueError("No examples to evaluate")

    log_probs = logits - logits.max(axis=-1, keepdims=True)
    log_probs -= np.log(np.exp(log_probs).sum(axis=-1, keepdims=True))
    probs = np.exp(log_probs)

    predictions = np.argmax(logits, axis=-1)
    top_label = np.argmax(targets, axis=-1)
    correct = predictions == top_label
    confidence = np.max(targets, axis=-1)

    metrics: dict[str, float] = {}
    subsets = {"": np.ones(len(targets), dtype=bool), "_conf50": confidence > 0.5, "_conf75": confidence > 0.75}
    for suffix, mask in subsets.items():
        if suffix:
            metrics[f"proportion{suffix}"] = float(mask.mean())
        if not mask.any():
            logger.warning("No examples in subset 'accuracy%s' (0 of %d); its accuracies are not reported", suffix, len(mask))
            continue
        metrics[f"accuracy{suffix}"] = float(correct[mask].mean())

        per_label, missing = [], []
        for index, name in enumerate(label_names):
            support = mask & (top_label == index)
            if support.any():
                metrics[f"accuracy{suffix}__{name}"] = float((predictions[support] == index).mean())
                per_label.append(metrics[f"accuracy{suffix}__{name}"])
            else:
                missing.append(name)
        if missing:
            logger.warning(
                "%d of %d labels are never the top label in subset 'accuracy%s' (%d examples): %s; "
                "their accuracy is not reported and the label average and minimum use the other %d labels",
                len(missing), len(label_names), suffix, int(mask.sum()), ", ".join(missing), len(per_label),
            )
        if per_label:
            metrics[f"accuracy_label_average{suffix}"] = float(np.mean(per_label))
            metrics[f"accuracy_label_min{suffix}"] = float(np.min(per_label))

    # Terms with zero probability contribute 0 (0 * log 0 = 0); np.where avoids log(0).
    log_targets = np.log(np.where(targets > 0, targets, 1.0))
    mixture = (targets + probs) / 2
    log_mixture = np.log(np.where(mixture > 0, mixture, 1.0))
    kl = (targets * (log_targets - log_probs)).sum(axis=-1)
    js = 0.5 * (targets * (log_targets - log_mixture)).sum(axis=-1) + 0.5 * (probs * (log_probs - log_mixture)).sum(
        axis=-1
    )
    metrics["kl_divergence_bits"] = float(kl.mean() / math.log(2))
    metrics["js_divergence_bits"] = float(js.mean() / math.log(2))
    metrics["brier_score"] = float(((probs - targets) ** 2).sum(axis=-1).mean())
    return metrics


class SoftClassificationTrainer(Trainer):
    """Trainer with a soft-label cross-entropy loss and soft-label metrics."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # The loss is a batch mean that ignores num_items_in_batch, which the Trainer
        # computes as labels != -100 and so would count every soft-label entry. With
        # this flag the Trainer scales the loss for gradient accumulation itself.
        self.model_accepts_loss_kwargs = False

        try:
            self.remove_callback(PrinterCallback)
            self.add_callback(LogCallback)
            # self.add_callback(SIGUSR1Callback(self))
        except ValueError:
            logger.warning("Couldn't remove PrinterCallback")

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """Cross-entropy between the label distributions and the predictions, in fp32.

        The label temperature applies only in training (``model.training``), so
        evaluation losses and metrics always compare with the original labels.
        ``kwargs`` absorbs ``num_items_in_batch``, which newer transformers versions
        pass; see ``__init__`` for why it isn't used.
        """
        inputs = dict(inputs)
        labels = inputs.pop("labels").float()
        if model.training and self.args.label_temperature != 1.0:
            labels = temper_labels(labels, self.args.label_temperature)

        outputs = model(**inputs)

        logits = outputs.logits
        loss = torch.nn.functional.cross_entropy(logits.float(), labels)

        return (loss, outputs) if return_outputs else loss

    def _get_eval_sampler(self, eval_dataset: datasets.Dataset) -> Sampler | None:
        """Evaluate in dataset order, never length-grouped.

        With ``train_sampling_strategy="group_by_length"``, transformers 5 also shuffles
        evaluation batches by length. The evaluation sets are sorted by length instead
        (see ``main``), which saves as much padding and keeps predictions in the order
        of the dataset's ``id`` and ``language`` columns.
        """
        if self.args.world_size <= 1:
            return SequentialSampler(eval_dataset)
        return None  # the distributed default, which also keeps dataset order


def tokenize_split(
    dataset: datasets.Dataset,
    tokenizer: PreTrainedTokenizerBase,
    args: ScriptArguments,
    description: str,
) -> datasets.Dataset:
    """Tokenize a split once, keeping the token IDs, their count, the labels, and the IDs.

    The result is cached next to the dataset by ``datasets``, so later runs with the
    same tokenizer and settings reuse it.

    Raises:
        ValueError: If the split has none of the label fields.
    """
    label_columns = [name for name in args.label_field if name in dataset.column_names]
    if not label_columns:
        raise ValueError(f"Split {description!r} has none of the label fields {args.label_field}")

    # Local copies: the map cache key hashes what the function uses, and the full
    # arguments differ between runs (output directory, logging, ...).
    template, max_length = args.template, args.max_length

    def tokenize(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        rows = [dict(zip(batch, values)) for values in zip(*batch.values())]
        encoded = tokenizer(
            [template.format(**row) for row in rows],
            truncation=True,
            max_length=max_length,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return {"input_ids": encoded["input_ids"], "length": [len(ids) for ids in encoded["input_ids"]]}

    keep = set(label_columns) | set(ID_COLUMNS)
    with args.main_process_first(desc=f"tokenizing {description}"):
        return dataset.map(
            tokenize,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=[name for name in dataset.column_names if name not in keep],
            desc=f"Tokenizing {description}",
        )


def resolve_attn_implementation(requested: str) -> str:
    """Return the attention implementation to load the model with."""
    if requested != "auto":
        return requested
    return "flash_attention_2" if is_flash_attn_2_available() else "sdpa"


def write_predictions(
    path: Path,
    dataset: datasets.Dataset,
    logits: np.ndarray,
    label_names: Sequence[str],
) -> None:
    """Write one JSON line per test document with its predicted label distribution.

    Args:
        path: Output JSONL path.
        dataset: The dataset the predictions were made on, in prediction order.
        logits: Model logits, one row per document.
        label_names: Label names in column order.

    Raises:
        ValueError: If the number of predictions doesn't match the dataset.
    """
    if len(logits) != len(dataset):
        raise ValueError(f"{len(logits)} predictions for {len(dataset)} documents")
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probs = np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True)
    ids = dataset["id"] if "id" in dataset.column_names else [None] * len(dataset)
    languages = dataset["language"] if "language" in dataset.column_names else [None] * len(dataset)
    with path.open("w", encoding="utf-8") as handle:
        for doc_id, language, row in zip(ids, languages, probs):
            record = {
                "id": doc_id,
                "language": language,
                "label": label_names[int(np.argmax(row))],
                "label_probs": {name: float(value) for name, value in zip(label_names, row)},
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def subsample(dataset: datasets.Dataset, limit: int | None, seed: int, description: str) -> datasets.Dataset:
    """Return a seeded random subset of at most ``limit`` rows (the same subset for every run)."""
    if limit is None:
        return dataset
    if limit >= len(dataset):
        logger.info(f"{description}: limit {limit} >= {len(dataset)} documents; using all")
        return dataset
    logger.info(f"{description}: using a random subset of {limit} of {len(dataset)} documents (seed {seed})")
    return dataset.shuffle(seed=seed).select(range(limit))


def check_trial_arguments(args: ScriptArguments) -> None:
    """Check that a hyperparameter-search trial is set up to be scored fairly and cheaply.

    Raises:
        ValueError: If the trial would touch the test set, save checkpoints, or have
            nothing to be scored on.
    """
    if not args.do_train:
        raise ValueError("A hyperparameter trial needs --do_train")
    if args.do_predict:
        raise ValueError("A hyperparameter trial must not use the test split; drop --do_predict")
    if args.eval_strategy == "no":
        raise ValueError("A hyperparameter trial is scored on its evaluations; set --eval_strategy")
    if args.save_strategy != "no" or args.load_best_model_at_end:
        raise ValueError(
            "Trials keep no checkpoints (the best evaluation is read from the log); "
            "use --save_strategy no without --load_best_model_at_end"
        )
    if not args.metric_for_best_model:
        raise ValueError("Set --metric_for_best_model (and --greater_is_better) as the trial objective")


def main() -> None:
    """Train, evaluate, and predict as selected by ``--do_train``, ``--do_eval``, ``--do_predict``."""
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    parser = HfArgumentParser(ScriptArguments)
    args, = parser.parse_args_into_dataclasses()
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {args.local_rank}, device: {args.device}, n_gpu: {args.n_gpu}, "
        + f"distributed training: {bool(args.local_rank != -1)}, bf16: {args.bf16}, fp16: {args.fp16}"
    )
    trial_record: dict[str, Any] | None = None
    if args.hp_search_space is not None:
        check_trial_arguments(args)
        space_path = Path(args.hp_search_space)
        params = hp_search.sample_trial(hp_search.load_search_space(space_path), args.hp_search_seed, args.hp_trial)
        trial_dir = Path(args.output_dir) / hp_search.trial_dir_name(args.hp_trial)
        previous = hp_search.read_trial(trial_dir)
        if previous is not None and previous["status"] == "complete":
            if previous["params"] != params:
                raise ValueError(
                    f"{trial_dir} holds a completed trial with other parameters {previous['params']}; the search "
                    "space or seed changed, so use a new output directory"
                )
            logger.info(f"Trial {args.hp_trial} is already complete in {trial_dir}; nothing to do")
            return
        hp_search.apply_trial_params(args, params)
        args.output_dir = str(trial_dir)
        trial_record = {
            "trial": args.hp_trial,
            "status": "running",
            "params": params,
            "search_space": str(space_path),
            "search_seed": args.hp_search_seed,
            "dataset_path": args.dataset_path,
            "model_name": args.model_name,
            "max_train_samples": args.max_train_samples,
            "max_eval_samples": args.max_eval_samples,
            "metric": args.metric_for_best_model,
            "greater_is_better": args.greater_is_better,
        }
        if args.process_index == 0:
            hp_search.write_trial(trial_dir, trial_record)
        logger.info(f"Hyperparameter trial {args.hp_trial}: {params}")

    logger.info(f"Training/evaluation parameters {args}")
    # Detecting last checkpoint.
    last_checkpoint = None
    output_dir = Path(args.output_dir)
    if output_dir.is_dir():
        last_checkpoint = get_last_checkpoint(str(output_dir))
        if last_checkpoint is not None and args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` to train from scratch."
            )

    dataset_path = Path(args.dataset_path)
    raw_datasets = datasets.load_from_disk(str(dataset_path))  # memory-mapped Arrow files
    label_info = json.loads((dataset_path / "labels.json").read_text(encoding="utf-8"))
    label_names: list[str] = label_info["label_names"]
    logger.info(f"{len(label_names)} {label_info['label_type']} labels: {label_names}")

    # Set seed before initializing model.
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError(f"The tokenizer of {args.model_name} has neither a pad nor an EOS token")
        logger.warning("The tokenizer has no pad token; using the EOS token for padding")
        tokenizer.pad_token = tokenizer.eos_token

    run_eval = args.do_eval or args.eval_strategy != "no"
    splits = {
        name: split
        for name, split, needed in (
            ("train", args.train_split, args.do_train),
            ("validation", args.validation_split, run_eval),
            ("test", args.test_split, args.do_predict),
        )
        if needed
    }
    for split in splits.values():
        if split not in raw_datasets:
            raise KeyError(f"{dataset_path} has no split {split!r}; it has {list(raw_datasets)}")
    limits = {"train": args.max_train_samples, "validation": args.max_eval_samples, "test": None}
    tokenized = {
        name: tokenize_split(subsample(raw_datasets[split], limits[name], args.seed, split), tokenizer, args, split)
        for name, split in splits.items()
    }
    for name in ("validation", "test"):
        if name in tokenized:
            # Longest first: batches of similar length, and an out-of-memory error shows up at once.
            tokenized[name] = tokenized[name].sort("length", reverse=True)
    for name, dataset in tokenized.items():
        lengths = np.asarray(dataset["length"])
        logger.info(
            f"{name}: {len(dataset)} documents, median {np.median(lengths):.0f} tokens, "
            f"{(lengths >= args.max_length).mean():.1%} truncated at {args.max_length}"
        )

    config = AutoConfig.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
        num_labels=len(label_names),
        id2label=dict(enumerate(label_names)),
        label2id={name: index for index, name in enumerate(label_names)},
    )
    if args.classifier_pooling is not None:
        if not hasattr(config, "classifier_pooling"):
            raise ValueError(f"{args.model_name} ({config.model_type}) has no classifier_pooling option")
        config.classifier_pooling = args.classifier_pooling
    if getattr(config, "classifier_pooling", None) is not None:
        logger.info(f"Classifier pooling: {config.classifier_pooling}")
    if getattr(config, "pad_token_id", None) is None:
        config.pad_token_id = tokenizer.pad_token_id

    attn_implementation = resolve_attn_implementation(args.attn_implementation)
    logger.info(f"Attention implementation: {attn_implementation}")
    # Weights stay in fp32; --bf16 / --fp16 run the forward pass in mixed precision.
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        config=config,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
        dtype=torch.float32,
        attn_implementation=attn_implementation,
    )

    if args.freeze_encoder:
        # base_model is the encoder without the classification head, for any architecture.
        if model.base_model is model:
            raise ValueError(f"Can't separate the encoder from the head of {type(model).__name__}")
        for param in model.base_model.parameters():
            param.requires_grad = False

    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    logger.info(f"Model: {model}")
    logger.info(f"Trainable parameters: {trainable:_} of {total:_}")

    # Initialize our Trainer
    trainer = SoftClassificationTrainer(
        model=model,
        args=args,
        train_dataset=tokenized.get("train"),
        eval_dataset=tokenized.get("validation"),
        processing_class=tokenizer,
        data_collator=DataCollator(args, tokenizer, len(label_names)),
        compute_metrics=lambda eval_pred: soft_label_metrics(eval_pred.predictions, eval_pred.label_ids, label_names),
    )

    if trainer.is_fsdp_enabled:
        from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy

        def layer_policy_fn(module: torch.nn.Module) -> bool:
            return "layer" in module.__class__.__name__.lower()

        auto_wrap_policy = functools.partial(lambda_auto_wrap_policy,
                                             lambda_fn=layer_policy_fn)
        trainer.accelerator.state.fsdp_plugin.auto_wrap_policy = auto_wrap_policy

    # Training
    if args.do_train:
        checkpoint = None
        if args.resume_from_checkpoint is not None:
            checkpoint = args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        try:
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
        except BaseException:
            if trial_record is not None and trainer.is_world_process_zero():
                hp_search.write_trial(Path(args.output_dir), {**trial_record, "status": "failed"})
            raise
        if trial_record is None:
            trainer.save_model()  # a trial's model isn't kept; the best settings are retrained

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        if trial_record is not None:
            key, best, evaluations = hp_search.best_evaluation(
                trainer.state.log_history, args.metric_for_best_model, args.greater_is_better
            )
            trial_record.update(
                status="complete",
                objective=key,
                best_value=best[key],
                best_epoch=best.get("epoch"),
                best_step=best.get("step"),
                train_documents=len(tokenized["train"]),
                train_runtime=metrics.get("train_runtime"),
                evaluations=evaluations,
            )
            if trainer.is_world_process_zero():
                hp_search.write_trial(Path(args.output_dir), trial_record)
            logger.info(f"Trial {args.hp_trial}: best {key} = {best[key]:.5f} at epoch {best.get('epoch')}")

    # Evaluation
    if args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if args.do_predict:
        logger.info("*** Predict ***")
        output = trainer.predict(tokenized["test"])
        metrics = dict(output.metrics)

        test_dataset = tokenized["test"]
        if "language" in test_dataset.column_names:
            languages = np.asarray(test_dataset["language"])
            for language in sorted(set(languages)):
                mask = languages == language
                language_metrics = soft_label_metrics(output.predictions[mask], output.label_ids[mask], label_names)
                metrics.update(
                    {f"test_{language}_{name}": language_metrics[name] for name in SUMMARY_METRICS
                     if name in language_metrics}
                )

        if trainer.is_world_process_zero():
            write_predictions(output_dir / "predictions.jsonl", test_dataset, output.predictions, label_names)

        trainer.log_metrics("pred", metrics)
        trainer.save_metrics("pred", metrics)


if __name__ == "__main__":
    main()
