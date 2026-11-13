"""Predict with WebOrganizer's original classifiers on a dataset made for finetune.py.

Runs one of ``WebOrganizer/{Topic,Format}Classifier`` (URL and text) or
``WebOrganizer/{Topic,Format}Classifier-NoURL`` (text only), gte-base-en-v1.5 models with
24 labels, over a split of a dataset made by ``create_datasets.py`` or
``create_weborganizer_eval_dataset.py``, and writes:

- ``predictions.jsonl``: one line per document in dataset order, in the format of
  ``finetune.write_predictions`` (``id``, ``language``, ``source``, ``label``, ``label_probs``
  keyed by WebOrganizer's 24 label names), so the analysis notebooks read it like our own
  models' predictions;
- ``metrics.json``: agreement with the dataset's target distributions on the labels the model
  shares with the dataset, overall and per language (below).

**Input.** WebOrganizer's classifiers were trained on ``{url}\\n\\n{text}`` (the ``-NoURL``
ones on ``{text}``) with the URL as it is in the data, not percent-decoded, truncated to 8,192
tokens. The input is built with ``finetune.render_input`` with ``decode_url=False``. The
input type follows the model name unless ``--input`` says otherwise, and a mismatch is an
error.

**Metrics.** A document counts when its target top label (over all of the dataset's labels)
is one of the model's labels, so documents whose top label is *Other* or *Bad example* are
left out. The model's prediction is its top label. As in WebOrganizer's
``train_classifier.py``, accuracy is reported on all counted documents and on those whose
target top label has probability > 0.75, with per-label accuracy (recall) averaged over the
labels and its minimum. Labels without any such document are left out of the average and
minimum (WebOrganizer's code would give NaN); ``metrics.json`` says how many labels count.

**Loading.** The models' remote code (``modeling.py`` in each model repository) is run with
``trust_remote_code``. xformers attention and unpadding (on in their config.json) are turned
off, as in the model card's example. The rotary-embedding tables and the embeddings'
position IDs are not stored in the checkpoint but computed when the model is built.
transformers 5 builds models on the meta device and leaves such buffers uninitialised, so this
script rebuilds them after loading (and logs which had differed), and checks that the loaded
config kept config.json's RoPE settings. Reproducing WebOrganizer's Table 7 on its own
validation split checks the whole path.

Example:
    python scripts/model_evaluation/predict_weborganizer.py --model WebOrganizer/TopicClassifier \\
        --dataset_path results/datasets/topic/weborganizer-405B --split validation \\
        --output_dir results/models/topic/weborganizer-405B/weborganizer-url-validation
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import datasets
import numpy as np
import pyarrow.compute as pc
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# finetune.py's input rendering and prediction format, so both projects' models are fed and
# written the same way.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model_training"))
from finetune import render_input, write_predictions  # noqa: E402

LOGGER = logging.getLogger(__name__)

TEMPLATES = {"url": "{url}\n\n{text}", "text": "{text}"}
CONFIDENCE = 0.75


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command-line arguments.

    Args:
        argv: Arguments to parse instead of ``sys.argv``.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="e.g. WebOrganizer/TopicClassifier or WebOrganizer/TopicClassifier-NoURL")
    parser.add_argument("--dataset_path", type=Path, required=True, help="Dataset directory with labels.json")
    parser.add_argument("--split", required=True, help="Split to predict, e.g. test or validation")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--input", choices=sorted(TEMPLATES), default=None, help="url: URL and text; text: text only (default: from the model name)"
    )
    parser.add_argument("--max_length", type=int, default=8192, help="Tokens per document (WebOrganizer: 8192)")
    parser.add_argument(
        "--max_batch_tokens", type=int, default=65536, help="Padded tokens per batch (batch size x longest document)"
    )
    parser.add_argument("--chunk_size", type=int, default=1024, help="Documents tokenized at a time")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True, help="bf16 autocast (default on)")
    parser.add_argument("--max_documents", type=int, default=None, help="Only the first N documents (for tests)")
    parser.add_argument("--cache_dir", default=None, help="Hugging Face cache directory")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs")
    return parser.parse_args(argv)


def input_type(model: str, requested: str | None) -> str:
    """Return the input type for a model, checking it against the model name.

    Args:
        model: Model name or path.
        requested: ``url``, ``text``, or None to infer it.

    Returns:
        ``url`` or ``text``.

    Raises:
        ValueError: If the requested input contradicts the model name.
    """
    inferred = "text" if model.rstrip("/").lower().endswith("-nourl") else "url"
    if requested is not None and requested != inferred:
        raise ValueError(f"{model} takes {inferred!r} input by its name, but --input {requested} was given")
    return inferred


# Non-persistent buffers registered by WebOrganizer's modeling.py: rotary tables and the
# embeddings' position IDs. None is in the checkpoint; all are rebuilt after loading.
ROTARY_BUFFERS = ("inv_freq", "cos_cached", "sin_cached")
KNOWN_BUFFERS = frozenset(ROTARY_BUFFERS + ("position_ids",))
# Config fields that decide the architecture and positions; they must match config.json.
CHECKED_CONFIG_FIELDS = (
    "rope_scaling", "rope_theta", "max_position_embeddings", "position_embedding_type", "hidden_size",
    "num_attention_heads", "num_hidden_layers", "vocab_size", "type_vocab_size", "pad_token_id",
)


def check_config(model: torch.nn.Module, model_name: str, cache_dir: str | None) -> None:
    """Check that the loaded config kept the architecture fields of the repository's config.json.

    transformers 5 warns that it doesn't set ``rope_scaling`` from these older configs; a lost
    NTK scaling would silently change every rotary frequency.

    Args:
        model: The loaded model.
        model_name: Hugging Face model name or local path.
        cache_dir: Hugging Face cache directory.

    Raises:
        ValueError: If a field differs from config.json, or the rotary module doesn't match it.
    """
    local = Path(model_name) / "config.json"
    path = local if local.is_file() else Path(hf_hub_download(model_name, "config.json", cache_dir=cache_dir))
    raw = json.loads(path.read_text(encoding="utf-8"))
    for field_name in CHECKED_CONFIG_FIELDS:
        loaded = getattr(model.config, field_name, None)
        expected = raw.get(field_name)
        if field_name == "rope_scaling" and isinstance(loaded, dict) and isinstance(expected, dict):
            # transformers 5 may add its own keys (e.g. rope_type); config.json's must be kept.
            loaded = {key: loaded.get(key) for key in expected}
        if loaded != expected:
            raise ValueError(f"Loaded config has {field_name}={loaded!r}, but config.json has {raw.get(field_name)!r}")
    rotary = [module for module in model.modules() if hasattr(module, "inv_freq")]
    if len(rotary) != 1:
        raise ValueError(f"Expected one rotary-embedding module, found {len(rotary)}")
    scaling = raw.get("rope_scaling")
    module = rotary[0]
    if module.base != raw["rope_theta"] or (scaling is not None) != hasattr(module, "scaling_factor"):
        raise ValueError(f"Rotary module {type(module).__name__} (base {module.base}) doesn't match config.json")
    if scaling is not None and module.scaling_factor != scaling["factor"]:
        raise ValueError(f"Rotary scaling factor {module.scaling_factor}, config.json {scaling['factor']}")
    LOGGER.info("Config checked against config.json: rope_scaling %s, rope_theta %s, %s",
                scaling, raw["rope_theta"], type(module).__name__)


def rebuild_buffers(model: torch.nn.Module) -> tuple[int, int]:
    """Recompute the non-persistent buffers of a gte (``NewModel``) model on the CPU.

    transformers 5 builds models on the meta device and materialises them from the
    checkpoint, which leaves buffers that are only computed in ``__init__`` uninitialised:
    garbage ``position_ids`` crash the embedding lookup with an index error, and wrong rotary
    tables would silently change the outputs. Rotary modules (``RotaryEmbedding`` and its NTK
    subclass) are rebuilt with their own constructor arguments; ``position_ids`` becomes
    ``arange`` of its length. Any other non-persistent buffer is an error, so that a new
    kind of buffer can't be missed.

    Args:
        model: The loaded model, still on the CPU.

    Returns:
        The number of buffers rebuilt, and how many of them had differed.

    Raises:
        ValueError: If the model has an unknown non-persistent buffer, or no rotary module.
    """
    rebuilt = differed = 0
    rotary_modules = 0
    for name, module in model.named_modules():
        buffers = set(module._non_persistent_buffers_set)
        unknown = buffers - KNOWN_BUFFERS
        if unknown:
            raise ValueError(f"{name or 'model'} has non-persistent buffers {sorted(unknown)} that aren't rebuilt")
        fresh: dict[str, torch.Tensor] = {}
        if buffers & set(ROTARY_BUFFERS):
            kwargs: dict[str, Any] = {}
            if hasattr(module, "scaling_factor"):
                kwargs = {"scaling_factor": module.scaling_factor, "mixed_b": module.mixed_b}
            rebuilt_module = type(module)(module.dim, module.max_position_embeddings, module.base, device="cpu", **kwargs)
            fresh.update({buffer: getattr(rebuilt_module, buffer) for buffer in ROTARY_BUFFERS})
            module.max_seq_len_cached = rebuilt_module.max_seq_len_cached
            rotary_modules += 1
        if "position_ids" in buffers:
            fresh["position_ids"] = torch.arange(module.position_ids.shape[0])
        for buffer, new in fresh.items():
            old = getattr(module, buffer)
            if old.is_meta or old.shape != new.shape or not torch.equal(old.cpu().to(new.dtype), new):
                differed += 1
                LOGGER.warning("Buffer %s.%s differed from a fresh computation after loading; replaced",
                               name or "model", buffer)
            module.register_buffer(buffer, new, persistent=False)
            rebuilt += 1
    if rotary_modules == 0:
        raise ValueError("No rotary-embedding module found; is this a gte-base-en-v1.5 (NewModel) checkpoint?")
    return rebuilt, differed


def arrow_column(dataset: datasets.Dataset, column: str) -> Any:
    """Return one column as a single Arrow array, respecting any ``select`` on the dataset.

    One Arrow query instead of datasets 4's lazy ``Column``, whose per-element indexing is slow.
    """
    return dataset.select_columns([column]).with_format("arrow")[:][column].combine_chunks()


def batches(lengths: Sequence[int], max_batch_tokens: int) -> list[list[int]]:
    """Group positions (longest first) into batches whose padded size fits the token budget.

    Args:
        lengths: Token count per document.
        max_batch_tokens: Maximum of batch size times the longest document in the batch.

    Returns:
        Lists of positions into ``lengths``.
    """
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    groups: list[list[int]] = []
    current: list[int] = []
    for position in order:
        longest = lengths[current[0]] if current else lengths[position]
        if current and longest * (len(current) + 1) > max_batch_tokens:
            groups.append(current)
            current = []
        current.append(position)
    if current:
        groups.append(current)
    return groups


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    tokenizer: Any,
    dataset: datasets.Dataset,
    template: str,
    args: argparse.Namespace,
    device: torch.device,
) -> np.ndarray:
    """Compute the logits of every document, in dataset order.

    Documents are processed longest first (by characters) in chunks; each chunk is tokenized
    and split into batches by ``--max_batch_tokens``.

    Args:
        model: The classifier, on ``device``.
        tokenizer: Its tokenizer.
        dataset: The split to predict.
        template: Input template for ``finetune.render_input``.
        args: Parsed arguments.
        device: The device.

    Returns:
        Logits, one float32 row per document.

    Raises:
        ValueError: If a logit is not finite.
    """
    characters = pc.utf8_length(arrow_column(dataset, "text")).to_numpy(zero_copy_only=False)
    order = np.argsort(-characters, kind="stable")
    logits = np.full((len(dataset), model.config.num_labels), np.nan, dtype=np.float32)
    started = time.time()
    done = 0
    for start in range(0, len(order), args.chunk_size):
        indices = order[start : start + args.chunk_size]
        rows = dataset.select(indices)
        texts = [render_input(row, template, decode_url=False) for row in rows]
        encoded = tokenizer(texts, truncation=True, max_length=args.max_length)
        lengths = [len(ids) for ids in encoded["input_ids"]]
        for group in batches(lengths, args.max_batch_tokens):
            features = [{key: encoded[key][i] for key in encoded.keys()} for i in group]
            batch = tokenizer.pad(features, return_tensors="pt").to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.bf16):
                output = model(**batch).logits.float()
            logits[indices[group]] = output.cpu().numpy()
        done += len(indices)
        if start == 0 or done == len(order) or (start // args.chunk_size) % 20 == 0:
            rate = done / (time.time() - started)
            LOGGER.info("%d/%d documents, %.0f per second", done, len(order), rate)
    if not np.isfinite(logits).all():
        raise ValueError("Some logits are not finite (or some documents were not predicted)")
    return logits


def softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax."""
    shifted = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return shifted / shifted.sum(axis=-1, keepdims=True)


def shared_label_metrics(
    probs: np.ndarray, targets: np.ndarray, model_labels: Sequence[str], dataset_labels: Sequence[str]
) -> dict[str, Any]:
    """Agreement of the model's top label with the target top label, WebOrganizer style.

    Args:
        probs: Model probabilities in ``model_labels`` order.
        targets: Target distributions in ``dataset_labels`` order.
        model_labels: The model's label names.
        dataset_labels: The dataset's label names.

    Returns:
        Document counts, accuracy, and label-average and worst-label accuracy, on all counted
        documents and on those with target top-label probability > 0.75.

    Raises:
        ValueError: If a model label is not a dataset label.
    """
    missing = [name for name in model_labels if name not in dataset_labels]
    if missing:
        raise ValueError(f"Model labels {missing} are not in the dataset's labels")
    target_top = targets.argmax(axis=1)
    target_conf = targets.max(axis=1)
    to_model = {dataset_labels.index(name): i for i, name in enumerate(model_labels)}
    counted = np.isin(target_top, list(to_model))
    target_model = np.array([to_model.get(int(t), -1) for t in target_top])
    predicted = probs.argmax(axis=1)
    correct = predicted == target_model

    def subset(mask: np.ndarray) -> dict[str, Any]:
        recall = {
            model_labels[i]: float(correct[mask & (target_model == i)].mean())
            for i in range(len(model_labels))
            if (mask & (target_model == i)).any()
        }
        worst = min(recall, key=recall.get) if recall else None
        return {
            "documents": int(mask.sum()),
            "accuracy": float(correct[mask].mean()) if mask.any() else None,
            "labels_counted": len(recall),
            "label_average": float(np.mean(list(recall.values()))) if recall else None,
            "label_min": recall[worst] if worst else None,
            "label_min_name": worst,
            "per_label": recall,
        }

    confident = counted & (target_conf > CONFIDENCE)
    return {
        "documents": len(targets),
        "documents_counted": int(counted.sum()),
        "share_target_top_label_above_0.75": float(confident.sum() / counted.sum()) if counted.any() else None,
        "all": subset(counted),
        "target_top_label_above_0.75": subset(confident),
    }


def main(argv: Sequence[str] | None = None) -> None:
    """Predict and write the outputs.

    Args:
        argv: Arguments to parse instead of ``sys.argv``.

    Raises:
        FileExistsError: If outputs exist and ``--overwrite`` is not given.
        KeyError: If the dataset lacks the split or a column the input needs.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    kind = input_type(args.model, args.input)
    template = TEMPLATES[kind]
    predictions_path = args.output_dir / "predictions.jsonl"
    metrics_path = args.output_dir / "metrics.json"
    for path in (predictions_path, metrics_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to replace it")

    dataset_labels = json.loads((args.dataset_path / "labels.json").read_text(encoding="utf-8"))["label_names"]
    splits = datasets.load_from_disk(str(args.dataset_path))
    if args.split not in splits:
        raise KeyError(f"{args.dataset_path} has no split {args.split!r}; it has {list(splits)}")
    dataset = splits[args.split]
    if kind == "url" and "url" not in dataset.column_names:
        raise KeyError(f"{args.model} needs URLs, but {args.dataset_path} has no url column")
    if args.max_documents is not None:
        dataset = dataset.select(range(min(args.max_documents, len(dataset))))
    LOGGER.info("%s, %s input %r, on %s/%s: %d documents", args.model, kind, template, args.dataset_path, args.split,
                len(dataset))
    LOGGER.info("First input:\n%s", render_input(dataset[0], template, decode_url=False)[:500])

    if not torch.cuda.is_available():
        raise RuntimeError("No GPU available")
    device = torch.device("cuda")
    # trust_remote_code: the repository's config has an auto_map, so the tokenizer would
    # otherwise stop at an interactive prompt.
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir, trust_remote_code=True)
    model, loading_info = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        trust_remote_code=True,
        cache_dir=args.cache_dir,
        dtype=torch.float32,
        use_memory_efficient_attention=False,
        unpad_inputs=False,
        output_loading_info=True,
    )
    # A weight missing from the checkpoint would be randomly initialised; fail instead.
    if loading_info.get("missing_keys"):
        raise ValueError(f"Weights missing from the checkpoint: {sorted(loading_info['missing_keys'])}")
    if loading_info.get("unexpected_keys"):
        LOGGER.warning("Unused checkpoint weights: %s", sorted(loading_info["unexpected_keys"]))
    check_config(model, args.model, args.cache_dir)
    rebuilt, differed = rebuild_buffers(model)
    LOGGER.info("Rebuilt %d non-persistent buffers; %d had differed after loading", rebuilt, differed)
    model.to(device).eval()
    model_labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    LOGGER.info("%d model labels: %s", len(model_labels), model_labels)

    started = time.time()
    logits = predict(model, tokenizer, dataset, template, args, device)
    elapsed = time.time() - started

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(predictions_path, dataset, logits, model_labels)
    probs = softmax(logits)
    labels = arrow_column(dataset, "labels")
    targets = labels.flatten().to_numpy(zero_copy_only=False).astype(np.float64).reshape(len(dataset), -1)
    if targets.shape[1] != len(dataset_labels):
        raise ValueError(f"Targets have {targets.shape[1]} labels, labels.json {len(dataset_labels)}")
    metrics: dict[str, Any] = {
        "model": args.model,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "input": kind,
        "template": template,
        "dataset_path": str(args.dataset_path),
        "split": args.split,
        "max_length": args.max_length,
        "bf16": args.bf16,
        "seconds": elapsed,
        "documents_per_second": len(dataset) / elapsed,
        "overall": shared_label_metrics(probs, targets, model_labels, dataset_labels),
    }
    if "language" in dataset.column_names:
        languages = np.asarray(arrow_column(dataset, "language").to_pylist())
        metrics["per_language"] = {
            language: shared_label_metrics(probs[languages == language], targets[languages == language], model_labels,
                                           dataset_labels)
            for language in sorted(set(languages))
        }
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    overall = metrics["overall"]
    confident = overall["target_top_label_above_0.75"]
    LOGGER.info(
        "Counted %d of %d documents: accuracy %.4f; target top label > 0.75 (%d documents): accuracy %.4f, "
        "label average %.4f, worst label %.4f (%s), over %d labels",
        overall["documents_counted"], overall["documents"], overall["all"]["accuracy"], confident["documents"],
        confident["accuracy"], confident["label_average"], confident["label_min"], confident["label_min_name"],
        confident["labels_counted"],
    )
    LOGGER.info("Wrote %s and %s", predictions_path, metrics_path)


if __name__ == "__main__":
    main()
