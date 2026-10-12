"""Shared resources for the LLM labeling pipelines.

Used by ``generate_multilabel.py`` (structured multi-label output),
``generate_label_distributions.py`` (label probability distributions), and
``generate_label_clef.py`` (label distributions from the Clef decision model). Keeping
input loading, label resources, prompt shuffling, result storage, batching,
and configuration handling here ensures the pipelines process documents the
same way, which keeps their outputs comparable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sys
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, NoReturn, TypeVar

from pydantic import (  # type: ignore
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)
import yaml  # type: ignore

if TYPE_CHECKING:
    # Imported lazily in build_llm, so modules that don't run a local model (e.g. the
    # OpenAI API labeller) work without vLLM installed.
    from vllm import LLM  # type: ignore

LOGGER = logging.getLogger(__name__)


def normalize_language(language: str) -> str:
    """Normalize a language name to initial-capital form."""
    normalized = language.strip().capitalize()
    if not normalized:
        raise ValueError("language must not be empty")
    return normalized


def normalize_label_type(label_type: str) -> str:
    """Normalize and validate the supported labeling schemas."""
    normalized = label_type.strip().lower()
    if normalized not in {"topic", "format"}:
        raise ValueError("label_type must be either 'topic' or 'format'")
    return normalized


class InputDocument(BaseModel):
    """One input JSONL record."""

    model_config = ConfigDict(extra="allow", strict=True)

    doc_id: str = Field(min_length=1)
    text: str = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def normalize_doc_id(cls, value: Any) -> Any:
        """Accept exactly one of ``doc_id``, ``id``, or ``warc_record_id``.

        Records without any of these fields, such as the WebOrganizer
        annotation datasets, fall back to ``metadata["WARC-Record-ID"]``. A
        record with no ID at all still fails validation.
        """
        if not isinstance(value, dict):
            return value
        id_fields = ("doc_id", "id", "warc_record_id")
        provided = [field for field in id_fields if field in value]
        if len(provided) > 1:
            raise ValueError(
                "Exactly one document ID field is allowed; found: "
                f"{', '.join(provided)}"
            )
        if provided and provided[0] != "doc_id":
            normalized = dict(value)
            normalized["doc_id"] = normalized.pop(provided[0])
            return normalized
        metadata = value.get("metadata")
        if not provided and isinstance(metadata, dict) and "WARC-Record-ID" in metadata:
            normalized = dict(value)
            normalized["doc_id"] = metadata["WARC-Record-ID"]
            return normalized
        return value


def document_url(document: InputDocument) -> str | None:
    """Return the document URL from its extra fields, or ``None`` if absent."""
    url = document.model_extra.get("url", "") if document.model_extra else None
    return str(url) if url else None


class LabelDefinition(BaseModel):
    """Stable label definition loaded from a YAML file."""

    model_config = ConfigDict(extra="forbid", strict=True)

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    short_name: str = Field(min_length=1)
    definition: str = Field(min_length=1)


class LabelExample(BaseModel):
    """One label demonstration loaded from an examples YAML file."""

    model_config = ConfigDict(extra="forbid", strict=True)

    url: str
    text: str = Field(min_length=1)
    label: str = Field(min_length=1)
    explanation: str = Field(min_length=1)


def load_label_resources(
    labels_path: Path, examples_path: Path, label_type: str = "topic"
) -> tuple[list[LabelDefinition], list[LabelExample]]:
    """Load and validate labels and demonstrations for a labeling schema."""
    label_type = normalize_label_type(label_type)
    with labels_path.open("r", encoding="utf-8") as handle:
        raw_labels = yaml.safe_load(handle)
    with examples_path.open("r", encoding="utf-8") as handle:
        raw_examples = yaml.safe_load(handle)

    if not isinstance(raw_labels, list):
        raise ValueError(f"Expected a YAML list in {labels_path}")
    if not isinstance(raw_examples, dict) or not isinstance(
        raw_examples.get("examples"), list
    ):
        raise ValueError(f"Expected an 'examples' list in {examples_path}")

    labels = [LabelDefinition.model_validate(item) for item in raw_labels]
    normalized_examples = []
    for item in raw_examples["examples"]:
        if not isinstance(item, dict):
            raise ValueError(f"Example must be a mapping in {examples_path}")
        example = dict(item)
        if label_type == "format":
            if "choice" not in example or "label" in example:
                raise ValueError(
                    f"Format example must contain 'choice' in {examples_path}"
                )
            example["label"] = example.pop("choice")
        normalized_examples.append(LabelExample.model_validate(example))
    examples = normalized_examples
    stable_ids = [label.id for label in labels]
    if len(stable_ids) != len(set(stable_ids)):
        raise ValueError(f"Duplicate stable label IDs in {labels_path}")
    label_names = {label.name for label in labels}
    if len(label_names) != len(labels):
        raise ValueError(f"Duplicate label names in {labels_path}")
    short_names = {label.short_name for label in labels}
    if len(short_names) != len(labels):
        raise ValueError(f"Duplicate label short names in {labels_path}")
    unknown_example_labels = {example.label for example in examples} - label_names
    if unknown_example_labels:
        raise ValueError(
            f"Examples reference unknown label names: {sorted(unknown_example_labels)}"
        )
    return labels, examples


def prompt_seed(seed: int, doc_id: str) -> int:
    """Derive a deterministic per-document shuffle seed from the run seed."""
    digest = hashlib.sha256(f"{seed}:{doc_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def shuffle_prompt_order(
    num_labels: int, num_examples: int, seed: int
) -> tuple[list[int], list[int]]:
    """Return the per-document presentation order of labels and examples.

    Both pipelines use this function, so for the same run seed and document
    they present labels and examples in the same order.

    Args:
        num_labels: Number of label definitions.
        num_examples: Number of demonstrations.
        seed: Per-document seed from :func:`prompt_seed`.

    Returns:
        Label indices and example indices in presentation order.
    """
    label_order = list(range(num_labels))
    example_order = list(range(num_examples))
    randomizer = random.Random(seed)
    randomizer.shuffle(label_order)
    randomizer.shuffle(example_order)
    return label_order, example_order


PROMPT_ORDER_MODES = ("random", "rotated")


def rotated_prompt_order(
    num_labels: int, num_examples: int, seed: int, doc_id: str
) -> tuple[list[int], list[int]]:
    """Return one of ``num_labels`` rotations of a run-level base order.

    The run seed fixes one shuffled base order of labels and examples; each
    document gets a rotation of it chosen by a hash of its ID. Across the
    rotations every label appears at every letter position exactly once, so
    position bias averages out as with per-document shuffles. Because only
    ``num_labels`` distinct orders exist, the prompt text before the document
    repeats across many documents and vLLM's prefix cache can reuse it.

    Args:
        num_labels: Number of label definitions.
        num_examples: Number of demonstrations.
        seed: Run seed; different seeds give different base orders.
        doc_id: Document ID used to pick the rotation.

    Returns:
        Label indices and example indices in presentation order.
    """
    if num_labels < 1 or num_examples < 1:
        raise ValueError("num_labels and num_examples must be >= 1")
    base_labels, base_examples = shuffle_prompt_order(num_labels, num_examples, seed)
    rotation = prompt_seed(seed, doc_id) % num_labels
    example_shift = rotation % num_examples
    return (
        base_labels[rotation:] + base_labels[:rotation],
        base_examples[example_shift:] + base_examples[:example_shift],
    )


def prompt_order_for_document(
    mode: str, num_labels: int, num_examples: int, seed: int, doc_id: str
) -> tuple[list[int], list[int]]:
    """Return a document's label and example order for a prompt order mode.

    Args:
        mode: ``"random"`` for an independent shuffle per document, or
            ``"rotated"`` for :func:`rotated_prompt_order`.
        num_labels: Number of label definitions.
        num_examples: Number of demonstrations.
        seed: Run seed.
        doc_id: Document ID.

    Returns:
        Label indices and example indices in presentation order.

    Raises:
        ValueError: For an unknown mode.
    """
    if mode == "random":
        return shuffle_prompt_order(num_labels, num_examples, prompt_seed(seed, doc_id))
    if mode == "rotated":
        return rotated_prompt_order(num_labels, num_examples, seed, doc_id)
    raise ValueError(f"prompt order must be one of {PROMPT_ORDER_MODES}; got {mode!r}")


def stream_jsonl(path: Path) -> Iterator[tuple[int, InputDocument]]:
    """Yield validated documents without loading the input file into memory."""
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Empty JSONL line in {path}:{line_number}")
            try:
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise TypeError("record must be a JSON object")
                yield line_number, InputDocument.model_validate(raw)
            except (
                json.JSONDecodeError,
                TypeError,
                ValidationError,
                ValueError,
            ) as exc:
                raise ValueError(
                    f"Invalid input record in {path}:{line_number}: {exc}"
                ) from exc


def stream_huggingface_dataset(
    dataset_name: str,
    *,
    dataset_config: str | None = None,
    dataset_split: str = "train",
) -> Iterator[tuple[int, InputDocument]]:
    """Stream and validate documents from a HuggingFace dataset."""
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required when using --input-dataset"
        ) from exc

    dataset = load_dataset(
        dataset_name,
        name=dataset_config,
        split=dataset_split,
        streaming=True,
    )
    for row_number, row in enumerate(dataset, start=1):
        if not isinstance(row, dict):
            raise ValueError(
                f"Invalid HuggingFace dataset record at streamed row {row_number}: "
                "record must be a mapping"
            )
        try:
            yield row_number, InputDocument.model_validate(row)
        except ValidationError as exc:
            raise ValueError(
                f"Invalid HuggingFace dataset record at streamed row "
                f"{row_number}: {exc}"
            ) from exc


class SavedRecordBase(BaseModel):
    """Base class for output records; every record is keyed by document ID."""

    model_config = ConfigDict(extra="forbid", strict=True)

    id: str = Field(min_length=1)


R = TypeVar("R", bound=SavedRecordBase)


class JsonlResultStore(Generic[R]):
    """Append validated results and recover completed documents on restart."""

    def __init__(
        self,
        path: Path,
        record_model: type[R],
        validate_record: Callable[[R], None],
    ) -> None:
        """Open the store and load previously completed results.

        Args:
            path: Output JSONL path.
            record_model: Pydantic model of one output line.
            validate_record: Pipeline-specific check that raises ``ValueError``
                for a record that does not match the current label set.
        """
        self.path = path
        self.record_model = record_model
        self.validate_record = validate_record
        self.completed: set[str] = set()
        self._load_existing()

    def _load_existing(self) -> None:
        """Validate existing output lines and record their document IDs."""
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ValueError(
                        f"Empty output JSONL line in {self.path}:{line_number}"
                    )
                try:
                    record = self.record_model.model_validate_json(line)
                    self.validate_record(record)
                except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"Invalid saved result in {self.path}:{line_number}: {exc}"
                    ) from exc
                if record.id in self.completed:
                    raise ValueError(
                        f"Duplicate completed doc_id {record.id!r} in {self.path}"
                    )
                self.completed.add(record.id)
        LOGGER.info(
            "Resuming with %d validated results from %s", len(self.completed), self.path
        )

    def append(
        self, documents: Sequence[tuple[int, InputDocument]], records: Sequence[R]
    ) -> None:
        """Validate and durably append one batch of results.

        Args:
            documents: Input documents with their source line numbers.
            records: One result per document, in the same order.

        Raises:
            ValueError: If the inputs do not match or a document is duplicated.
        """
        if len(documents) != len(records):
            raise ValueError("The number of documents and results must match")
        if not documents:
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            for (_, document), record in zip(documents, records):
                if document.doc_id in self.completed:
                    raise ValueError(
                        f"Attempted to save duplicate doc_id {document.doc_id!r}"
                    )
                validated = self.record_model.model_validate(record)
                if validated.id != document.doc_id:
                    raise ValueError(
                        f"Result for {document.doc_id!r} contains mismatched id "
                        f"{validated.id!r}"
                    )
                self.validate_record(validated)
                handle.write(validated.model_dump_json() + "\n")
                self.completed.add(document.doc_id)
            handle.flush()
            os.fsync(handle.fileno())


def process_documents_in_batches(
    *,
    documents: Iterable[tuple[int, InputDocument]],
    store: JsonlResultStore[Any],
    process_batch: Callable[[Sequence[tuple[int, InputDocument]]], int],
    batch_size: int,
    max_documents: int | None,
) -> int:
    """Stream documents in batches, skipping completed ones, and resume safely.

    Args:
        documents: Validated input documents with their source line numbers.
        store: Result store that already contains completed documents.
        process_batch: Labels one batch, appends it to ``store``, and returns
            the number of documents saved.
        batch_size: Maximum number of documents per batch.
        max_documents: Stop once the store holds this many results.

    Returns:
        The number of newly processed documents.

    Raises:
        ValueError: On duplicate input IDs or output IDs missing from the input.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if max_documents is not None and max_documents < 1:
        raise ValueError("max_documents must be >= 1")
    seen_input_ids: set[str] = set()
    processed = 0
    stopped_early = False

    pending: list[tuple[int, InputDocument]] = []
    document_iterator = iter(documents)
    try:
        for line_number, document in document_iterator:
            if max_documents is not None and len(store.completed) >= max_documents:
                LOGGER.info(
                    "Reached max_documents=%d; stopping input processing",
                    max_documents,
                )
                stopped_early = True
                break
            if document.doc_id in seen_input_ids:
                raise ValueError(f"Duplicate doc_id {document.doc_id!r} in input")
            seen_input_ids.add(document.doc_id)
            if document.doc_id in store.completed:
                continue
            pending.append((line_number, document))
            remaining = (
                max_documents - len(store.completed)
                if max_documents is not None
                else batch_size
            )
            target_batch_size = min(batch_size, remaining)
            if len(pending) < target_batch_size:
                continue

            batch_processed = process_batch(pending)
            processed += batch_processed
            LOGGER.info(
                "Processed %d documents in batch; total documents processed: %d",
                batch_processed,
                len(store.completed),
            )
            pending = []
            if max_documents is not None and len(store.completed) >= max_documents:
                LOGGER.info(
                    "Reached max_documents=%d; stopping input processing",
                    max_documents,
                )
                stopped_early = True
                break
    finally:
        close = getattr(document_iterator, "close", None)
        if close is not None:
            close()

    if pending and (max_documents is None or len(store.completed) < max_documents):
        batch_processed = process_batch(pending)
        processed += batch_processed
        LOGGER.info(
            "Processed %d documents in batch; total documents processed: %d",
            batch_processed,
            len(store.completed),
        )
    unknown_completed = store.completed - seen_input_ids
    if unknown_completed and not stopped_early:
        raise ValueError(
            "Output contains doc_ids missing from input: "
            f"{sorted(unknown_completed)[:5]!r}"
        )
    LOGGER.info(
        "Completed %d new documents; output contains %d results",
        processed,
        len(store.completed),
    )
    return processed


def build_llm(
    *,
    model_name: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    dtype: str,
    max_model_len: int,
    seed: int,
    **engine_kwargs: Any,
) -> LLM:
    """Create the vLLM engine with the settings shared by both pipelines.

    Args:
        model_name: HuggingFace model name or path.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        gpu_memory_utilization: Fraction of GPU memory vLLM may use.
        dtype: Model dtype passed to vLLM.
        max_model_len: Maximum sequence length in tokens.
        seed: vLLM seed.
        **engine_kwargs: Pipeline-specific engine arguments.

    Returns:
        The loaded vLLM engine.
    """
    from vllm import LLM  # type: ignore

    return LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=dtype,
        max_model_len=max_model_len,
        performance_mode="throughput",
        seed=seed,
        language_model_only=True,
        **engine_kwargs,
    )


def thinking_chat_template_kwargs(
    model_name: str, thinking_mode: str
) -> dict[str, Any] | None:
    """Return the thinking effort kwargs for the chat template.

    Mistral uses reasoning_effort, while gemma and qwen models use enable_thinking.
    This might need to be modified if other models are used in the future.
    """
    if thinking_mode == "disabled":
        if "mistral" in model_name.lower():
            return {"reasoning_effort": "none"}
        else:
            return {"enable_thinking": False}
    if thinking_mode == "enabled":
        if "mistral" in model_name.lower():
            return {"reasoning_effort": "high"}
        else:
            return {"enable_thinking": True}
    if thinking_mode == "template-default":
        return None
    raise ValueError(f"Unknown thinking mode: {thinking_mode}")


# Marker appended to character-truncated documents, as in the WebOrganizer template.
TRUNCATION_MARKER = "... (truncated)"


def truncate_document_chars(text: str, max_chars: int) -> str:
    """Truncate text by characters and mark it, as the WebOrganizer template does.

    Used instead of token truncation when ``max_document_chars`` is set, so
    documents match the WebOrganizer annotations for direct comparison.

    Args:
        text: Document text.
        max_chars: Maximum number of characters kept.

    Returns:
        The text unchanged, or its first ``max_chars`` characters followed by
        ``TRUNCATION_MARKER``.
    """
    if len(text) > max_chars:
        return text[:max_chars] + TRUNCATION_MARKER
    return text


def truncate_document_text(tokenizer: Any, text: str, max_tokens: int) -> str:
    """Tokenize and truncate document text with the loaded model tokenizer."""
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_tokens,
    )
    input_ids = encoded["input_ids"]
    return tokenizer.decode(input_ids, skip_special_tokens=True)


def exit_without_finalization(exit_code: int) -> NoReturn:
    """Flush logs and exit immediately after a completed run.

    When a run stops at ``max_documents`` while streaming a HuggingFace
    dataset, the dataset's read-ahead threads are still downloading the
    current parquet file. During normal interpreter shutdown they fail on
    closed file descriptors and can abort the process with ``Fatal Python
    error: PyGILState_Release``, so SLURM reports exit code 134 although all
    results were saved. Skipping interpreter finalization avoids this. Nothing
    is lost: ``JsonlResultStore`` fsyncs every batch, and vLLM has already shut
    down when ``main()`` returns and releases the model client.

    Args:
        exit_code: Process exit code returned by ``main()``.
    """
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)


def log_model_throughput(
    *, input_tokens: int, generated_tokens: int, elapsed_seconds: float
) -> None:
    """Log token throughput for one vLLM call."""
    if elapsed_seconds <= 0:
        LOGGER.warning(
            "Elapsed time is non-positive (%.3f seconds); cannot compute throughput",
            elapsed_seconds,
        )
        return
    total_throughput = (input_tokens + generated_tokens) / elapsed_seconds
    input_throughput = input_tokens / elapsed_seconds
    output_throughput = generated_tokens / elapsed_seconds
    LOGGER.info(
        "Processed %d total tokens in %.3f seconds (%.2f tokens/sec (input: %.2f tokens/sec, output: %.2f tokens/sec))",
        input_tokens + generated_tokens,
        elapsed_seconds,
        total_throughput,
        input_throughput,
        output_throughput,
    )


# Nested YAML config keys and the argparse destinations they map to.
CONFIG_DESTINATIONS: dict[str, dict[str, str]] = {
    "paths": {
        "base_dir": "base_dir",
        "input_jsonl": "input_jsonl",
        "input_dataset": "input_dataset",
        "dataset_config": "dataset_config",
        "dataset_split": "dataset_split",
        "output": "output_path",
        "labels": "labels_path",
        "examples": "examples_path",
    },
    "model": {
        "name": "model_name",
        "tensor_parallel_size": "tensor_parallel_size",
        "gpu_memory_utilization": "gpu_memory_utilization",
        "dtype": "dtype",
        "max_model_len": "max_model_len",
        "seed": "seed",
        "thinking_mode": "thinking_mode",
    },
    "run": {
        "batch_size": "batch_size",
        "max_document_tokens": "max_document_tokens",
        "max_documents": "max_documents",
        "language": "language",
        "label_type": "label_type",
        "log_level": "log_level",
    },
}
CONFIG_PATH_DESTINATIONS = frozenset(
    {"input_jsonl", "output_path", "labels_path", "examples_path"}
)
# Path placeholder resolved after the CLI and config are merged, so one config
# can serve both label types, e.g. ``${base_dir}/scripts/${label_type}s.yaml``.
LABEL_TYPE_PLACEHOLDER = "${label_type}"
COMMON_DEFAULTS: dict[str, Any] = {
    "labels_path": None,
    "examples_path": None,
    "model_name": "Qwen/Qwen3.8-27B",
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.9,
    "dtype": "auto",
    "max_model_len": 32768,
    "seed": 0,
    "thinking_mode": "disabled",
    "batch_size": 32,
    "max_document_tokens": 8192,
    "max_documents": None,
    "language": "English",
    "label_type": "topic",
    "log_level": "INFO",
}


def build_common_arg_parser(
    *,
    description: str,
    thinking_modes: Sequence[str],
    thinking_mode_help: str | None = None,
) -> argparse.ArgumentParser:
    """Build the command-line arguments shared by both pipelines.

    Args:
        description: Parser description.
        thinking_modes: Allowed values of ``--thinking-mode``.
        thinking_mode_help: Optional help text for ``--thinking-mode``.

    Returns:
        A parser that the pipelines can extend with their own arguments.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, help="YAML configuration file")
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--input-jsonl", dest="input_jsonl", type=Path)
    input_group.add_argument("--input-dataset", dest="input_dataset")
    parser.add_argument("--dataset-config")
    parser.add_argument("--dataset-split", default=None)
    parser.add_argument("--output", dest="output_path", type=Path)
    parser.add_argument("--labels", dest="labels_path", type=Path)
    parser.add_argument("--examples", dest="examples_path", type=Path)
    parser.add_argument("--model-name")
    parser.add_argument("--tensor-parallel-size", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--dtype")
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--thinking-mode", choices=tuple(thinking_modes), help=thinking_mode_help
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-document-tokens", type=int)
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--language")
    parser.add_argument("--label-type", choices=("topic", "format"))
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
    )
    return parser


def load_cli_config(
    path: Path,
    *,
    extra_run_keys: Mapping[str, str] | None = None,
    unsupported_run_keys: Mapping[str, str] | None = None,
    extra_model_keys: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Load the supported nested YAML configuration structure.

    Config paths may use ``${base_dir}``, which is expanded here, and
    ``${label_type}``, which :func:`parse_cli_and_config` expands once the
    final label type is known.

    Args:
        path: YAML config path.
        extra_run_keys: Pipeline-specific ``run`` keys mapped to their argparse
            destinations.
        unsupported_run_keys: ``run`` keys to reject, mapped to the reason
            shown in the error message.
        extra_model_keys: Pipeline-specific ``model`` keys mapped to their
            argparse destinations.

    Returns:
        Flat mapping from argparse destination names to configured values.

    Raises:
        ValueError: If the config has an unexpected structure or keys.
    """
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Top-level config must be a mapping: {path}")

    destinations = {
        section: dict(keys) for section, keys in CONFIG_DESTINATIONS.items()
    }
    destinations["run"].update(extra_run_keys or {})
    destinations["model"].update(extra_model_keys or {})
    unknown = set(raw) - set(destinations)
    if unknown:
        raise ValueError(f"Unknown config sections: {sorted(unknown)}")
    flattened: dict[str, Any] = {}
    for section, keys in destinations.items():
        value = raw.get(section, {})
        if not isinstance(value, dict):
            raise ValueError(f"Config section {section!r} must be a mapping")
        if section == "run":
            for key, reason in (unsupported_run_keys or {}).items():
                if key in value:
                    raise ValueError(
                        f"run.{key} is not supported: {reason}; remove it from {path}"
                    )
        unknown_keys = set(value) - set(keys)
        if unknown_keys:
            raise ValueError(
                f"Unknown keys in config section {section!r}: {sorted(unknown_keys)}"
            )
        for key, item in value.items():
            flattened[keys[key]] = item
    configured_base_dir = flattened.pop("base_dir", None)
    if configured_base_dir is None:
        base_dir = path.parent
    else:
        base_dir = Path(configured_base_dir)
        if not base_dir.is_absolute():
            base_dir = path.parent / base_dir
    base_dir = base_dir.resolve()

    resolved: dict[str, Any] = {}
    for name, value in flattened.items():
        if name in CONFIG_PATH_DESTINATIONS:
            if not isinstance(value, str):
                raise ValueError(f"Config path {name!r} must be a string")
            expanded = value.replace("${base_dir}", str(base_dir))
            if "${" in expanded.replace(LABEL_TYPE_PLACEHOLDER, ""):
                raise ValueError(f"Unknown variable in config path {value!r}")
            candidate = Path(expanded)
            resolved[name] = (
                candidate if candidate.is_absolute() else base_dir / candidate
            )
        else:
            resolved[name] = value
    return resolved


def parse_cli_and_config(
    parser: argparse.ArgumentParser,
    argv: Sequence[str] | None,
    *,
    thinking_modes: Sequence[str],
    extra_defaults: Mapping[str, Any] | None = None,
    extra_run_keys: Mapping[str, str] | None = None,
    unsupported_run_keys: Mapping[str, str] | None = None,
    extra_model_keys: Mapping[str, str] | None = None,
) -> argparse.Namespace:
    """Parse YAML defaults first, then let explicit CLI values override them.

    Applies defaults and validates the arguments shared by both pipelines.
    Pipelines validate their own extra arguments afterwards.

    Args:
        parser: Parser from :func:`build_common_arg_parser`, possibly extended.
        argv: Command-line arguments, or ``None`` for ``sys.argv``.
        thinking_modes: Allowed thinking modes, also enforced for config values.
        extra_defaults: Defaults for pipeline-specific arguments.
        extra_run_keys: Pipeline-specific ``run`` config keys.
        unsupported_run_keys: ``run`` config keys to reject with a reason.
        extra_model_keys: Pipeline-specific ``model`` config keys.

    Returns:
        The merged and validated arguments.
    """
    preliminary = parser.parse_args(argv)
    if preliminary.config is not None:
        parser.set_defaults(
            **load_cli_config(
                preliminary.config,
                extra_run_keys=extra_run_keys,
                unsupported_run_keys=unsupported_run_keys,
                extra_model_keys=extra_model_keys,
            )
        )
    args = parser.parse_args(argv)
    defaults = {**COMMON_DEFAULTS, **(extra_defaults or {})}
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    try:
        args.label_type = normalize_label_type(args.label_type)
    except ValueError as exc:
        parser.error(str(exc))
    for name in CONFIG_PATH_DESTINATIONS:
        value = getattr(args, name)
        if value is not None and LABEL_TYPE_PLACEHOLDER in str(value):
            setattr(
                args,
                name,
                Path(str(value).replace(LABEL_TYPE_PLACEHOLDER, args.label_type)),
            )
    if args.input_jsonl is None and args.input_dataset is None:
        parser.error("exactly one of --input-jsonl or --input-dataset is required")
    if args.input_jsonl is not None and args.input_dataset is not None:
        parser.error("--input-jsonl and --input-dataset are mutually exclusive")
    if args.output_path is None:
        parser.error("--output is required or must be in config")
    if args.input_jsonl is not None and args.input_jsonl.is_dir():
        jsonl_inputs = sorted(args.input_jsonl.glob("*.jsonl"))
        if len(jsonl_inputs) != 1:
            parser.error(
                f"Input directory must contain exactly one JSONL file: "
                f"{args.input_jsonl} (found {len(jsonl_inputs)})"
            )
        args.input_jsonl = jsonl_inputs[0]
    if args.input_dataset is not None and not args.input_dataset.strip():
        parser.error("--input-dataset must be a non-empty dataset name")
    if args.dataset_config is not None and not args.dataset_config.strip():
        parser.error("--dataset-config must be a non-empty string")
    if args.dataset_split is None:
        args.dataset_split = "train"
    elif not args.dataset_split.strip():
        parser.error("--dataset-split must be a non-empty string")
    if not isinstance(args.model_name, str) or not args.model_name.strip():
        parser.error("--model-name must be a non-empty string")
    if not isinstance(args.dtype, str) or not args.dtype.strip():
        parser.error("--dtype must be a non-empty string")
    # Config values bypass argparse choices, so check thinking_mode explicitly.
    if args.thinking_mode not in thinking_modes:
        parser.error(
            f"thinking_mode must be one of {list(thinking_modes)}; "
            f"got {args.thinking_mode!r}"
        )
    if args.batch_size < 1 or args.max_document_tokens < 1:
        parser.error("--batch-size and --max-document-tokens must be >= 1")
    if args.max_documents is not None and args.max_documents < 1:
        parser.error("--max-documents must be >= 1")
    try:
        args.language = normalize_language(args.language)
    except ValueError as exc:
        parser.error(str(exc))
    if args.labels_path is None:
        args.labels_path = Path(f"scripts/data_processing/{args.label_type}s.yaml")
    if args.examples_path is None:
        args.examples_path = Path(
            f"scripts/data_processing/{args.label_type}_examples.yaml"
        )
    if args.max_model_len < 1 or args.tensor_parallel_size < 1:
        parser.error("--max-model-len and --tensor-parallel-size must be >= 1")
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    if args.seed < 0:
        parser.error("--seed must be >= 0")
    return args
