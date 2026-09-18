from __future__ import annotations

import argparse
import json
import hashlib
import logging
import os
import random
import time
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, TypeVar

from pydantic import (  # type: ignore
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)  # type: ignore
import yaml  # type: ignore
from vllm import LLM, SamplingParams  # type: ignore
from vllm.sampling_params import StructuredOutputsParams  # type: ignore
from transformers.tokenization_utils_base import BatchEncoding  # type: ignore

try:
    from .prompts import (
        get_english_topic_classification_prompt,
        get_multilingual_topic_classification_prompt,
    )
except ImportError:
    from prompts import (
        get_english_topic_classification_prompt,
        get_multilingual_topic_classification_prompt,
    )

M = TypeVar("M", bound=BaseModel)
LOGGER = logging.getLogger(__name__)


def normalize_language(language: str) -> str:
    """Normalize a language name to initial-capital form."""
    normalized = language.strip().capitalize()
    if not normalized:
        raise ValueError("language must not be empty")
    return normalized


class InputDocument(BaseModel):
    """One input JSONL record."""

    model_config = ConfigDict(extra="allow", strict=True)

    doc_id: str = Field(min_length=1)
    text: str = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def normalize_doc_id(cls, value: Any) -> Any:
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
        return value


class LabelSelection(BaseModel):
    """Ordered structured response for a single document."""

    model_config = ConfigDict(extra="forbid", strict=True)

    rationale: str | None = None
    labels: list[str] = Field(min_length=1)

    @field_validator("labels")
    @classmethod
    def labels_must_be_unique(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("labels must not contain duplicates")
        return value


class SavedLabelRecord(BaseModel):
    """Durable output envelope used for resume and post-run validation."""

    model_config = ConfigDict(extra="forbid", strict=True)

    doc_id: str = Field(min_length=1)
    input_line: int = Field(ge=1)
    result: dict[str, Any]


class TopicLabel(BaseModel):
    """Stable label definition loaded from the topic YAML file."""

    model_config = ConfigDict(extra="forbid", strict=True)

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    definition: str = Field(min_length=1)


class TopicExample(BaseModel):
    """One demonstration loaded from the examples YAML file."""

    model_config = ConfigDict(extra="forbid", strict=True)

    url: str
    text: str = Field(min_length=1)
    label: str = Field(min_length=1)
    explanation: str = Field(min_length=1)


def load_topic_resources(
    labels_path: Path, examples_path: Path
) -> tuple[list[TopicLabel], list[TopicExample]]:
    """Load and validate stable labels and demonstrations from YAML files."""
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

    labels = [TopicLabel.model_validate(item) for item in raw_labels]
    examples = [TopicExample.model_validate(item) for item in raw_examples["examples"]]
    stable_ids = [label.id for label in labels]
    if len(stable_ids) != len(set(stable_ids)):
        raise ValueError(f"Duplicate stable label IDs in {labels_path}")
    label_names = {label.name for label in labels}
    if len(label_names) != len(labels):
        raise ValueError(f"Duplicate label names in {labels_path}")
    unknown_example_labels = {example.label for example in examples} - label_names
    if unknown_example_labels:
        raise ValueError(
            f"Examples reference unknown label names: {sorted(unknown_example_labels)}"
        )
    return labels, examples


def _prompt_seed(seed: int, doc_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{doc_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _format_prompt_resources(
    labels: Sequence[TopicLabel], examples: Sequence[TopicExample], seed: int
) -> tuple[str, str, dict[str, str]]:
    """Shuffle resources and return prompt text plus temporary-to-stable IDs."""
    shuffled_labels = list(labels)
    shuffled_examples = list(examples)
    randomizer = random.Random(seed)
    randomizer.shuffle(shuffled_labels)
    randomizer.shuffle(shuffled_examples)

    temporary_to_stable = {
        f"T{index:02d}": label.id
        for index, label in enumerate(shuffled_labels, start=1)
    }
    stable_to_temporary = {
        stable_id: temporary_id
        for temporary_id, stable_id in temporary_to_stable.items()
    }
    name_to_stable = {label.name: label.id for label in labels}
    label_lines = [
        f"{temporary_id}: {label.name} - {label.definition}"
        for temporary_id, label in zip(temporary_to_stable, shuffled_labels)
    ]
    example_blocks = []
    for example in shuffled_examples:
        try:
            stable_id = name_to_stable[example.label]
            temporary_id = stable_to_temporary[stable_id]
        except KeyError as exc:
            raise ValueError(
                f"Example references unknown topic label name {example.label!r}"
            ) from exc
        example_blocks.append(
            f"URL: {example.url}\n"
            f"Content: {example.text}\n"
            f"Label: {temporary_id}\n"
            f"Explanation: {example.explanation}"
        )
    return "\n".join(label_lines), "\n\n".join(example_blocks), temporary_to_stable


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


class JsonlResultStore:
    """Append validated results and recover completed documents on restart."""

    def __init__(self, path: Path, result_model: type[M], labels: set[str]) -> None:
        self.path = path
        self.result_model = result_model
        self.labels = labels
        self.completed: set[str] = set()
        self._load_existing()

    def _load_existing(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ValueError(
                        f"Empty output JSONL line in {self.path}:{line_number}"
                    )
                try:
                    record = SavedLabelRecord.model_validate_json(line)
                    result = self.result_model.model_validate(record.result)
                except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"Invalid saved result in {self.path}:{line_number}: {exc}"
                    ) from exc
                self._validate_result(record.doc_id, result)
                if record.doc_id in self.completed:
                    raise ValueError(
                        f"Duplicate completed doc_id {record.doc_id!r} in {self.path}"
                    )
                self.completed.add(record.doc_id)
        LOGGER.info(
            "Resuming with %d validated results from %s", len(self.completed), self.path
        )

    def _validate_result(self, doc_id: str, result: M) -> None:
        labels = getattr(result, "labels", None)
        if labels is None:
            label = getattr(result, "label", None)
            labels = [label]
        unknown_labels = set(labels) - self.labels
        if unknown_labels:
            raise ValueError(
                f"Result for {doc_id!r} selected unknown labels: "
                f"{sorted(unknown_labels)!r}"
            )
        if len(labels) != len(set(labels)):
            raise ValueError(f"Result for {doc_id!r} contains duplicate labels")

    def append(
        self, documents: Sequence[tuple[int, InputDocument]], results: Sequence[M]
    ) -> None:
        if len(documents) != len(results):
            raise ValueError("The number of documents and results must match")
        if not documents:
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            for (line_number, document), result in zip(documents, results):
                if document.doc_id in self.completed:
                    raise ValueError(
                        f"Attempted to save duplicate doc_id {document.doc_id!r}"
                    )
                validated = self.result_model.model_validate(result)
                self._validate_result(document.doc_id, validated)
                record = SavedLabelRecord(
                    doc_id=document.doc_id,
                    input_line=line_number,
                    result=validated.model_dump(mode="json"),
                )
                handle.write(record.model_dump_json() + "\n")
                self.completed.add(document.doc_id)
            handle.flush()
            os.fsync(handle.fileno())


def build_label_schema(result_model: type[M], labels: Sequence[str]) -> dict[str, Any]:
    """Build a JSON schema constraining ordered ``labels`` to known IDs."""
    unique_labels = list(dict.fromkeys(labels))
    if not unique_labels or any(
        not isinstance(label, str) or not label for label in unique_labels
    ):
        raise ValueError("labels must contain at least one non-empty string")
    schema = result_model.model_json_schema()
    properties = schema.get("properties", {})
    if "labels" not in properties:
        raise ValueError(f"{result_model.__name__} must define a 'labels' field")
    properties["labels"] = {
        "type": "array",
        "items": {"type": "string", "enum": unique_labels},
        "minItems": 1,
    }
    return schema


class StructuredModelOutputError(RuntimeError):
    """The model returned a completion that violated the structured-output contract."""


class ModelClient:
    def __init__(
        self,
        *,
        model_name: str,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        dtype: str,
        max_model_len: int,
        seed: int,
        thinking_mode: str,
    ) -> None:
        self.model_name = model_name
        self.seed = seed
        self.thinking_mode = thinking_mode

        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            max_model_len=max_model_len,
            performance_mode="throughput",
            seed=seed,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.max_model_len = int(self.llm.model_config.max_model_len)

    @property
    def chat_template_kwargs(self) -> dict[str, Any] | None:
        if self.thinking_mode == "disabled":
            return {"enable_thinking": False}
        if self.thinking_mode == "enabled":
            return {"enable_thinking": True}
        if self.thinking_mode == "template-default":
            return None
        raise ValueError(f"Unknown thinking mode: {self.thinking_mode}")

    def _sampling_params(
        self,
        json_schema: dict,
        *,
        max_tokens: int,
    ) -> SamplingParams:
        structured = StructuredOutputsParams(json=json_schema)
        return SamplingParams(
            temperature=0.0,
            seed=self.seed,
            max_tokens=max_tokens,
            structured_outputs=structured,
        )

    def _chat_token_count(self, messages: list[dict[str, str]]) -> int:
        kwargs = self.chat_template_kwargs or {}
        tokenized = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            **kwargs,
        )

        # Handle BatchEncoding objects from transformers
        if BatchEncoding is not None and isinstance(tokenized, BatchEncoding):
            if "input_ids" not in tokenized:
                raise TypeError(
                    "Tokenizer chat template returned BatchEncoding without input_ids"
                )
            tokenized = tokenized["input_ids"]
        elif isinstance(tokenized, dict):
            if "input_ids" not in tokenized:
                raise TypeError(
                    "Tokenizer chat template returned a dict without input_ids"
                )
            tokenized = tokenized["input_ids"]

        # Convert tensor-like objects to list (e.g., PyTorch tensors, numpy arrays)
        if hasattr(tokenized, "tolist"):
            tokenized = tokenized.tolist()

        if not isinstance(tokenized, (list, tuple)):
            raise TypeError(
                "Tokenizer apply_chat_template() returned unsupported type "
                f"{type(tokenized).__name__}"
            )
        return len(tokenized)

    def _log_model_throughput(
        self, *, input_tokens: int, generated_tokens: int, elapsed_seconds: float
    ) -> None:
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

    def _run_structured_chat(
        self,
        input_batch: list[list[dict[str, str]]],
        json_schemas: Sequence[dict[str, Any]],
        result_model: type[M],
        max_tokens: int,
    ) -> list[M]:
        if len(input_batch) != len(json_schemas):
            raise ValueError("The number of prompts and schemas must match")

        sampling_params = [
            self._sampling_params(schema, max_tokens=max_tokens)
            for schema in json_schemas
        ]

        kwargs: dict[str, Any] = {}
        if self.chat_template_kwargs is not None:
            kwargs["chat_template_kwargs"] = self.chat_template_kwargs

        generate_start_time = time.time()

        # Generate outputs using the LLM's chat method
        outputs = self.llm.chat(
            messages=input_batch,
            sampling_params=sampling_params,
            use_tqdm=False,
            **kwargs,
        )

        generate_end_time = time.time()
        if len(outputs) != len(input_batch):
            raise RuntimeError(
                f"vLLM returned {len(outputs)} outputs for {len(input_batch)} prompts"
            )

        for index, request_output in enumerate(outputs):
            if len(request_output.outputs) != 1:
                raise RuntimeError(
                    f"expected exactly one completion, "
                    f"got {len(request_output.outputs)}"
                )

        input_tokens = sum(len(output.prompt_token_ids) for output in outputs)
        generated_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        self._log_model_throughput(
            input_tokens=input_tokens,
            generated_tokens=generated_tokens,
            elapsed_seconds=generate_end_time - generate_start_time,
        )

        parsed: list[M] = []
        for request_output in outputs:
            completion = request_output.outputs[0]

            if completion.finish_reason != "stop":
                raise StructuredModelOutputError(
                    f"generation ended with finish_reason="
                    f"{completion.finish_reason!r}; output may be truncated.\n"
                    f"Configured max_tokens={max_tokens}\n"
                    f"Generated tokens: {len(completion.token_ids)}\n"
                    f"Generated text:\n{completion.text!r}"
                )

            generated_text = completion.text
            if not generated_text:
                raise StructuredModelOutputError(f"model returned empty output")

            try:
                parsed.append(result_model.model_validate_json(generated_text))
            except (
                ValidationError,
                json.JSONDecodeError,
                TypeError,
                ValueError,
            ) as exc:
                preview = generated_text[:2000]
                raise StructuredModelOutputError(
                    f"model output failed "
                    f"{result_model.__name__} validation: {exc}. "
                    f"Output preview: {preview!r}"
                ) from exc

        return parsed

    def _run_documents(
        self,
        *,
        documents: Iterable[tuple[int, InputDocument]],
        output_path: Path,
        labels: Sequence[TopicLabel],
        examples: Sequence[TopicExample],
        result_model: type[M] = LabelSelection,
        batch_size: int = 32,
        max_tokens: int = 256,
        max_document_tokens: int = 8192,
        max_documents: int | None = None,
        language: str = "English",
    ) -> int:
        """Process validated documents, save results, and resume safely."""
        stable_ids = [label.id for label in labels]
        if not stable_ids or len(stable_ids) != len(set(stable_ids)):
            raise ValueError("labels must contain unique stable IDs")
        if not examples:
            raise ValueError("examples must contain at least one demonstration")
        if max_document_tokens < 1:
            raise ValueError("max_document_tokens must be >= 1")
        if max_documents is not None and max_documents < 1:
            raise ValueError("max_documents must be >= 1")
        language = normalize_language(language)
        store = JsonlResultStore(output_path, result_model, set(stable_ids))
        seen_input_ids: set[str] = set()
        processed = 0
        stopped_early = False

        pending: list[tuple[int, InputDocument]] = []
        for line_number, document in documents:
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

            batch_processed = self._process_batch(
                pending,
                store,
                labels,
                examples,
                result_model,
                max_tokens,
                max_document_tokens,
                language,
            )
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

        if pending and (max_documents is None or len(store.completed) < max_documents):
            batch_processed = self._process_batch(
                pending,
                store,
                labels,
                examples,
                result_model,
                max_tokens,
                max_document_tokens,
                language,
            )
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

    def run_jsonl(
        self,
        *,
        input_path: Path,
        output_path: Path,
        labels: Sequence[TopicLabel],
        examples: Sequence[TopicExample],
        result_model: type[M] = LabelSelection,
        batch_size: int = 32,
        max_tokens: int = 256,
        max_document_tokens: int = 8192,
        max_documents: int | None = None,
        language: str = "English",
    ) -> int:
        """Stream a local JSONL file, label it, save results, and resume."""
        if input_path.resolve() == output_path.resolve():
            raise ValueError("input_path and output_path must be different files")
        return self._run_documents(
            documents=stream_jsonl(input_path),
            output_path=output_path,
            labels=labels,
            examples=examples,
            result_model=result_model,
            batch_size=batch_size,
            max_tokens=max_tokens,
            max_document_tokens=max_document_tokens,
            max_documents=max_documents,
            language=language,
        )

    def run_dataset(
        self,
        *,
        dataset_name: str,
        output_path: Path,
        labels: Sequence[TopicLabel],
        examples: Sequence[TopicExample],
        dataset_config: str | None = None,
        dataset_split: str = "train",
        result_model: type[M] = LabelSelection,
        batch_size: int = 32,
        max_tokens: int = 256,
        max_document_tokens: int = 8192,
        max_documents: int | None = None,
        language: str = "English",
    ) -> int:
        """Stream a HuggingFace dataset, label it, save results, and resume."""
        if not dataset_name.strip():
            raise ValueError("dataset_name must not be empty")
        if not dataset_split.strip():
            raise ValueError("dataset_split must not be empty")
        return self._run_documents(
            documents=stream_huggingface_dataset(
                dataset_name,
                dataset_config=dataset_config,
                dataset_split=dataset_split,
            ),
            output_path=output_path,
            labels=labels,
            examples=examples,
            result_model=result_model,
            batch_size=batch_size,
            max_tokens=max_tokens,
            max_document_tokens=max_document_tokens,
            max_documents=max_documents,
            language=language,
        )

    def _process_batch(
        self,
        documents: Sequence[tuple[int, InputDocument]],
        store: JsonlResultStore,
        labels: Sequence[TopicLabel],
        examples: Sequence[TopicExample],
        result_model: type[M],
        max_tokens: int,
        max_document_tokens: int,
        language: str,
    ) -> int:
        messages = []
        schemas = []
        mappings = []
        for _, document in documents:
            url = document.model_extra.get("url", "") if document.model_extra else None
            document_text = self._truncate_document_text(
                document.text, max_document_tokens
            )
            prompt_seed = _prompt_seed(self.seed, document.doc_id)
            label_text, example_text, temporary_to_stable = _format_prompt_resources(
                labels, examples, prompt_seed
            )
            if language == "English":
                prompt = get_english_topic_classification_prompt(
                    text=document_text,
                    url=str(url) if url else None,
                    labels=label_text,
                    examples=example_text,
                )
            else:
                prompt = get_multilingual_topic_classification_prompt(
                    text=document_text,
                    url=str(url) if url else None,
                    labels=label_text,
                    examples=example_text,
                    language=language,
                )
            messages.append(prompt)
            schemas.append(build_label_schema(result_model, list(temporary_to_stable)))
            mappings.append(temporary_to_stable)

        temporary_results = self._run_structured_chat(
            messages, schemas, result_model, max_tokens
        )
        results = []
        for result, temporary_to_stable in zip(temporary_results, mappings):
            stable_result = result.model_dump(mode="python")
            try:
                stable_result["labels"] = [
                    temporary_to_stable[temporary_label]
                    for temporary_label in result.labels
                ]
            except KeyError as exc:
                raise StructuredModelOutputError(
                    f"Model returned unknown temporary label {exc.args[0]!r}"
                ) from exc
            results.append(result_model.model_validate(stable_result))
        store.append(documents, results)
        return len(results)

    def _truncate_document_text(self, text: str, max_tokens: int) -> str:
        """Tokenize and truncate document text with the loaded model tokenizer."""
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_tokens,
        )
        input_ids = encoded["input_ids"]
        return self.tokenizer.decode(input_ids, skip_special_tokens=True)


def run_topic_jsonl(
    client: ModelClient,
    *,
    input_path: Path,
    output_path: Path,
    labels_path: Path,
    examples_path: Path,
    batch_size: int = 32,
    max_tokens: int = 256,
    max_document_tokens: int = 8192,
    max_documents: int | None = None,
    language: str = "English",
) -> int:
    """Run topic classification using YAML labels and demonstrations."""
    labels, examples = load_topic_resources(labels_path, examples_path)
    return client.run_jsonl(
        input_path=input_path,
        output_path=output_path,
        labels=labels,
        examples=examples,
        batch_size=batch_size,
        max_tokens=max_tokens,
        max_document_tokens=max_document_tokens,
        max_documents=max_documents,
        language=language,
    )


def run_topic_dataset(
    client: ModelClient,
    *,
    dataset_name: str,
    output_path: Path,
    labels_path: Path,
    examples_path: Path,
    dataset_config: str | None = None,
    dataset_split: str = "train",
    batch_size: int = 32,
    max_tokens: int = 256,
    max_document_tokens: int = 8192,
    max_documents: int | None = None,
    language: str = "English",
) -> int:
    """Run topic classification from a streamed HuggingFace dataset."""
    labels, examples = load_topic_resources(labels_path, examples_path)
    return client.run_dataset(
        dataset_name=dataset_name,
        output_path=output_path,
        labels=labels,
        examples=examples,
        dataset_config=dataset_config,
        dataset_split=dataset_split,
        batch_size=batch_size,
        max_tokens=max_tokens,
        max_document_tokens=max_document_tokens,
        max_documents=max_documents,
        language=language,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for topic classification."""
    parser = argparse.ArgumentParser(
        description=(
            "Classify local JSONL or streamed HuggingFace web documents "
            "with structured LLM output."
        )
    )
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
        "--thinking-mode",
        choices=("disabled", "enabled", "template-default"),
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--max-document-tokens", type=int)
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--language")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
    )
    return parser


def _load_cli_config(path: Path) -> dict[str, Any]:
    """Load the supported nested YAML configuration structure."""
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Top-level config must be a mapping: {path}")

    allowed = {"paths", "model", "run"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown config sections: {sorted(unknown)}")
    section_keys = {
        "paths": {
            "base_dir",
            "input_jsonl",
            "input_dataset",
            "dataset_config",
            "dataset_split",
            "output",
            "labels",
            "examples",
        },
        "model": {
            "name",
            "tensor_parallel_size",
            "gpu_memory_utilization",
            "dtype",
            "max_model_len",
            "seed",
            "thinking_mode",
        },
        "run": {
            "batch_size",
            "max_tokens",
            "max_document_tokens",
            "max_documents",
            "language",
            "log_level",
        },
    }
    flattened: dict[str, Any] = {}
    destinations = {
        ("paths", "base_dir"): "base_dir",
        ("paths", "input_jsonl"): "input_jsonl",
        ("paths", "input_dataset"): "input_dataset",
        ("paths", "dataset_config"): "dataset_config",
        ("paths", "dataset_split"): "dataset_split",
        ("paths", "output"): "output_path",
        ("paths", "labels"): "labels_path",
        ("paths", "examples"): "examples_path",
        ("model", "name"): "model_name",
        ("model", "tensor_parallel_size"): "tensor_parallel_size",
        ("model", "gpu_memory_utilization"): "gpu_memory_utilization",
        ("model", "dtype"): "dtype",
        ("model", "max_model_len"): "max_model_len",
        ("model", "seed"): "seed",
        ("model", "thinking_mode"): "thinking_mode",
        ("run", "batch_size"): "batch_size",
        ("run", "max_tokens"): "max_tokens",
        ("run", "max_document_tokens"): "max_document_tokens",
        ("run", "max_documents"): "max_documents",
        ("run", "language"): "language",
        ("run", "log_level"): "log_level",
    }
    for section, keys in section_keys.items():
        value = raw.get(section, {})
        if not isinstance(value, dict):
            raise ValueError(f"Config section {section!r} must be a mapping")
        unknown_keys = set(value) - keys
        if unknown_keys:
            raise ValueError(
                f"Unknown keys in config section {section!r}: {sorted(unknown_keys)}"
            )
        for key, item in value.items():
            flattened[destinations[(section, key)]] = item
    configured_base_dir = flattened.get("base_dir")
    if configured_base_dir is None:
        base_dir = path.parent
    else:
        base_dir = Path(configured_base_dir)
        if not base_dir.is_absolute():
            base_dir = path.parent / base_dir
    base_dir = base_dir.resolve()

    resolved: dict[str, Any] = {}
    for name, value in flattened.items():
        if name == "base_dir":
            continue
        if name in {
            "input_jsonl",
            "output_path",
            "labels_path",
            "examples_path",
        }:
            if not isinstance(value, str):
                raise ValueError(f"Config path {name!r} must be a string")
            expanded = value.replace("${base_dir}", str(base_dir))
            if "${" in expanded:
                raise ValueError(f"Unknown variable in config path {value!r}")
            candidate = Path(expanded)
            resolved[name] = (
                candidate if candidate.is_absolute() else base_dir / candidate
            )
        else:
            resolved[name] = value
    return resolved


def _merge_cli_and_config(
    parser: argparse.ArgumentParser, argv: Sequence[str] | None
) -> argparse.Namespace:
    """Parse YAML defaults first, then let explicit CLI values override them."""
    preliminary = parser.parse_args(argv)
    config_values: dict[str, Any] = {}
    config_path = preliminary.config
    if config_path is not None:
        config_values = _load_cli_config(config_path)
        for name in ("input_jsonl", "output_path", "labels_path", "examples_path"):
            value = config_values.get(name)
            if value is not None:
                config_values[name] = Path(value)
                if not config_values[name].is_absolute():
                    config_values[name] = config_path.parent / config_values[name]

    parser.set_defaults(**config_values)
    args = parser.parse_args(argv)
    defaults = {
        "labels_path": Path("scripts/data_processing/topics.yaml"),
        "examples_path": Path("scripts/data_processing/topic_examples.yaml"),
        "model_name": "Qwen/Qwen3.6-35B-A3B",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.9,
        "dtype": "auto",
        "max_model_len": 32768,
        "seed": 0,
        "thinking_mode": "disabled",
        "batch_size": 32,
        "max_tokens": 256,
        "max_document_tokens": 8192,
        "max_documents": None,
        "language": "English",
        "log_level": "INFO",
    }
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
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
    if args.batch_size < 1 or args.max_tokens < 1 or args.max_document_tokens < 1:
        parser.error(
            "--batch-size, --max-tokens, and --max-document-tokens must be >= 1"
        )
    if args.max_documents is not None and args.max_documents < 1:
        parser.error("--max-documents must be >= 1")
    try:
        args.language = normalize_language(args.language)
    except ValueError as exc:
        parser.error(str(exc))
    if args.max_model_len < 1 or args.tensor_parallel_size < 1:
        parser.error("--max-model-len and --tensor-parallel-size must be >= 1")
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    if args.seed < 0:
        parser.error("--seed must be >= 0")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Run topic classification from CLI and YAML configuration."""
    parser = build_arg_parser()
    args = _merge_cli_and_config(parser, argv)
    LOGGER.info("Run args:")
    for key, value in vars(args).items():
        LOGGER.info("  %s: %s", key, value)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    client = ModelClient(
        model_name=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        seed=args.seed,
        thinking_mode=args.thinking_mode,
    )
    common_kwargs = {
        "output_path": args.output_path,
        "labels_path": args.labels_path,
        "examples_path": args.examples_path,
        "batch_size": args.batch_size,
        "max_tokens": args.max_tokens,
        "max_document_tokens": args.max_document_tokens,
        "max_documents": args.max_documents,
        "language": args.language,
    }
    if args.input_jsonl is not None:
        run_topic_jsonl(client, input_path=args.input_jsonl, **common_kwargs)
    else:
        run_topic_dataset(
            client,
            dataset_name=args.input_dataset,
            dataset_config=args.dataset_config,
            dataset_split=args.dataset_split,
            **common_kwargs,
        )
    return 0


if __name__ == "__main__":
    main()
