from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Callable, Iterable, Sequence
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
from vllm import SamplingParams  # type: ignore
from vllm.sampling_params import StructuredOutputsParams  # type: ignore
from transformers.tokenization_utils_base import BatchEncoding  # type: ignore

try:
    from .labeling_common import (
        InputDocument,
        JsonlResultStore,
        LabelDefinition,
        LabelExample,
        SavedRecordBase,
        build_common_arg_parser,
        build_llm,
        document_url,
        exit_without_finalization,
        load_label_resources,
        log_model_throughput,
        normalize_label_type,
        normalize_language,
        parse_cli_and_config,
        process_documents_in_batches,
        prompt_seed,
        shuffle_prompt_order,
        stream_huggingface_dataset,
        stream_jsonl,
        thinking_chat_template_kwargs,
        truncate_document_text,
    )
    from .multilabel_prompts import (
        get_format_classification_prompt,
        get_topic_classification_prompt,
    )
except ImportError:
    from labeling_common import (
        InputDocument,
        JsonlResultStore,
        LabelDefinition,
        LabelExample,
        SavedRecordBase,
        build_common_arg_parser,
        build_llm,
        document_url,
        exit_without_finalization,
        load_label_resources,
        log_model_throughput,
        normalize_label_type,
        normalize_language,
        parse_cli_and_config,
        process_documents_in_batches,
        prompt_seed,
        shuffle_prompt_order,
        stream_huggingface_dataset,
        stream_jsonl,
        thinking_chat_template_kwargs,
        truncate_document_text,
    )
    from multilabel_prompts import (
        get_format_classification_prompt,
        get_topic_classification_prompt,
    )

M = TypeVar("M", bound=BaseModel)
LOGGER = logging.getLogger(__name__)

THINKING_MODES = ("disabled", "enabled", "template-default")


class LabelSelection(BaseModel):
    """Ordered structured response for a single document."""

    model_config = ConfigDict(extra="forbid", strict=True)

    rationale: str | None = Field(default=None, max_length=1000)
    labels: list[str] = Field(min_length=1)
    bad_example: bool

    @field_validator("labels")
    @classmethod
    def labels_must_be_unique(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("labels must not contain duplicates")
        return value


class SavedLabelRecord(SavedRecordBase):
    """Flat JSONL output record with the document metadata and labels."""

    text: str = ""
    label: str = Field(min_length=1)
    labels: list[str] = Field(min_length=1)
    label_rationale: str = ""
    bad_example: bool = False

    @field_validator("labels")
    @classmethod
    def labels_must_be_unique(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("labels must not contain duplicates")
        return value

    @model_validator(mode="after")
    def ensure_primary_label_matches_first_label(self) -> "SavedLabelRecord":
        if not self.labels:
            raise ValueError("labels must not be empty")
        if self.label != self.labels[0]:
            raise ValueError(
                f"label must be the first label in labels; got {self.label!r} "
                f"and {self.labels[0]!r}"
            )
        return self


def label_record_validator(
    allowed_labels: set[str],
) -> Callable[[SavedLabelRecord], None]:
    """Build a store check that saved labels belong to the current label set.

    Args:
        allowed_labels: Label short names of the current labeling schema.

    Returns:
        A function raising ``ValueError`` for records with unknown labels.
    """

    def validate(record: SavedLabelRecord) -> None:
        unknown_labels = set(record.labels) - allowed_labels
        if unknown_labels:
            raise ValueError(
                f"Result for {record.id!r} selected unknown labels: "
                f"{sorted(unknown_labels)!r}"
            )

    return validate


def open_result_store(
    output_path: Path, labels: Sequence[LabelDefinition]
) -> JsonlResultStore[SavedLabelRecord]:
    """Open the multi-label result store for the given label set."""
    return JsonlResultStore(
        output_path,
        SavedLabelRecord,
        label_record_validator({label.short_name for label in labels}),
    )


def _format_prompt_resources(
    labels: Sequence[LabelDefinition], examples: Sequence[LabelExample], seed: int
) -> tuple[str, str, dict[str, str]]:
    """Shuffle resources and return prompt text plus temporary-to-stable IDs."""
    label_order, example_order = shuffle_prompt_order(len(labels), len(examples), seed)
    shuffled_labels = [labels[index] for index in label_order]
    shuffled_examples = [examples[index] for index in example_order]

    temporary_to_stable = {
        f"L{index:02d}": label.id
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
                f"Example references unknown label name {example.label!r}"
            ) from exc
        example_blocks.append(
            f"URL: {example.url}\n"
            f"Content: {example.text}\n"
            f"Label: {temporary_id}\n"
            f"Explanation: {example.explanation}"
        )
    return "\n".join(label_lines), "\n\n".join(example_blocks), temporary_to_stable


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
        label_type: str = "topic",
    ) -> None:
        self.model_name = model_name
        self.seed = seed
        self.thinking_mode = thinking_mode
        self.label_type = normalize_label_type(label_type)

        if "eagle" in model_name.lower() and "mistral" in model_name.lower():
            speculative_config = {
                "model": "mistralai/Mistral-Medium-3.5-128B-EAGLE",
                "num_speculative_tokens": 3,
                "method": "eagle",
                "max_model_len": max_model_len,
            }
        else:
            speculative_config = None

        self.llm = build_llm(
            model_name=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            max_model_len=max_model_len,
            seed=seed,
            speculative_config=speculative_config,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.max_model_len = int(self.llm.model_config.max_model_len)

    @property
    def chat_template_kwargs(self) -> dict[str, Any] | None:
        """Return the thinking effort kwargs for the chat template."""
        return thinking_chat_template_kwargs(self.model_name, self.thinking_mode)

    def _sampling_params(
        self,
        json_schema: dict,
        *,
        max_tokens: int,
        temperature_override: float | None = None,
    ) -> SamplingParams:
        structured = StructuredOutputsParams(json=json_schema)

        # Set temperature based on model family
        if "qwen" in self.model_name.lower():
            temperature = 0.7
            top_p = 0.80
            top_k = 20
        elif "gemma" in self.model_name.lower():
            temperature = 1.0
            top_p = 0.95
            top_k = 64
        elif "mistral" in self.model_name.lower():
            temperature = 0.5
            top_p = 0.95
            top_k = 64
        else:
            temperature = 0.7
            top_p = 0.95
            top_k = 64

        return SamplingParams(
            temperature=(
                temperature if temperature_override is None else temperature_override
            ),
            seed=self.seed,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
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

    def _run_structured_chat(
        self,
        input_batch: list[list[dict[str, str]]],
        json_schemas: Sequence[dict[str, Any]],
        result_model: type[M],
        max_tokens: int,
        temperature_override: float | None = None,
    ) -> tuple[list[M | None], list[StructuredModelOutputError | None]]:
        if len(input_batch) != len(json_schemas):
            raise ValueError("The number of prompts and schemas must match")

        sampling_params = [
            self._sampling_params(
                schema,
                max_tokens=max_tokens,
                temperature_override=temperature_override,
            )
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
        log_model_throughput(
            input_tokens=input_tokens,
            generated_tokens=generated_tokens,
            elapsed_seconds=generate_end_time - generate_start_time,
        )

        parsed: list[M | None] = []
        errors: list[StructuredModelOutputError | None] = []
        for request_output in outputs:
            completion = request_output.outputs[0]

            if completion.finish_reason != "stop":
                parsed.append(None)
                errors.append(
                    StructuredModelOutputError(
                        f"generation ended with finish_reason="
                        f"{completion.finish_reason!r}; output may be truncated.\n"
                        f"Configured max_tokens={max_tokens}\n"
                        f"Generated tokens: {len(completion.token_ids)}\n"
                        f"Generated text:\n{completion.text!r}"
                    )
                )
                continue

            generated_text = completion.text
            if not generated_text:
                parsed.append(None)
                errors.append(StructuredModelOutputError("model returned empty output"))
                continue

            try:
                # Drop duplicate labels while preserving order, then validate the result model
                # If duplicate labels are present, the last occurrenece will be dropped.
                raw_result = json.loads(generated_text)
                if isinstance(raw_result, dict) and isinstance(
                    raw_result.get("labels"), list
                ):
                    raw_result["labels"] = list(dict.fromkeys(raw_result["labels"]))
                parsed.append(result_model.model_validate(raw_result))
                errors.append(None)
            except (
                ValidationError,
                json.JSONDecodeError,
                TypeError,
                ValueError,
            ) as exc:
                preview = generated_text[:2000]
                parsed.append(None)
                errors.append(
                    StructuredModelOutputError(
                        f"model output failed "
                        f"{result_model.__name__} validation: {exc}. "
                        f"Output preview: {preview!r}"
                    )
                )

        return parsed, errors

    def _run_documents(
        self,
        *,
        documents: Iterable[tuple[int, InputDocument]],
        output_path: Path,
        labels: Sequence[LabelDefinition],
        examples: Sequence[LabelExample],
        result_model: type[M] = LabelSelection,
        batch_size: int = 32,
        max_tokens: int = 256,
        max_document_tokens: int = 8192,
        max_documents: int | None = None,
        language: str = "English",
        label_type: str = "topic",
    ) -> int:
        """Process validated documents, save results, and resume safely."""
        stable_ids = [label.id for label in labels]
        if not stable_ids or len(stable_ids) != len(set(stable_ids)):
            raise ValueError("labels must contain unique stable IDs")
        if not examples:
            raise ValueError("examples must contain at least one demonstration")
        if max_document_tokens < 1:
            raise ValueError("max_document_tokens must be >= 1")
        language = normalize_language(language)
        label_type = normalize_label_type(label_type)
        short_names = [label.short_name for label in labels]
        if len(short_names) != len(set(short_names)):
            raise ValueError("labels must contain unique short names")
        store = open_result_store(output_path, labels)
        return process_documents_in_batches(
            documents=documents,
            store=store,
            process_batch=lambda batch: self._process_batch(
                batch,
                store,
                labels,
                examples,
                result_model,
                max_tokens,
                max_document_tokens,
                language,
                label_type,
            ),
            batch_size=batch_size,
            max_documents=max_documents,
        )

    def run_jsonl(
        self,
        *,
        input_path: Path,
        output_path: Path,
        labels: Sequence[LabelDefinition],
        examples: Sequence[LabelExample],
        result_model: type[M] = LabelSelection,
        batch_size: int = 32,
        max_tokens: int = 256,
        max_document_tokens: int = 8192,
        max_documents: int | None = None,
        language: str = "English",
        label_type: str = "topic",
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
            label_type=label_type,
        )

    def run_dataset(
        self,
        *,
        dataset_name: str,
        output_path: Path,
        labels: Sequence[LabelDefinition],
        examples: Sequence[LabelExample],
        dataset_config: str | None = None,
        dataset_split: str = "train",
        result_model: type[M] = LabelSelection,
        batch_size: int = 32,
        max_tokens: int = 256,
        max_document_tokens: int = 8192,
        max_documents: int | None = None,
        language: str = "English",
        label_type: str = "topic",
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
            label_type=label_type,
        )

    def _process_batch(
        self,
        documents: Sequence[tuple[int, InputDocument]],
        store: JsonlResultStore[SavedLabelRecord],
        labels: Sequence[LabelDefinition],
        examples: Sequence[LabelExample],
        result_model: type[M],
        max_tokens: int,
        max_document_tokens: int,
        language: str,
        label_type: str,
    ) -> int:
        messages = []
        schemas = []
        mappings = []
        stable_to_short = {label.id: label.short_name for label in labels}
        for _, document in documents:
            document_text = truncate_document_text(
                self.tokenizer, document.text, max_document_tokens
            )
            label_text, example_text, temporary_to_stable = _format_prompt_resources(
                labels, examples, prompt_seed(self.seed, document.doc_id)
            )
            prompt_function = (
                get_topic_classification_prompt
                if label_type == "topic"
                else get_format_classification_prompt
            )
            prompt = prompt_function(
                text=document_text,
                url=document_url(document),
                labels=label_text,
                examples=example_text,
                language=language,
            )
            messages.append(prompt)
            schemas.append(build_label_schema(result_model, list(temporary_to_stable)))
            mappings.append(temporary_to_stable)

        successful_results: list[M | None] = [None] * len(documents)
        failed_indices = list(range(len(documents)))
        failure_messages: dict[int, str] = {}

        # Retry failed documents up to 4 times, with the last attempt using temperature=0.0
        for attempt in range(4):
            if not failed_indices:
                break
            temperature_override = 0.0 if attempt == 3 else None
            retry_indices = failed_indices
            temporary_results, errors = self._run_structured_chat(
                [messages[index] for index in retry_indices],
                [schemas[index] for index in retry_indices],
                result_model,
                max_tokens,
                temperature_override=temperature_override,
            )
            next_failed_indices: list[int] = []
            for retry_index, document_index in enumerate(retry_indices):
                result = temporary_results[retry_index]
                error = errors[retry_index]
                if result is not None and error is None:
                    successful_results[document_index] = result
                else:
                    next_failed_indices.append(document_index)
                    failure_messages[document_index] = str(error)
            failed_indices = next_failed_indices
            if failed_indices and attempt < 3:
                retry_mode = "temperature=0" if attempt == 2 else "same settings"
                LOGGER.warning(
                    "%d/%d document outputs failed; retrying attempt %d/4 with %s",
                    len(failed_indices),
                    len(retry_indices),
                    attempt + 2,
                    retry_mode,
                )

        successful_documents: list[tuple[int, InputDocument]] = []
        results: list[SavedLabelRecord] = []
        for document_index, (
            (line_number, document),
            result,
            temporary_to_stable,
        ) in enumerate(zip(documents, successful_results, mappings)):
            if result is None:
                continue
            stable_result = result.model_dump(mode="python")
            try:
                stable_labels = [
                    temporary_to_stable[temporary_label]
                    for temporary_label in result.labels
                ]
                ordered_labels = [
                    stable_to_short[stable_label] for stable_label in stable_labels
                ]
            except KeyError as exc:
                raise StructuredModelOutputError(
                    f"Model returned unknown label {exc.args[0]!r}"
                ) from exc
            results.append(
                SavedLabelRecord(
                    id=document.doc_id,
                    text=document.text,
                    label=ordered_labels[0],
                    labels=ordered_labels,
                    label_rationale=result.rationale or "",
                    bad_example=result.bad_example,
                )
            )
            successful_documents.append((line_number, document))

        if failed_indices:
            failure_details = "; ".join(
                f"{documents[index][1].doc_id}: {failure_messages[index]}"
                for index in failed_indices
            )
            raise StructuredModelOutputError(
                f"{len(failed_indices)} document outputs failed after 4 attempts: "
                f"{failure_details}"
            )
        if successful_documents:
            store.append(successful_documents, results)
        return len(successful_documents)


def run_jsonl(
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
    label_type: str = "topic",
) -> int:
    """Run classification using YAML labels and demonstrations."""
    labels, examples = load_label_resources(labels_path, examples_path, label_type)
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
        label_type=label_type,
    )


def run_dataset(
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
    label_type: str = "topic",
) -> int:
    """Run classification from a streamed HuggingFace dataset."""
    labels, examples = load_label_resources(labels_path, examples_path, label_type)
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
        label_type=label_type,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for topic classification."""
    parser = build_common_arg_parser(
        description=(
            "Classify local JSONL or streamed HuggingFace web documents "
            "with structured LLM output."
        ),
        thinking_modes=THINKING_MODES,
    )
    parser.add_argument("--max-tokens", type=int)
    return parser


def _merge_cli_and_config(
    parser: argparse.ArgumentParser, argv: Sequence[str] | None
) -> argparse.Namespace:
    """Parse YAML defaults first, then let explicit CLI values override them."""
    args = parse_cli_and_config(
        parser,
        argv,
        thinking_modes=THINKING_MODES,
        extra_defaults={"max_tokens": 256},
        extra_run_keys={"max_tokens": "max_tokens"},
    )
    if args.max_tokens < 1:
        parser.error("--max-tokens must be >= 1")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Run topic classification from CLI and YAML configuration."""
    parser = build_arg_parser()
    args = _merge_cli_and_config(parser, argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    LOGGER.info("Run args:")
    for key, value in vars(args).items():
        LOGGER.info("  %s: %s", key, value)

    # Check if we already have enough completed documents before loading the model
    if args.max_documents is not None:
        labels, _ = load_label_resources(
            args.labels_path, args.examples_path, args.label_type
        )
        completed_results = open_result_store(args.output_path, labels)
        completed_count = len(completed_results.completed)
        if completed_count >= args.max_documents:
            LOGGER.info(
                "Already have %d completed documents; max_documents=%d reached "
                "before loading the model. Terminating early.",
                completed_count,
                args.max_documents,
            )
            return 0

    client = ModelClient(
        model_name=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        seed=args.seed,
        thinking_mode=args.thinking_mode,
        label_type=args.label_type,
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
        "label_type": args.label_type,
    }
    if args.input_jsonl is not None:
        run_jsonl(client, input_path=args.input_jsonl, **common_kwargs)
    else:
        run_dataset(
            client,
            dataset_name=args.input_dataset,
            dataset_config=args.dataset_config,
            dataset_split=args.dataset_split,
            **common_kwargs,
        )
    return 0


if __name__ == "__main__":
    # Exceptions propagate normally; only a completed run skips interpreter
    # finalization. See labeling_common.exit_without_finalization.
    exit_without_finalization(main())
