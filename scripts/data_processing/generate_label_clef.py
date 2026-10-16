#!/usr/bin/env python3
"""Label documents with topic or format label distributions using Cloudflare/clef.

Clef is a decision model built on Qwen3.8-27B: it takes a state (here the web
page), instructions, and the allowed options with their descriptions (here the
labels with their definitions), and scores every option in a single forward
pass. The output is a full probability distribution over the labels for each
document, with no text generation and no output parsing.

This script is the Clef counterpart of ``generate_label_distributions.py`` and
shares its inputs, configuration, and output format, so the evaluation and
training scripts read Clef results like those of any other model:

* Labels and their definitions come from the same label YAML files, and the
  instructions are the chat prompt's rules (``get_decision_instructions``).
* Input streaming, resuming, batching, and the output records
  (``SavedDistributionRecord``) are the shared ones.

Differences from the chat-model pipeline, all imposed by Clef:

* **No demonstrations.** Clef's input has no place for few-shot examples.
* **Fixed label order.** Clef sorts the options by their ID (the label name),
  so every document sees the labels in the same alphabetical order. Records
  store ``prompt_order="fixed"``. Position effects can therefore not be
  averaged out by shuffling, and the letter-bias analysis doesn't apply.
* **No prefix sharing.** The page comes before the labels in Clef's input, so
  the label definitions are processed again for every document.

Clef ships its own model code (``joint_schema_model.py`` in the model
repository), which this script imports and runs. ``REVIEWED_REVISION`` pins the
repository commit whose code was read before use; pass ``--revision`` to use
another one deliberately.

Example:
    python generate_label_clef.py --config configs/roihu/distribution/generate_clef.yaml \
        --dataset-config fin_Latn --dataset-split tower72b_parallel --language finnish \
        --label-type topic --output results/distribution/clef-annotations/topic/finnish-parallel.jsonl
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

try:
    from .distribution_prompts import (
        BAD_EXAMPLE_LABEL,
        OTHER_LABEL,
        get_decision_instructions,
    )
    from .distribution_records import (
        FIXED_PROMPT_ORDER,
        SavedDistributionRecord,
        compute_label_distribution,
        open_result_store,
    )
    from .labeling_common import (
        InputDocument,
        JsonlResultStore,
        LabelDefinition,
        build_common_arg_parser,
        document_url,
        exit_without_finalization,
        load_label_resources,
        log_model_throughput,
        normalize_label_type,
        normalize_language,
        parse_cli_and_config,
        process_documents_in_batches,
        stream_huggingface_dataset,
        stream_jsonl,
        TRUNCATION_MARKER,
        truncate_document_chars,
        truncate_document_text,
    )
except ImportError:
    from distribution_prompts import (
        BAD_EXAMPLE_LABEL,
        OTHER_LABEL,
        get_decision_instructions,
    )
    from distribution_records import (
        FIXED_PROMPT_ORDER,
        SavedDistributionRecord,
        compute_label_distribution,
        open_result_store,
    )
    from labeling_common import (
        InputDocument,
        JsonlResultStore,
        LabelDefinition,
        build_common_arg_parser,
        document_url,
        exit_without_finalization,
        load_label_resources,
        log_model_throughput,
        normalize_label_type,
        normalize_language,
        parse_cli_and_config,
        process_documents_in_batches,
        stream_huggingface_dataset,
        stream_jsonl,
        TRUNCATION_MARKER,
        truncate_document_chars,
        truncate_document_text,
    )

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL = "Cloudflare/clef"
# Commit of the model repository whose joint_schema_model.py was read before use.
# The script executes that file, so the default is pinned for safety and repeatability.
REVIEWED_REVISION = "ed3eed331870db2eff4b0db01237128ede8a00ce"
# Clef's input always ends with an empty thinking block; there is no other mode.
THINKING_MODES = ("disabled",)
# Clef's own default context length; the base model supports more.
DEFAULT_MAX_MODEL_LEN = 16384
# Upper bound on the padded size (documents x longest document, in tokens) of one
# forward pass. The 27B model takes about 54 GB of a 96 GB GH200 in bf16.
DEFAULT_MAX_BATCH_TOKENS = 32768
# Documents read, scored, and saved together; forward passes are smaller (see above).
DEFAULT_BATCH_SIZE = 256
# Minutes without a completed batch before the stall watchdog ends a run. Batches of
# 512 documents normally take 3-6 minutes on one GH200.
DEFAULT_STALL_TIMEOUT_MINUTES = 20.0


def render_state(text: str, url: str | None) -> str:
    """Render one web page as Clef's state.

    Args:
        text: Page text, already truncated.
        url: URL of the page, if available.

    Returns:
        The page text, preceded by its URL when there is one.
    """
    return f"URL: {url}\n\n{text}" if url else text


def build_criteria(labels: Sequence[LabelDefinition]) -> dict[str, str]:
    """Return Clef's options: label name -> definition.

    The full label name is the option ID, as in the chat prompt, where each
    label is shown as ``<name> - <definition>``.

    Raises:
        ValueError: If two labels share a name.
    """
    criteria = {label.name: label.definition for label in labels}
    if len(criteria) != len(labels):
        raise ValueError("labels must have unique names")
    return criteria


def build_clef_record(
    state: str, question_id: str, instructions: str, criteria: dict[str, str]
) -> dict[str, Any]:
    """Build a Clef record asking one choice question about one page.

    Args:
        state: The page, passed to Clef as the state.
        question_id: ID of the question, the label type.
        instructions: Task instructions.
        criteria: Options, label name -> definition.

    Returns:
        A record in Clef's input format.
    """
    return {
        "state": state,
        "questions": {
            question_id: {
                "type": "choice",
                "instructions": instructions,
                "criteria": criteria,
            },
        },
    }


def pack_batches(lengths: Sequence[int], max_batch_tokens: int) -> list[list[int]]:
    """Group documents into forward passes of bounded padded size.

    Documents are sorted by length, so each pass pads little, and a pass is
    closed when adding the next document would make ``documents x longest
    document`` exceed ``max_batch_tokens``. A document longer than the bound
    gets a pass of its own.

    Args:
        lengths: Token count of each document's model input.
        max_batch_tokens: Maximum padded tokens per forward pass.

    Returns:
        Lists of document indices, shortest documents first.

    Raises:
        ValueError: If ``max_batch_tokens`` is not positive.
    """
    if max_batch_tokens < 1:
        raise ValueError("max_batch_tokens must be >= 1")
    order = sorted(range(len(lengths)), key=lambda index: lengths[index])
    batches: list[list[int]] = []
    current: list[int] = []
    for index in order:
        # Sorted ascending, so this document is the longest of its batch.
        if current and (len(current) + 1) * lengths[index] > max_batch_tokens:
            batches.append(current)
            current = []
        current.append(index)
    if current:
        batches.append(current)
    return batches


def build_distribution_record(
    document: InputDocument,
    option_ids: Sequence[str],
    logits: Sequence[float],
    labels: Sequence[LabelDefinition],
) -> SavedDistributionRecord:
    """Convert Clef's option logits for one document into a distribution record.

    Args:
        document: The scored input document.
        option_ids: Option IDs (label names) in the order Clef presented them.
        logits: One logit per option, in the same order.
        labels: Label definitions in label-file order.

    Returns:
        The label distribution record for the document.

    Raises:
        RuntimeError: If the options don't match the labels, or a logit is not finite.
    """
    name_to_short = {label.name: label.short_name for label in labels}
    if len(option_ids) != len(logits):
        raise RuntimeError(
            f"{document.doc_id!r}: {len(logits)} logits for {len(option_ids)} options"
        )
    if len(option_ids) != len(labels) or set(option_ids) != set(name_to_short):
        raise RuntimeError(
            f"{document.doc_id!r}: option IDs {list(option_ids)!r} do not match the label set"
        )
    if not all(math.isfinite(value) for value in logits):
        raise RuntimeError(f"{document.doc_id!r}: non-finite option logits {list(logits)!r}")
    # A softmax over the options, in log space; the same normalization as for
    # the answer-letter log-probabilities of the chat models.
    normalized = compute_label_distribution(list(logits))
    presented_short_names = [name_to_short[option_id] for option_id in option_ids]
    logprob_by_name = dict(zip(presented_short_names, normalized))
    label_logprobs = {
        label.short_name: logprob_by_name[label.short_name] for label in labels
    }
    label_probs = {name: math.exp(value) for name, value in label_logprobs.items()}
    # First maximum in label-file order, as in the chat-model pipeline.
    top_label = max(label_probs, key=label_probs.__getitem__)
    return SavedDistributionRecord(
        id=document.doc_id,
        text=document.text,
        # The same value the state used (render_state), from the same helper.
        url=document_url(document),
        label=top_label,
        label_prob=label_probs[top_label],
        label_probs=label_probs,
        label_logprobs=label_logprobs,
        label_order=presented_short_names,
        example_order=[],
        prompt_order=FIXED_PROMPT_ORDER,
    )


def resolve_model_dir(model: str, revision: str | None) -> Path:
    """Return a local directory with the model, downloading it if needed.

    Args:
        model: HF repo id or an existing local directory.
        revision: HF revision (commit hash); must be None for a local directory.

    Returns:
        Path to the local snapshot directory.

    Raises:
        ValueError: If a revision is given together with a local directory.
    """
    local = Path(model)
    if local.is_dir():
        if revision is not None:
            raise ValueError("--revision cannot be used with a local model directory")
        return local
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model, revision=revision))


class ClefModelClient:
    """Clef client that scores all labels of each document in one forward pass."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL,
        revision: str | None = REVIEWED_REVISION,
        dtype: str = "auto",
        max_model_len: int = DEFAULT_MAX_MODEL_LEN,
        seed: int = 0,
        device: str = "cuda",
        thinking_mode: str = "disabled",
    ) -> None:
        """Load Clef and its release code.

        Args:
            model_name: HF repo id or local snapshot directory.
            revision: Repository commit to load; None for a local directory.
            dtype: Torch dtype name, or ``"auto"`` for Clef's default (bfloat16).
            max_model_len: Maximum total input length in tokens.
            seed: Torch seed. Scoring is deterministic; the seed has no effect
                on the label order, which Clef fixes.
            device: Torch device string, e.g. ``"cuda"``.
            thinking_mode: Must be ``"disabled"``.

        Raises:
            ValueError: If the thinking mode or dtype is not supported.
        """
        if thinking_mode not in THINKING_MODES:
            raise ValueError("thinking_mode must be 'disabled': Clef has no thinking mode")
        if max_model_len < 1:
            raise ValueError("max_model_len must be >= 1")
        import torch  # imported here so the module's pure helpers work without torch

        torch_dtype = torch.bfloat16 if dtype == "auto" else getattr(torch, dtype, None)
        if not isinstance(torch_dtype, torch.dtype):
            raise ValueError(f"Unknown torch dtype {dtype!r}")
        torch.manual_seed(seed)

        self.model_name = model_name
        self.max_model_len = max_model_len
        self._torch = torch
        self.device = torch.device(device)

        model_dir = resolve_model_dir(model_name, revision)
        LOGGER.info("Loading Clef from %s (revision %s)", model_dir, revision or "local directory")
        # The release ships its own model code; importing it executes code from the repo.
        sys.path.insert(0, str(model_dir))
        from joint_schema_model import collate_records, encode_record, load_release_model

        self._collate_records = collate_records
        self._encode_record = encode_record
        self.model, self.processor = load_release_model(
            str(model_dir), device=device, dtype=torch_dtype
        )
        self.model.eval()
        self.tokenizer = self.processor.tokenizer
        if self.tokenizer.pad_token_id is None:
            raise ValueError("Clef's tokenizer has no pad token")

    def _encode(self, record: dict[str, Any]) -> Any:
        """Encode one record with Clef's release code."""
        return self._encode_record(
            self.tokenizer, record, max_length=self.max_model_len, processor=self.processor
        )

    def _score(self, encoded: Sequence[Any]) -> list[tuple[tuple[str, ...], list[float]]]:
        """Run one forward pass and return (option IDs, logits) per record.

        Raises:
            RuntimeError: If the GPU runs out of memory or the output doesn't
                have exactly one question per record.
        """
        torch = self._torch
        batch = self._collate_records(list(encoded), self.tokenizer.pad_token_id, self.device)
        try:
            with torch.inference_mode():
                logits = self.model(batch)
        except torch.OutOfMemoryError as exc:
            raise RuntimeError(
                f"Out of GPU memory scoring {len(encoded)} documents of up to "
                f"{max(len(record.input_ids) for record in encoded)} tokens; "
                "lower --max-batch-tokens"
            ) from exc
        if len(logits) != len(encoded):
            raise RuntimeError(f"model returned {len(logits)} outputs for {len(encoded)} records")
        results = []
        for record, record_logits in zip(encoded, logits, strict=True):
            if len(record.questions) != 1 or len(record_logits) != 1:
                raise RuntimeError(
                    f"expected 1 question per record, got {len(record.questions)} "
                    f"with {len(record_logits)} outputs"
                )
            results.append((record.questions[0].option_ids, record_logits[0].float().tolist()))
        return results

    def _run_documents(
        self,
        *,
        documents: Iterable[tuple[int, InputDocument]],
        output_path: Path,
        labels: Sequence[LabelDefinition],
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_batch_tokens: int = DEFAULT_MAX_BATCH_TOKENS,
        max_document_tokens: int = 8192,
        max_document_chars: int | None = None,
        max_documents: int | None = None,
        language: str = "English",
        label_type: str = "topic",
        stall_timeout_seconds: float | None = None,
    ) -> int:
        """Process validated documents, save results, and resume safely."""
        if not labels:
            raise ValueError("labels must not be empty")
        short_names = [label.short_name for label in labels]
        if len(short_names) != len(set(short_names)):
            raise ValueError("labels must contain unique short names")
        if max_document_tokens < 1:
            raise ValueError("max_document_tokens must be >= 1")
        if max_document_chars is not None and max_document_chars < 1:
            raise ValueError("max_document_chars must be >= 1")
        if max_batch_tokens < 1:
            raise ValueError("max_batch_tokens must be >= 1")
        language = normalize_language(language)
        label_type = normalize_label_type(label_type)
        criteria = build_criteria(labels)

        # Everything but the page is the same for every document without a URL:
        # measure it once, so an oversized setup fails before any scoring.
        instructions = self._instructions(label_type, language, labels, has_url=False)
        overhead = len(self._encode(build_clef_record("", label_type, instructions, criteria)).input_ids)
        LOGGER.info(
            "Clef input: %d tokens of instructions and labels per document, "
            "plus up to %d document tokens; max_model_len=%d",
            overhead,
            max_document_tokens,
            self.max_model_len,
        )
        if max_document_chars is None and overhead + max_document_tokens >= self.max_model_len:
            raise ValueError(
                f"{overhead} tokens of instructions and labels plus max_document_tokens="
                f"{max_document_tokens} do not fit max_model_len={self.max_model_len}"
            )

        store = open_result_store(output_path, labels, FIXED_PROMPT_ORDER)
        return process_documents_in_batches(
            documents=documents,
            store=store,
            process_batch=lambda batch: self._process_batch(
                batch,
                store,
                labels,
                criteria,
                max_batch_tokens,
                max_document_tokens,
                max_document_chars,
                language,
                label_type,
            ),
            batch_size=batch_size,
            max_documents=max_documents,
            stall_timeout_seconds=stall_timeout_seconds,
        )

    @staticmethod
    def _instructions(
        label_type: str, language: str, labels: Sequence[LabelDefinition], *, has_url: bool
    ) -> str:
        """Return the task instructions; fallback rules follow the taxonomy."""
        label_names = {label.name for label in labels}
        return get_decision_instructions(
            label_type,
            language,
            has_url=has_url,
            has_other_label=OTHER_LABEL in label_names,
            has_bad_example_label=BAD_EXAMPLE_LABEL in label_names,
        )

    def run_jsonl(self, *, input_path: Path, output_path: Path, **kwargs: Any) -> int:
        """Stream a local JSONL file, score it, save results, and resume."""
        if input_path.resolve() == output_path.resolve():
            raise ValueError("input_path and output_path must be different files")
        return self._run_documents(
            documents=stream_jsonl(input_path), output_path=output_path, **kwargs
        )

    def run_dataset(
        self,
        *,
        dataset_name: str,
        output_path: Path,
        dataset_config: str | None = None,
        dataset_split: str = "train",
        **kwargs: Any,
    ) -> int:
        """Stream a HuggingFace dataset, score it, save results, and resume."""
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
            **kwargs,
        )

    def _process_batch(
        self,
        documents: Sequence[tuple[int, InputDocument]],
        store: JsonlResultStore[SavedDistributionRecord],
        labels: Sequence[LabelDefinition],
        criteria: dict[str, str],
        max_batch_tokens: int,
        max_document_tokens: int,
        max_document_chars: int | None,
        language: str,
        label_type: str,
    ) -> int:
        """Score one batch of documents and append the results to the store."""
        encoded = []
        for _, document in documents:
            # Truncate here, as the chat-model pipeline does: Clef itself would cut
            # an oversized page silently, without a marker.
            if max_document_chars is not None:
                document_text = truncate_document_chars(document.text, max_document_chars)
            else:
                document_text = truncate_document_text(
                    self.tokenizer, document.text, max_document_tokens
                )
            url = document_url(document)
            instructions = self._instructions(label_type, language, labels, has_url=bool(url))
            record = self._encode(
                build_clef_record(render_state(document_text, url), label_type, instructions, criteria)
            )
            # Clef cuts the page to fit max_model_len; an input that fills the whole
            # context may have been cut.
            if len(record.input_ids) >= self.max_model_len:
                raise ValueError(
                    f"Input for {document.doc_id!r} fills max_model_len={self.max_model_len}, "
                    "so Clef may have cut the page; lower max_document_tokens (or "
                    "max_document_chars) or raise max_model_len"
                )
            encoded.append(record)

        scored: list[tuple[tuple[str, ...], list[float]] | None] = [None] * len(encoded)
        start_time = time.time()
        for indices in pack_batches([len(record.input_ids) for record in encoded], max_batch_tokens):
            for index, result in zip(indices, self._score([encoded[index] for index in indices]), strict=True):
                scored[index] = result
        log_model_throughput(
            input_tokens=sum(len(record.input_ids) for record in encoded),
            generated_tokens=0,
            elapsed_seconds=time.time() - start_time,
        )

        records = []
        for (_, document), result in zip(documents, scored, strict=True):
            if result is None:
                raise RuntimeError(f"{document.doc_id!r} was not scored")
            option_ids, logits = result
            records.append(build_distribution_record(document, option_ids, logits, labels))
        store.append(documents, records)
        return len(records)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for Clef label distribution scoring."""
    parser = build_common_arg_parser(
        description=(
            "Score local JSONL or streamed HuggingFace web documents with "
            "probability distributions over topic or format labels using Clef."
        ),
        thinking_modes=THINKING_MODES,
        thinking_mode_help="Only 'disabled' is supported; Clef has no thinking mode.",
    )
    parser.add_argument(
        "--revision",
        dest="model_revision",
        help=(
            "Commit of the model repository to load. Default: the commit whose "
            f"model code was reviewed ({REVIEWED_REVISION[:12]}). Not used with a "
            "local model directory."
        ),
    )
    parser.add_argument("--device", help="Torch device. Default: cuda.")
    parser.add_argument(
        "--max-batch-tokens",
        type=int,
        help=(
            "Maximum padded tokens (documents x longest document) per forward "
            f"pass. Default: {DEFAULT_MAX_BATCH_TOKENS}."
        ),
    )
    parser.add_argument(
        "--max-document-chars",
        type=int,
        help=(
            "Truncate documents to this many characters and append "
            f"'{TRUNCATION_MARKER}', as WebOrganizer does, instead of using "
            "--max-document-tokens. Unset by default."
        ),
    )
    parser.add_argument(
        "--stall-timeout-minutes",
        type=float,
        help=(
            "Exit with a traceback if no batch completes for this many minutes, so "
            "a hung job frees its GPUs; resubmit to resume. 0 disables it. "
            f"Default: {DEFAULT_STALL_TIMEOUT_MINUTES:g}."
        ),
    )
    return parser


def _merge_cli_and_config(
    parser: argparse.ArgumentParser, argv: Sequence[str] | None
) -> argparse.Namespace:
    """Parse YAML defaults first, then let explicit CLI values override them."""
    args = parse_cli_and_config(
        parser,
        argv,
        thinking_modes=THINKING_MODES,
        extra_defaults={
            "model_name": DEFAULT_MODEL,
            "max_model_len": DEFAULT_MAX_MODEL_LEN,
            "batch_size": DEFAULT_BATCH_SIZE,
            "max_batch_tokens": DEFAULT_MAX_BATCH_TOKENS,
            "device": "cuda",
            "stall_timeout_minutes": DEFAULT_STALL_TIMEOUT_MINUTES,
        },
        extra_run_keys={
            "max_document_chars": "max_document_chars",
            "max_batch_tokens": "max_batch_tokens",
            "stall_timeout_minutes": "stall_timeout_minutes",
        },
        extra_model_keys={"revision": "model_revision", "device": "device"},
        unsupported_run_keys={
            "prompt_order": "Clef presents the labels in a fixed order",
            "max_tokens": "Clef scores the labels directly and generates no text",
        },
    )
    if args.max_document_chars is not None and args.max_document_chars < 1:
        parser.error("--max-document-chars must be >= 1")
    if args.max_batch_tokens < 1:
        parser.error("--max-batch-tokens must be >= 1")
    if args.stall_timeout_minutes < 0:
        parser.error("--stall-timeout-minutes must be >= 0 (0 disables the watchdog)")
    # A local directory has no revision; a repository defaults to the reviewed commit.
    if args.model_revision is None and not Path(args.model_name).is_dir():
        args.model_revision = REVIEWED_REVISION
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Run Clef label distribution scoring from CLI and YAML configuration."""
    parser = build_arg_parser()
    args = _merge_cli_and_config(parser, argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    LOGGER.info("Run args:")
    for key, value in vars(args).items():
        LOGGER.info("  %s: %s", key, value)
    # Shared config keys that only apply to vLLM.
    if args.tensor_parallel_size != 1:
        LOGGER.warning(
            "tensor_parallel_size=%d is ignored: Clef runs on one GPU",
            args.tensor_parallel_size,
        )
    LOGGER.info("gpu_memory_utilization and the label/example order seed have no effect for Clef")

    # The examples file is validated with the labels but not used: Clef takes no demonstrations.
    labels, _ = load_label_resources(args.labels_path, args.examples_path, args.label_type)

    # Check if we already have enough completed documents before loading the model
    if args.max_documents is not None:
        completed_results = open_result_store(args.output_path, labels, FIXED_PROMPT_ORDER)
        completed_count = len(completed_results.completed)
        if completed_count >= args.max_documents:
            LOGGER.info(
                "Already have %d completed documents; max_documents=%d reached "
                "before loading the model. Terminating early.",
                completed_count,
                args.max_documents,
            )
            return 0

    client = ClefModelClient(
        model_name=args.model_name,
        revision=args.model_revision,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        seed=args.seed,
        device=args.device,
        thinking_mode=args.thinking_mode,
    )
    common_kwargs = {
        "output_path": args.output_path,
        "labels": labels,
        "batch_size": args.batch_size,
        "max_batch_tokens": args.max_batch_tokens,
        "max_document_tokens": args.max_document_tokens,
        "max_document_chars": args.max_document_chars,
        "max_documents": args.max_documents,
        "language": args.language,
        "label_type": args.label_type,
        "stall_timeout_seconds": (
            args.stall_timeout_minutes * 60 if args.stall_timeout_minutes > 0 else None
        ),
    }
    if args.input_jsonl is not None:
        client.run_jsonl(input_path=args.input_jsonl, **common_kwargs)
    else:
        client.run_dataset(
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
