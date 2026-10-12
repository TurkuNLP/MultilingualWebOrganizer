"""Label documents with probability distributions over topic or format labels.

Instead of generating hard multi-label answers (see ``generate_multilabel.py``), this
pipeline presents each label under a single-letter answer ID and reads the
model's next-token log-probabilities over those letters. The per-answer scores
are normalized into a probability distribution in the same way as the
WebOrganizer template (``original_template.py``).

The log-probabilities are obtained in a single forward pass per document:
vLLM is started with ``logprobs_mode="processed_logprobs"`` and every request
restricts the next token to the answer letters with ``allowed_token_ids``. Each
returned distribution is checked to cover exactly the answer tokens and to sum
to one, so a vLLM behavior change fails loudly instead of producing silently
wrong probabilities.

With ``prompt_order="rotated"``, documents share one of as many label and
example orders as there are labels, so the prompt before the document repeats
and vLLM's prefix cache reuses it. The fixed instructions, labels, and examples are most of each prompt,
so this cuts compute substantially without changing the prompt itself.
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from vllm import SamplingParams  # type: ignore
from vllm.inputs import TokensPrompt  # type: ignore

try:
    from .distribution_prompts import (
        ANSWER_IDS,
        BAD_EXAMPLE_LABEL,
        ChatMessage,
        OTHER_LABEL,
        PromptResources,
        format_prompt_resources,
        get_format_distribution_prompt,
        get_topic_distribution_prompt,
    )
    from .distribution_records import (
        PROBABILITY_SUM_TOLERANCE,
        SavedDistributionRecord,
        compute_label_distribution,
        logsumexp,
        open_result_store,
    )
    from .labeling_common import (
        InputDocument,
        JsonlResultStore,
        LabelDefinition,
        LabelExample,
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
        PROMPT_ORDER_MODES,
        prompt_order_for_document,
        stream_huggingface_dataset,
        stream_jsonl,
        thinking_chat_template_kwargs,
        TRUNCATION_MARKER,
        truncate_document_chars,
        truncate_document_text,
    )
except ImportError:
    from distribution_prompts import (
        ANSWER_IDS,
        BAD_EXAMPLE_LABEL,
        ChatMessage,
        OTHER_LABEL,
        PromptResources,
        format_prompt_resources,
        get_format_distribution_prompt,
        get_topic_distribution_prompt,
    )
    from distribution_records import (
        PROBABILITY_SUM_TOLERANCE,
        SavedDistributionRecord,
        compute_label_distribution,
        logsumexp,
        open_result_store,
    )
    from labeling_common import (
        InputDocument,
        JsonlResultStore,
        LabelDefinition,
        LabelExample,
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
        PROMPT_ORDER_MODES,
        prompt_order_for_document,
        stream_huggingface_dataset,
        stream_jsonl,
        thinking_chat_template_kwargs,
        TRUNCATION_MARKER,
        truncate_document_chars,
        truncate_document_text,
    )

LOGGER = logging.getLogger(__name__)

# The answer letter must be the first generated token, so thinking stays off.
THINKING_MODES = ("disabled",)


class LogprobOutputError(RuntimeError):
    """The model returned log-probabilities that violate the scoring contract."""


class LogprobModelClient:
    """vLLM client that scores the answer letters of each document prompt."""

    def __init__(
        self,
        *,
        model_name: str,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        dtype: str,
        max_model_len: int,
        seed: int,
        thinking_mode: str = "disabled",
    ) -> None:
        """Load the model and resolve the token ID of each answer letter.

        Args:
            model_name: HuggingFace model name or path.
            tensor_parallel_size: Number of GPUs for tensor parallelism.
            gpu_memory_utilization: Fraction of GPU memory vLLM may use.
            dtype: Model dtype passed to vLLM.
            max_model_len: Maximum prompt plus answer length in tokens.
            seed: Seed for vLLM and the per-document prompt shuffles.
            thinking_mode: Must be ``"disabled"``; the answer letter has to be
                the first generated token.

        Raises:
            ValueError: If thinking is not disabled or the answer letters are
                not single tokens for this tokenizer.
        """
        if thinking_mode not in THINKING_MODES:
            raise ValueError(
                "thinking_mode must be 'disabled': the answer letter must be the "
                "first generated token"
            )
        self.model_name = model_name
        self.seed = seed
        self.thinking_mode = thinking_mode

        self.llm = build_llm(
            model_name=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            max_model_len=max_model_len,
            seed=seed,
            # Return log-probabilities after allowed_token_ids masking, so the
            # returned top-k covers exactly the answer letters.
            logprobs_mode="processed_logprobs",
            max_logprobs=len(ANSWER_IDS),
            # Reuses the shared prompt prefix; effective with prompt_order="rotated".
            enable_prefix_caching=True,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.max_model_len = int(self.llm.model_config.max_model_len)
        self.answer_token_ids = self._resolve_answer_token_ids()

    @property
    def chat_template_kwargs(self) -> dict[str, Any]:
        """Return chat template kwargs that disable thinking."""
        return thinking_chat_template_kwargs(self.model_name, self.thinking_mode) or {}

    def _encode(self, text: str) -> list[int]:
        """Tokenize already-rendered text without adding special tokens."""
        input_ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.tolist()
        if not isinstance(input_ids, list) or not all(
            isinstance(token_id, int) for token_id in input_ids
        ):
            raise TypeError(
                f"Tokenizer returned unsupported input_ids type {type(input_ids)!r}"
            )
        return input_ids

    def _uses_mistral_common(self) -> bool:
        """Return whether the tokenizer is Mistral's ``mistral-common`` backend.

        Its chat template renders control tokens such as ``[INST]`` as plain
        text, and encoding that text again turns them into ordinary text tokens,
        so rendered text must never be re-encoded for these models.
        """
        return "mistral" in type(self.tokenizer).__name__.lower()

    def _render_chat(self, messages: Sequence[ChatMessage]) -> str:
        """Render chat messages with the generation prompt as text.

        Only used to verify tokenization for Hugging Face tokenizers; prompts are
        built from token IDs with :meth:`_chat_token_ids`.
        """
        rendered = self.tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=True,
            **self.chat_template_kwargs,
        )
        if not isinstance(rendered, str):
            raise TypeError(
                "Tokenizer apply_chat_template(tokenize=False) returned "
                f"{type(rendered).__name__}, expected str"
            )
        return rendered

    def _chat_token_ids(self, messages: Sequence[ChatMessage]) -> list[int]:
        """Return the prompt token IDs, ending with the generation prompt.

        The chat template tokenizes directly (``tokenize=True``), so control
        tokens become their real token IDs for every tokenizer, including
        Mistral's, where rendering to text and re-encoding would not.
        """
        tokenized = self.tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=True,
            **self.chat_template_kwargs,
        )
        # Depending on the tokenizer, the result is a list, a tensor, or a
        # BatchEncoding/dict holding "input_ids".
        if isinstance(tokenized, Mapping):
            if "input_ids" not in tokenized:
                raise TypeError("apply_chat_template returned a mapping without input_ids")
            tokenized = tokenized["input_ids"]
        if hasattr(tokenized, "tolist"):
            tokenized = tokenized.tolist()
        if isinstance(tokenized, list) and len(tokenized) == 1 and isinstance(tokenized[0], list):
            tokenized = tokenized[0]
        if not isinstance(tokenized, list) or not all(isinstance(t, int) for t in tokenized):
            raise TypeError(
                f"apply_chat_template(tokenize=True) returned unsupported type {type(tokenized)!r}"
            )
        return tokenized

    def _resolve_answer_token_ids(self) -> tuple[int, ...]:
        """Find the token each answer letter encodes to after the generation prompt.

        Mirrors the template's check that every answer is scored on its own
        tokens: each letter must be a single, distinct token. The demonstrated
        answer format is the letter, a newline, and an explanation, so the letter
        must also stay a separate token when a newline follows; otherwise the
        model's natural first token would be a merged token, and scoring the bare
        letter could bias letters unevenly.

        For Hugging Face tokenizers the checks also run in context: the rendered
        prompt must re-encode to exactly the prompt token IDs, and appending a
        letter must add exactly that one token. For Mistral's tokenizer the
        rendered text can't be re-encoded (see :meth:`_uses_mistral_common`);
        there the generation prompt ends with a control token, after which the
        letter starts a new text segment, so the checks on the letter alone apply.

        Returns:
            One token ID per entry in ``ANSWER_IDS``.

        Raises:
            ValueError: If a letter is not a single, distinct token, or if the
                rendered text and the token IDs disagree.
        """
        probe = [{"role": "user", "content": "probe"}]
        prompt_ids = self._chat_token_ids(probe)
        in_context = not self._uses_mistral_common()
        if in_context:
            prompt_text = self._render_chat(probe)
            if self._encode(prompt_text) != prompt_ids:
                raise ValueError(
                    f"The rendered chat prompt of {self.model_name!r} does not "
                    "re-encode to the chat template's token IDs"
                )
        else:
            LOGGER.info("Mistral tokenizer: checking answer letters outside the rendered prompt")
        # In context, the letter is encoded after the rendered prompt, so tokenizers
        # that mark word starts (e.g. SentencePiece) give the token the model would
        # actually generate; outside it (Mistral), the letter is encoded on its own.
        prefix = prompt_text if in_context else ""
        prefix_ids = prompt_ids if in_context else []
        answer_token_ids = []
        for answer_id in ANSWER_IDS:
            full_ids = self._encode(prefix + answer_id)
            if len(full_ids) != len(prefix_ids) + 1 or full_ids[: len(prefix_ids)] != prefix_ids:
                raise ValueError(
                    f"Answer ID {answer_id!r} does not encode to a single token "
                    f"after the generation prompt of {self.model_name!r}"
                )
            answer_ids = self._encode(prefix + answer_id + "\nThe page is about this category.")
            if (
                answer_ids[: len(prefix_ids)] != prefix_ids
                or len(answer_ids) <= len(prefix_ids)
                or answer_ids[len(prefix_ids)] != full_ids[-1]
            ):
                raise ValueError(
                    f"Answer ID {answer_id!r} merges with the following newline "
                    f"in the tokenizer of {self.model_name!r}; the scored token "
                    "would differ from the demonstrated answer format"
                )
            answer_token_ids.append(full_ids[-1])
        if len(set(answer_token_ids)) != len(answer_token_ids):
            raise ValueError("Answer IDs must encode to distinct tokens")
        LOGGER.info(
            "Answer token IDs: %s",
            dict(zip(ANSWER_IDS, answer_token_ids)),
        )
        return tuple(answer_token_ids)

    def _run_documents(
        self,
        *,
        documents: Iterable[tuple[int, InputDocument]],
        output_path: Path,
        labels: Sequence[LabelDefinition],
        examples: Sequence[LabelExample],
        batch_size: int = 32,
        max_document_tokens: int = 8192,
        max_document_chars: int | None = None,
        prompt_order: str = "random",
        max_documents: int | None = None,
        language: str = "English",
        label_type: str = "topic",
    ) -> int:
        """Process validated documents, save results, and resume safely."""
        stable_ids = [label.id for label in labels]
        if not stable_ids or len(stable_ids) != len(set(stable_ids)):
            raise ValueError("labels must contain unique stable IDs")
        if len(labels) > len(ANSWER_IDS):
            raise ValueError(
                f"At most {len(ANSWER_IDS)} labels are supported; got {len(labels)}"
            )
        if not examples:
            raise ValueError("examples must contain at least one demonstration")
        if max_document_tokens < 1:
            raise ValueError("max_document_tokens must be >= 1")
        if max_document_chars is not None and max_document_chars < 1:
            raise ValueError("max_document_chars must be >= 1")
        if prompt_order not in PROMPT_ORDER_MODES:
            raise ValueError(f"prompt_order must be one of {PROMPT_ORDER_MODES}")
        language = normalize_language(language)
        label_type = normalize_label_type(label_type)
        short_names = [label.short_name for label in labels]
        if len(short_names) != len(set(short_names)):
            raise ValueError("labels must contain unique short names")
        store = open_result_store(output_path, labels, prompt_order)
        return process_documents_in_batches(
            documents=documents,
            store=store,
            process_batch=lambda batch: self._process_batch(
                batch,
                store,
                labels,
                examples,
                max_document_tokens,
                max_document_chars,
                prompt_order,
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
        batch_size: int = 32,
        max_document_tokens: int = 8192,
        max_document_chars: int | None = None,
        prompt_order: str = "random",
        max_documents: int | None = None,
        language: str = "English",
        label_type: str = "topic",
    ) -> int:
        """Stream a local JSONL file, score it, save results, and resume."""
        if input_path.resolve() == output_path.resolve():
            raise ValueError("input_path and output_path must be different files")
        return self._run_documents(
            documents=stream_jsonl(input_path),
            output_path=output_path,
            labels=labels,
            examples=examples,
            batch_size=batch_size,
            max_document_tokens=max_document_tokens,
            max_document_chars=max_document_chars,
            prompt_order=prompt_order,
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
        batch_size: int = 32,
        max_document_tokens: int = 8192,
        max_document_chars: int | None = None,
        prompt_order: str = "random",
        max_documents: int | None = None,
        language: str = "English",
        label_type: str = "topic",
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
            labels=labels,
            examples=examples,
            batch_size=batch_size,
            max_document_tokens=max_document_tokens,
            max_document_chars=max_document_chars,
            prompt_order=prompt_order,
            max_documents=max_documents,
            language=language,
            label_type=label_type,
        )

    def _process_batch(
        self,
        documents: Sequence[tuple[int, InputDocument]],
        store: JsonlResultStore[SavedDistributionRecord],
        labels: Sequence[LabelDefinition],
        examples: Sequence[LabelExample],
        max_document_tokens: int,
        max_document_chars: int | None,
        prompt_order: str,
        language: str,
        label_type: str,
    ) -> int:
        """Score one batch of documents and append the results to the store."""
        prompt_function = (
            get_topic_distribution_prompt
            if label_type == "topic"
            else get_format_distribution_prompt
        )
        answer_token_ids = self.answer_token_ids[: len(labels)]
        # The prompt explains the fallback labels only if the taxonomy has them.
        label_names = {label.name for label in labels}
        has_other_label = OTHER_LABEL in label_names
        has_bad_example_label = BAD_EXAMPLE_LABEL in label_names
        prompts: list[TokensPrompt] = []
        resources: list[PromptResources] = []
        for _, document in documents:
            if max_document_chars is not None:
                document_text = truncate_document_chars(
                    document.text, max_document_chars
                )
            else:
                document_text = truncate_document_text(
                    self.tokenizer, document.text, max_document_tokens
                )
            label_order, example_order = prompt_order_for_document(
                prompt_order, len(labels), len(examples), self.seed, document.doc_id
            )
            document_resources = format_prompt_resources(
                labels, examples, label_order, example_order
            )
            messages = prompt_function(
                text=document_text,
                url=document_url(document),
                labels=document_resources.label_text,
                examples=document_resources.examples,
                language=language,
                has_other_label=has_other_label,
                has_bad_example_label=has_bad_example_label,
            )
            prompt_token_ids = self._chat_token_ids(messages)
            # One position is needed for the scored answer token.
            if len(prompt_token_ids) + 1 > self.max_model_len:
                raise ValueError(
                    f"Prompt for {document.doc_id!r} has {len(prompt_token_ids)} "
                    f"tokens, which does not fit max_model_len={self.max_model_len}; "
                    "lower max_document_tokens (or max_document_chars) or raise "
                    "max_model_len"
                )
            prompts.append(TokensPrompt(prompt_token_ids=prompt_token_ids))
            resources.append(document_resources)

        # Greedy decoding keeps vLLM on the path where processed log-probs are
        # computed after masking but before any temperature scaling.
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            logprobs=len(answer_token_ids),
            allowed_token_ids=list(answer_token_ids),
        )
        generate_start_time = time.time()
        outputs = self.llm.generate(
            prompts,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        generate_end_time = time.time()
        if len(outputs) != len(prompts):
            raise RuntimeError(
                f"vLLM returned {len(outputs)} outputs for {len(prompts)} prompts"
            )
        log_model_throughput(
            input_tokens=sum(len(output.prompt_token_ids) for output in outputs),
            generated_tokens=sum(
                len(completion.token_ids)
                for output in outputs
                for completion in output.outputs
            ),
            elapsed_seconds=generate_end_time - generate_start_time,
        )

        records = [
            self._build_record(
                document=document,
                output=output,
                document_resources=document_resources,
                answer_token_ids=answer_token_ids,
                labels=labels,
                prompt_order=prompt_order,
            )
            for (_, document), output, document_resources in zip(
                documents, outputs, resources
            )
        ]
        store.append(documents, records)
        return len(records)

    def _build_record(
        self,
        *,
        document: InputDocument,
        output: Any,
        document_resources: PromptResources,
        answer_token_ids: Sequence[int],
        labels: Sequence[LabelDefinition],
        prompt_order: str,
    ) -> SavedDistributionRecord:
        """Convert one vLLM output into a validated distribution record.

        Args:
            document: The scored input document.
            output: The vLLM ``RequestOutput`` for the document.
            document_resources: Prompt resources used for the document.
            answer_token_ids: Token IDs of the presented answer letters.
            labels: Label definitions in label-file order.
            prompt_order: Ordering scheme stored in the record.

        Returns:
            The label distribution record for the document.

        Raises:
            LogprobOutputError: If vLLM did not return a normalized distribution
                over exactly the answer tokens.
        """
        doc_id = document.doc_id
        if len(output.outputs) != 1:
            raise LogprobOutputError(
                f"{doc_id!r}: expected exactly one completion, "
                f"got {len(output.outputs)}"
            )
        completion = output.outputs[0]
        if completion.logprobs is None or len(completion.logprobs) != 1:
            raise LogprobOutputError(
                f"{doc_id!r}: expected log-probabilities for exactly one "
                f"generated token, got {completion.logprobs!r}"
            )
        position = completion.logprobs[0]
        if not isinstance(position, Mapping):
            raise LogprobOutputError(
                f"{doc_id!r}: unsupported log-probability container "
                f"{type(position).__name__}"
            )
        returned = {
            int(token_id): float(logprob.logprob)
            for token_id, logprob in position.items()
        }
        if set(returned) != set(answer_token_ids):
            raise LogprobOutputError(
                f"{doc_id!r}: log-probabilities cover tokens {sorted(returned)!r}, "
                f"expected exactly the answer tokens {sorted(answer_token_ids)!r}. "
                "vLLM must apply allowed_token_ids before computing "
                "processed_logprobs."
            )
        # Log-probabilities in presented answer order: A, B, C, ...
        answer_logprobs = [returned[token_id] for token_id in answer_token_ids]
        if not all(math.isfinite(value) for value in answer_logprobs):
            raise LogprobOutputError(
                f"{doc_id!r}: non-finite answer log-probabilities {answer_logprobs!r}"
            )
        log_mass = logsumexp(answer_logprobs)
        if abs(math.expm1(log_mass)) > PROBABILITY_SUM_TOLERANCE:
            raise LogprobOutputError(
                f"{doc_id!r}: answer probabilities sum to {math.exp(log_mass)!r}, "
                "expected 1. vLLM must return processed_logprobs normalized over "
                "the allowed answer tokens."
            )
        normalized = compute_label_distribution(answer_logprobs)

        # Undo the per-document shuffle, as the template does with
        # choice_loss[permutation] = permuted_choice_loss.
        stable_to_short = {label.id: label.short_name for label in labels}
        presented_short_names = [
            stable_to_short[document_resources.answer_to_stable[answer_id]]
            for answer_id in ANSWER_IDS[: len(labels)]
        ]
        logprob_by_name = dict(zip(presented_short_names, normalized))
        label_logprobs = {
            label.short_name: logprob_by_name[label.short_name] for label in labels
        }
        label_probs = {name: math.exp(value) for name, value in label_logprobs.items()}
        # First maximum in label-file order, matching np.argmax in the template.
        top_label = max(label_probs, key=label_probs.__getitem__)
        return SavedDistributionRecord(
            id=doc_id,
            text=document.text,
            label=top_label,
            label_prob=label_probs[top_label],
            label_probs=label_probs,
            label_logprobs=label_logprobs,
            label_order=presented_short_names,
            example_order=list(document_resources.example_order),
            prompt_order=prompt_order,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for label distribution scoring."""
    parser = build_common_arg_parser(
        description=(
            "Score local JSONL or streamed HuggingFace web documents with "
            "probability distributions over topic or format labels."
        ),
        thinking_modes=THINKING_MODES,
        thinking_mode_help=(
            "Only 'disabled' is supported; the answer must be the first token."
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
        "--prompt-order",
        choices=PROMPT_ORDER_MODES,
        help=(
            "'random' shuffles labels and examples per document; 'rotated' "
            "uses 24 balanced rotations so vLLM can reuse the shared prompt "
            "prefix. Default: random."
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
        extra_defaults={"prompt_order": "random"},
        extra_run_keys={
            "max_document_chars": "max_document_chars",
            "prompt_order": "prompt_order",
        },
        unsupported_run_keys={
            "max_tokens": (
                "the label distribution pipeline always scores exactly one "
                "answer token"
            )
        },
    )
    if args.max_document_chars is not None and args.max_document_chars < 1:
        parser.error("--max-document-chars must be >= 1")
    # Config values bypass argparse choices, so check prompt_order explicitly.
    if args.prompt_order not in PROMPT_ORDER_MODES:
        parser.error(f"prompt_order must be one of {list(PROMPT_ORDER_MODES)}")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Run label distribution scoring from CLI and YAML configuration."""
    parser = build_arg_parser()
    args = _merge_cli_and_config(parser, argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    LOGGER.info("Run args:")
    for key, value in vars(args).items():
        LOGGER.info("  %s: %s", key, value)

    labels, examples = load_label_resources(
        args.labels_path, args.examples_path, args.label_type
    )

    # Check if we already have enough completed documents before loading the model
    if args.max_documents is not None:
        completed_results = open_result_store(
            args.output_path, labels, args.prompt_order
        )
        completed_count = len(completed_results.completed)
        if completed_count >= args.max_documents:
            LOGGER.info(
                "Already have %d completed documents; max_documents=%d reached "
                "before loading the model. Terminating early.",
                completed_count,
                args.max_documents,
            )
            return 0

    client = LogprobModelClient(
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
        "labels": labels,
        "examples": examples,
        "batch_size": args.batch_size,
        "max_document_tokens": args.max_document_tokens,
        "max_document_chars": args.max_document_chars,
        "prompt_order": args.prompt_order,
        "max_documents": args.max_documents,
        "language": args.language,
        "label_type": args.label_type,
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
