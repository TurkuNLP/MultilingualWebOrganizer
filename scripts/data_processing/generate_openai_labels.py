"""Label a sample of documents with an OpenAI model through the Chat Completions API.

Frontier OpenAI reasoning models don't return log-probabilities, so this script
records one **hard label** per document: the letter the model answers with,
plus the explanation it writes after it. Everything else matches
``generate_label_distributions.py``: the same prompts (instructions, examples,
letter IDs), the same label and example order for a given ``--seed`` and
``--prompt-order``, and the same resumable JSONL output handling. Documents
labelled here can therefore be compared one to one with the open models' runs.

The script needs no GPU, vLLM or ``openai`` package; it uses the standard
library HTTP client. The API key is read from the ``OPENAI_API_KEY`` environment
variable and is never logged. Requests set ``store: false``.

Documents are truncated by characters (default 50,000, with WebOrganizer's
truncation marker), because the OpenAI tokenizer isn't available here. English
Nemotron-CC documents are much shorter than that, so truncation rarely applies.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator  # type: ignore

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
    from .labeling_common import (
        PROMPT_ORDER_MODES,
        InputDocument,
        JsonlResultStore,
        LabelDefinition,
        LabelExample,
        SavedRecordBase,
        document_url,
        exit_without_finalization,
        load_label_resources,
        normalize_label_type,
        normalize_language,
        process_documents_in_batches,
        prompt_order_for_document,
        stream_huggingface_dataset,
        stream_jsonl,
        truncate_document_chars,
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
    from labeling_common import (
        PROMPT_ORDER_MODES,
        InputDocument,
        JsonlResultStore,
        LabelDefinition,
        LabelExample,
        SavedRecordBase,
        document_url,
        exit_without_finalization,
        load_label_resources,
        normalize_label_type,
        normalize_language,
        process_documents_in_batches,
        prompt_order_for_document,
        stream_huggingface_dataset,
        stream_jsonl,
        truncate_document_chars,
    )

LOGGER = logging.getLogger(__name__)

DEFAULT_API_BASE = "https://api.openai.com/v1"
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high")
# HTTP statuses worth retrying: timeouts, conflicts, rate limits, server errors.
RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})
# How often an answer that doesn't start with a valid letter is asked again.
ANSWER_RETRIES = 2
# The answer must start with one capital letter, optionally wrapped in Markdown
# emphasis, followed by the end of the text, a line break, or punctuation
# ("B", "B\n...", "**B**", "B: ..."). A letter followed by a space and more text
# is rejected, so sentences like "I think ..." or "A good example ..." are not
# mistaken for the answers I or A.
ANSWER_PATTERN = re.compile(r"\s*[*_`]*([A-Z])[*_`]*(?:[.:)\]]|[ \t]*(?:\n|$))")
SCRIPT_DIR = Path(__file__).resolve().parent


class ApiRequestError(RuntimeError):
    """The API request failed and should not be retried."""


class AnswerFormatError(ValueError):
    """The model's answer does not start with a valid letter ID."""


class SavedApiLabelRecord(SavedRecordBase):
    """Flat JSONL output record with one hard label from an API model.

    Attributes:
        id: Document ID.
        text: Original, untruncated document text.
        url: URL of the page as shown to the model in the prompt, or ``None``
            if the document has no URL (then the prompt has none either).
        label: Short name of the chosen label.
        answer_letter: Letter the model answered with.
        explanation: Text the model wrote after the letter.
        label_order: Label short names in the order they were presented, i.e. as
            answers ``A``, ``B``, ``C``, ...
        example_order: Indices into the examples file in presentation order.
        prompt_order: Ordering scheme, ``"random"`` or ``"rotated"``.
        model: Model requested.
        response_model: Model the API reports having used (a dated snapshot).
        finish_reason: Why generation stopped.
        prompt_tokens: Prompt tokens billed.
        cached_prompt_tokens: Prompt tokens served from OpenAI's prompt cache.
        completion_tokens: Completion tokens billed, including reasoning tokens.
        reasoning_tokens: Hidden reasoning tokens, if the model reasons.
    """

    text: str = ""
    url: str | None = Field(default=None, min_length=1)
    label: str = Field(min_length=1)
    answer_letter: str = Field(min_length=1, max_length=1)
    explanation: str = ""
    label_order: list[str] = Field(min_length=1)
    example_order: list[int]
    prompt_order: str
    model: str = Field(min_length=1)
    response_model: str = Field(min_length=1)
    finish_reason: str
    prompt_tokens: int = Field(ge=0)
    cached_prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    reasoning_tokens: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_answer(self) -> "SavedApiLabelRecord":
        """Check that the answer letter, label, and orders are consistent."""
        if len(set(self.label_order)) != len(self.label_order):
            raise ValueError("label_order must not contain duplicates")
        if self.answer_letter not in ANSWER_IDS[: len(self.label_order)]:
            raise ValueError(
                f"answer_letter {self.answer_letter!r} is not a shown letter"
            )
        if self.label_order[ANSWER_IDS.index(self.answer_letter)] != self.label:
            raise ValueError("label must be the label shown under answer_letter")
        if sorted(self.example_order) != list(range(len(self.example_order))):
            raise ValueError("example_order must be a permutation of example indices")
        if self.prompt_order not in PROMPT_ORDER_MODES:
            raise ValueError(f"Unknown prompt_order {self.prompt_order!r}")
        return self


def api_record_validator(
    label_names: set[str], prompt_order: str, model: str
) -> Callable[[SavedApiLabelRecord], None]:
    """Build a store check that records match this run's labels, order, and model.

    One output file must never mix label sets, ordering schemes, or models.
    """

    def validate(record: SavedApiLabelRecord) -> None:
        if set(record.label_order) != label_names:
            raise ValueError(f"Result for {record.id!r} has a different label set")
        if record.prompt_order != prompt_order:
            raise ValueError(
                f"Result for {record.id!r} used prompt_order={record.prompt_order!r}, "
                f"but this run uses {prompt_order!r}; use a new output path"
            )
        if record.model != model:
            raise ValueError(
                f"Result for {record.id!r} came from model {record.model!r}, but this "
                f"run uses {model!r}; use a new output path"
            )

    return validate


def open_result_store(
    output_path: Path, labels: Sequence[LabelDefinition], prompt_order: str, model: str
) -> JsonlResultStore[SavedApiLabelRecord]:
    """Open the API result store for the given labels, ordering, and model."""
    return JsonlResultStore(
        output_path,
        SavedApiLabelRecord,
        api_record_validator(
            {label.short_name for label in labels}, prompt_order, model
        ),
    )


def parse_answer(content: str, num_labels: int) -> tuple[str, str]:
    """Split a model answer into its letter ID and the explanation after it.

    Args:
        content: The model's message text.
        num_labels: Number of labels shown; valid letters are the first ones.

    Returns:
        The letter and the stripped explanation (possibly empty).

    Raises:
        AnswerFormatError: If the answer doesn't start with a shown letter.
    """
    match = ANSWER_PATTERN.match(content)
    if match is None or match.group(1) not in ANSWER_IDS[:num_labels]:
        raise AnswerFormatError(
            f"answer does not start with a letter A-{ANSWER_IDS[num_labels - 1]}: {content[:200]!r}"
        )
    return match.group(1), content[match.end() :].strip()


class OpenAIChatClient:
    """Minimal Chat Completions client with retries, using only the standard library."""

    def __init__(
        self,
        *,
        api_base: str,
        api_key: str,
        model: str,
        reasoning_effort: str | None,
        max_completion_tokens: int,
        timeout: float,
        max_retries: int,
    ) -> None:
        """Store the request settings.

        Args:
            api_base: API base URL, e.g. ``https://api.openai.com/v1``.
            api_key: Secret API key.
            model: Model name.
            reasoning_effort: Reasoning effort to request, or ``None`` for the
                model's default.
            max_completion_tokens: Cap on generated tokens, including hidden
                reasoning tokens.
            timeout: Seconds to wait for one response.
            max_retries: Retries for rate limits, server errors, and network errors.
        """
        self.url = api_base.rstrip("/") + "/chat/completions"
        self._api_key = api_key
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_completion_tokens = max_completion_tokens
        self.timeout = timeout
        self.max_retries = max_retries

    def complete(self, messages: Sequence[ChatMessage]) -> dict[str, Any]:
        """Send one chat completion request and return the parsed JSON response.

        Raises:
            ApiRequestError: For non-retryable errors (e.g. invalid request,
                authentication), or when retries are exhausted.
        """
        body: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "max_completion_tokens": self.max_completion_tokens,
            # Don't keep the completion on OpenAI's platform for later retrieval.
            "store": False,
        }
        if self.reasoning_effort is not None:
            body["reasoning_effort"] = self.reasoning_effort
        payload = json.dumps(body).encode("utf-8")
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(
                self.url,
                data=payload,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.load(response)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                if exc.code not in RETRYABLE_STATUS or attempt == self.max_retries:
                    raise ApiRequestError(
                        f"HTTP {exc.code} from {self.url}: {detail}"
                    ) from exc
                try:
                    wait = min(60.0, float(exc.headers.get("Retry-After", "")))
                except ValueError:
                    wait = min(60, 2 ** (attempt + 1))
                LOGGER.warning(
                    "HTTP %d; retrying in %.0fs (attempt %d/%d)",
                    exc.code,
                    wait,
                    attempt + 1,
                    self.max_retries,
                )
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                if attempt == self.max_retries:
                    raise ApiRequestError(
                        f"Request to {self.url} failed: {exc}"
                    ) from exc
                wait = min(60, 2 ** (attempt + 1))
                LOGGER.warning(
                    "Network error (%s); retrying in %.0fs (attempt %d/%d)",
                    exc,
                    wait,
                    attempt + 1,
                    self.max_retries,
                )
            time.sleep(wait)
        raise AssertionError("unreachable")


def label_document(
    client: OpenAIChatClient,
    messages: Sequence[ChatMessage],
    num_labels: int,
    doc_id: str,
) -> tuple[str, str, dict[str, Any]]:
    """Ask for one document's label, re-asking if the answer format is wrong.

    Returns:
        The answer letter, the explanation, and the API response.

    Raises:
        AnswerFormatError: If the answer is still invalid after ``ANSWER_RETRIES``.
        ApiRequestError: If the request fails.
    """
    for attempt in range(ANSWER_RETRIES + 1):
        response = client.complete(messages)
        try:
            choice = response["choices"][0]
            content = choice["message"].get("content") or ""
            finish_reason = choice.get("finish_reason")
        except (KeyError, IndexError, TypeError) as exc:
            raise ApiRequestError(
                f"{doc_id!r}: unexpected response structure: {str(response)[:300]}"
            ) from exc
        if not content and finish_reason == "length":
            raise ApiRequestError(
                f"{doc_id!r}: the answer was cut off before any text "
                f"(max_completion_tokens={client.max_completion_tokens}); reasoning "
                "may have used the whole budget. Raise --max-completion-tokens or "
                "lower --reasoning-effort."
            )
        try:
            letter, explanation = parse_answer(content, num_labels)
            return letter, explanation, response
        except AnswerFormatError as exc:
            if attempt == ANSWER_RETRIES:
                raise AnswerFormatError(
                    f"{doc_id!r}: {exc} (after {ANSWER_RETRIES + 1} attempts)"
                ) from exc
            LOGGER.warning("%s: %s; asking again", doc_id, exc)
    raise AssertionError("unreachable")


def process_batch(
    documents: Sequence[tuple[int, InputDocument]],
    *,
    store: JsonlResultStore[SavedApiLabelRecord],
    client: OpenAIChatClient,
    labels: Sequence[LabelDefinition],
    examples: Sequence[LabelExample],
    label_type: str,
    language: str,
    seed: int,
    prompt_order: str,
    max_document_chars: int,
    concurrency: int,
) -> int:
    """Label one batch with concurrent API requests and append it to the store."""
    prompt_function = (
        get_topic_distribution_prompt
        if label_type == "topic"
        else get_format_distribution_prompt
    )
    label_names = {label.name for label in labels}
    stable_to_short = {label.id: label.short_name for label in labels}
    prepared: list[tuple[InputDocument, PromptResources, list[ChatMessage]]] = []
    for _, document in documents:
        label_order, example_order = prompt_order_for_document(
            prompt_order, len(labels), len(examples), seed, document.doc_id
        )
        resources = format_prompt_resources(
            labels, examples, label_order, example_order
        )
        messages = prompt_function(
            text=truncate_document_chars(document.text, max_document_chars),
            url=document_url(document),
            labels=resources.label_text,
            examples=resources.examples,
            language=language,
            has_other_label=OTHER_LABEL in label_names,
            has_bad_example_label=BAD_EXAMPLE_LABEL in label_names,
        )
        prepared.append((document, resources, messages))

    start = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        # map keeps input order and re-raises the first failure, so a failed
        # document stops the run before the batch is written.
        answers = list(
            pool.map(
                lambda item: label_document(
                    client, item[2], len(labels), item[0].doc_id
                ),
                prepared,
            )
        )
    elapsed = time.time() - start

    records = []
    totals = {"prompt": 0, "cached": 0, "completion": 0, "reasoning": 0}
    for (document, resources, _), (letter, explanation, response) in zip(
        prepared, answers
    ):
        usage = response.get("usage") or {}
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        reasoning = (usage.get("completion_tokens_details") or {}).get(
            "reasoning_tokens"
        ) or 0
        presented = [
            stable_to_short[resources.answer_to_stable[a]]
            for a in ANSWER_IDS[: len(labels)]
        ]
        record = SavedApiLabelRecord(
            id=document.doc_id,
            text=document.text,
            # The same value the prompt used, from the same helper.
            url=document_url(document),
            label=stable_to_short[resources.answer_to_stable[letter]],
            answer_letter=letter,
            explanation=explanation,
            label_order=presented,
            example_order=list(resources.example_order),
            prompt_order=prompt_order,
            model=client.model,
            response_model=str(response.get("model") or client.model),
            finish_reason=str(response["choices"][0].get("finish_reason")),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            cached_prompt_tokens=int(cached),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            reasoning_tokens=int(reasoning),
        )
        records.append(record)
        totals["prompt"] += record.prompt_tokens
        totals["cached"] += record.cached_prompt_tokens
        totals["completion"] += record.completion_tokens
        totals["reasoning"] += record.reasoning_tokens
    store.append(documents, records)
    LOGGER.info(
        "Labelled %d documents in %.1fs; tokens: prompt %d (cached %d), completion %d (reasoning %d)",
        len(records),
        elapsed,
        totals["prompt"],
        totals["cached"],
        totals["completion"],
        totals["reasoning"],
    )
    return len(records)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Label a sample of documents with one hard label per document from an "
            "OpenAI model, using the label distribution pipeline's prompts. "
            "Reads the API key from OPENAI_API_KEY."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-jsonl", type=Path, help="Local JSONL input")
    source.add_argument(
        "--input-dataset",
        help="HuggingFace dataset to stream, e.g. MultiSynt/MT-Nemotron-CC",
    )
    parser.add_argument("--dataset-config", help="Dataset config, e.g. eng_Latn")
    parser.add_argument(
        "--dataset-split", default="train", help="Dataset split, e.g. parallel"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output JSONL (resumable)"
    )
    parser.add_argument("--label-type", choices=("topic", "format"), default="topic")
    parser.add_argument(
        "--labels",
        type=Path,
        help="Label YAML; default: topics.yaml or formats.yaml next to this script",
    )
    parser.add_argument(
        "--examples",
        type=Path,
        help="Examples YAML; default: <label type>_examples.yaml next to this script",
    )
    parser.add_argument(
        "--language", default="English", help="Language of the documents"
    )
    parser.add_argument(
        "--model", required=True, help="OpenAI model name, e.g. gpt-5.5"
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=REASONING_EFFORTS,
        help="Reasoning effort to request; omit to use the model's default",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=4096,
        help="Cap on generated tokens per document, including hidden reasoning tokens",
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        required=True,
        help="Number of documents to label (required, to keep API costs bounded)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Documents per saved batch"
    )
    parser.add_argument(
        "--concurrency", type=int, default=4, help="Parallel API requests"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Prompt order seed; use the same seed as the run you compare with (English 0, Finnish 1, ...)",
    )
    parser.add_argument("--prompt-order", choices=PROMPT_ORDER_MODES, default="rotated")
    parser.add_argument(
        "--max-document-chars",
        type=int,
        default=50_000,
        help="Truncate documents to this many characters, with WebOrganizer's truncation marker",
    )
    parser.add_argument("--api-base", default=DEFAULT_API_BASE, help="API base URL")
    parser.add_argument(
        "--request-timeout", type=float, default=300, help="Seconds per request"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=6,
        help="Retries for rate limits and server errors",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default="INFO",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Label documents with an OpenAI model."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    for name in (
        "max_completion_tokens",
        "max_documents",
        "batch_size",
        "concurrency",
        "max_document_chars",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be >= 1")
    if args.max_retries < 0 or args.seed < 0:
        parser.error("--max-retries and --seed must be >= 0")
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        parser.error("set the OPENAI_API_KEY environment variable")
    label_type = normalize_label_type(args.label_type)
    language = normalize_language(args.language)
    labels_path = args.labels or SCRIPT_DIR / f"{label_type}s.yaml"
    examples_path = args.examples or SCRIPT_DIR / f"{label_type}_examples.yaml"

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    LOGGER.info("Run args:")
    for key, value in vars(args).items():
        LOGGER.info("  %s: %s", key, value)

    labels, examples = load_label_resources(labels_path, examples_path, label_type)
    if len(labels) > len(ANSWER_IDS):
        raise ValueError(
            f"At most {len(ANSWER_IDS)} labels are supported; got {len(labels)}"
        )
    store = open_result_store(args.output, labels, args.prompt_order, args.model)
    if len(store.completed) >= args.max_documents:
        LOGGER.info("Already have %d results; nothing to do", len(store.completed))
        return 0
    if (
        args.input_jsonl is not None
        and args.input_jsonl.resolve() == args.output.resolve()
    ):
        raise ValueError("--input-jsonl and --output must be different files")
    documents: Iterable[tuple[int, InputDocument]] = (
        stream_jsonl(args.input_jsonl)
        if args.input_jsonl is not None
        else stream_huggingface_dataset(
            args.input_dataset,
            dataset_config=args.dataset_config,
            dataset_split=args.dataset_split,
        )
    )
    client = OpenAIChatClient(
        api_base=args.api_base,
        api_key=api_key,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        max_completion_tokens=args.max_completion_tokens,
        timeout=args.request_timeout,
        max_retries=args.max_retries,
    )
    process_documents_in_batches(
        documents=documents,
        store=store,
        process_batch=lambda batch: process_batch(
            batch,
            store=store,
            client=client,
            labels=labels,
            examples=examples,
            label_type=label_type,
            language=language,
            seed=args.seed,
            prompt_order=args.prompt_order,
            max_document_chars=args.max_document_chars,
            concurrency=args.concurrency,
        ),
        batch_size=args.batch_size,
        max_documents=args.max_documents,
    )
    return 0


if __name__ == "__main__":
    # Exceptions propagate normally; only a completed run skips interpreter
    # finalization. See labeling_common.exit_without_finalization.
    exit_without_finalization(main())
