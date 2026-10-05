"""Download and cache rows of WebOrganizer annotation datasets.

Rows come from the Hugging Face dataset viewer API, so only the standard
library is needed and no full dataset download. Each sample is cached as JSONL
next to a ``.spec.json`` file describing it (dataset, split, offsets, length);
a cached file is only reused when its spec matches the request exactly.

Only the fields needed to rebuild WebOrganizer's label distributions are kept
(no document text): the WARC record ID, ``choice_loss``, ``label_permutation``,
``fewshot_permutation``, ``top_choice_index``, and the first line of
``top_choice``.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

ROWS_URL = "https://datasets-server.huggingface.co/rows"
# The dataset viewer API returns at most 100 rows per request.
MAX_ROWS_PER_REQUEST = 100
NUMERIC_FIELDS = ("choice_loss", "label_permutation", "fewshot_permutation", "top_choice_index")


def _request_rows(
    dataset: str, split: str, offset: int, length: int, max_retries: int
) -> list[dict[str, Any]]:
    """Request one page of rows, retrying on rate limits and transient errors."""
    query = urllib.parse.urlencode(
        {"dataset": dataset, "config": "default", "split": split, "offset": offset, "length": length}
    )
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(f"{ROWS_URL}?{query}", timeout=60) as response:
                return json.load(response)["rows"]
        except urllib.error.HTTPError as exc:
            last_error = exc
            retry_after = int(exc.headers.get("Retry-After", 0) or 0)
            wait = min(retry_after or 20 * (attempt + 1), 120)
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            wait = 10 * (attempt + 1)
        LOGGER.warning("Request for %s offset %d failed (%s); retrying in %ds", dataset, offset, last_error, wait)
        time.sleep(wait)
    raise RuntimeError(f"Could not fetch {dataset} offset {offset}: {last_error}")


def _compact_row(row: dict[str, Any]) -> dict[str, Any]:
    """Keep only the fields needed for label distributions."""
    if set(row.get("truncated_cells") or []) & {*NUMERIC_FIELDS, "metadata", "top_choice"}:
        raise ValueError(f"Row {row.get('row_idx')} has truncated label fields")
    data = row["row"]
    try:
        compact = {field: data[field] for field in NUMERIC_FIELDS}
        compact["row_idx"] = row["row_idx"]
        compact["metadata"] = {"WARC-Record-ID": data["metadata"]["WARC-Record-ID"]}
        compact["top_choice"] = data["top_choice"].split("\n")[0]
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"Row {row.get('row_idx')} is missing a field: {exc}") from exc
    return compact


def fetch_rows(
    dataset: str,
    offsets: Sequence[int],
    cache_path: Path,
    *,
    length: int = MAX_ROWS_PER_REQUEST,
    split: str = "train",
    pause_seconds: float = 2.0,
    max_retries: int = 8,
) -> list[dict[str, Any]]:
    """Return rows of a WebOrganizer dataset, downloading them if not cached.

    Args:
        dataset: Dataset name, e.g. ``WebOrganizer/TopicAnnotations-Llama-3.1-8B``.
        offsets: Row offsets to request; each request returns up to ``length``
            rows. Offsets past the end of the split return no rows.
        cache_path: JSONL cache file; a ``.spec.json`` file is written next to it.
        length: Rows per request, at most 100.
        split: Dataset split.
        pause_seconds: Pause between requests, to stay under rate limits.
        max_retries: Attempts per request before failing.

    Returns:
        The compact rows in request order.

    Raises:
        ValueError: If the cache was made for a different request, or rows are
            truncated or malformed.
        RuntimeError: If a request keeps failing.
    """
    if not 1 <= length <= MAX_ROWS_PER_REQUEST:
        raise ValueError(f"length must be between 1 and {MAX_ROWS_PER_REQUEST}")
    spec = {"dataset": dataset, "split": split, "offsets": list(offsets), "length": length}
    spec_path = cache_path.with_suffix(".spec.json")
    if cache_path.exists():
        if not spec_path.exists() or json.loads(spec_path.read_text(encoding="utf-8")) != spec:
            raise ValueError(
                f"{cache_path} was cached for a different request; delete it and its "
                ".spec.json file to download again"
            )
        with cache_path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle]
        LOGGER.info("Loaded %d cached rows from %s", len(rows), cache_path)
        return rows

    rows = []
    for number, offset in enumerate(offsets, start=1):
        rows.extend(_compact_row(row) for row in _request_rows(dataset, split, offset, length, max_retries))
        if number < len(offsets):
            time.sleep(pause_seconds)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    temporary.replace(cache_path)
    spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Downloaded %d rows of %s to %s", len(rows), dataset, cache_path)
    return rows
