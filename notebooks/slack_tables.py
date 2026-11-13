"""Copy notebook tables to the clipboard as aligned plain text, for pasting into Slack.

Slack messages don't render Markdown tables, so a table is pasted as an aligned monospace
block between triple backticks. Styled tables keep their formatting (percentages, thousands
separators) because the text comes from the Styler.

Usage in a notebook (the module sits next to the notebooks, which run from ``notebooks/``):

    import slack_tables
    slack_tables.track()          # once, in the setup cell: remember every table displayed

    ...                           # run the cells; their tables are remembered

    slack_tables.list_tables()    # numbered list of the remembered tables
    slack_tables.copy()           # copy the last table shown
    slack_tables.copy(3)          # copy table 3 of the list
    slack_tables.copy(some_df)    # or pass a DataFrame or Styler directly

The text is copied with ``xclip`` (Linux) when it is available, and printed either way.
Wide tables wrap in Slack; select columns first (``df[["A", "B"]]``) if needed.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from typing import Any

import pandas as pd
from pandas.io.formats.style import Styler

# Tables displayed since track() was called: (the cell's In [n] number, table).
SHOWN: list[tuple[int | None, pd.DataFrame | Styler]] = []
MAX_REMEMBERED = 200
NUMERIC = re.compile(r"^[+\-−]?[\d.,]+(?:e[+-]?\d+)?\s?%?$|^[—–-]$")


def track() -> None:
    """Remember every DataFrame and Styler the notebook displays from now on.

    Registers HTML formatters that record each table and then render it as usual, so
    ``display(...)`` and a cell's last expression are both caught.

    Raises:
        RuntimeError: If not running in IPython.
    """
    from IPython import get_ipython

    shell = get_ipython()
    if shell is None:
        raise RuntimeError("track() only works in a notebook or IPython")
    html = shell.display_formatter.formatters["text/html"]

    def remember(table: pd.DataFrame | Styler) -> str:
        # While a cell runs, execution_count is already the next cell's number.
        SHOWN.append((shell.execution_count - 1, table))
        del SHOWN[:-MAX_REMEMBERED]
        return table._repr_html_()

    html.for_type(pd.DataFrame, remember)
    html.for_type(Styler, remember)


def _title(table: pd.DataFrame | Styler) -> str:
    """Describe a table by its shape and first columns."""
    data = table.data if isinstance(table, Styler) else table
    columns = [" / ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in data.columns[:4]]
    more = ", ..." if len(data.columns) > 4 else ""
    return f"{data.shape[0]} rows × {data.shape[1]} columns: {', '.join(columns)}{more}"


def list_tables() -> None:
    """Print the remembered tables with their numbers for ``copy``."""
    if not SHOWN:
        print("No tables remembered; call slack_tables.track() first and rerun the cells")
    for number, (count, table) in enumerate(SHOWN):
        print(f"{number:3d}  [In {count}]  {_title(table)}")


def _cells(table: pd.DataFrame | Styler) -> tuple[list[list[str]], int]:
    """Return the table's formatted text cells, header rows first, with one header row per column level."""
    # Unstyled DataFrames get 4 decimals instead of the Styler default of 6.
    styler = table if isinstance(table, Styler) else table.style.format(precision=4)
    data = styler.data
    text = styler.to_string(delimiter="\t", sparse_index=False, sparse_columns=False)
    rows = [line.split("\t") for line in text.rstrip("\n").split("\n")]
    header_rows = data.columns.nlevels
    # A row with the index names follows the column headers when the index is named; merge
    # it into the last header row so each column has one label per level.
    if any(name is not None for name in data.index.names) and len(rows) > header_rows:
        names, last = rows[header_rows], rows[header_rows - 1]
        if all(cell == "" for cell in names[data.index.nlevels :]):
            rows[header_rows - 1] = [n or l for n, l in zip(names, last)] + last[len(names) :]
            del rows[header_rows]
    width = max(len(row) for row in rows)
    return [row + [""] * (width - len(row)) for row in rows], header_rows


def to_text(table: pd.DataFrame | Styler) -> str:
    """Render a table as aligned plain text: numbers right-aligned, text left-aligned.

    Args:
        table: A DataFrame or Styler.

    Returns:
        The table between triple backticks, ready to paste into Slack.
    """
    rows, header_rows = _cells(table)
    columns = list(zip(*rows))
    widths = [max(len(cell) for cell in column) for column in columns]
    numeric = [
        all(NUMERIC.match(cell) or cell == "" for cell in column[header_rows:]) and any(column[header_rows:])
        for column in columns
    ]
    lines = []
    for index, row in enumerate(rows):
        cells = [
            cell.rjust(width) if is_number else cell.ljust(width)
            for cell, width, is_number in zip(row, widths, numeric)
        ]
        lines.append("  ".join(cells).rstrip())
        if index == header_rows - 1:
            lines.append("  ".join("-" * width for width in widths))
    return "```\n" + "\n".join(lines) + "\n```"


def copy(table: int | pd.DataFrame | Styler = -1) -> str:
    """Copy a table to the clipboard as Slack-ready text, and print it.

    Args:
        table: A number from ``list_tables`` (default -1: the last table shown), or a
            DataFrame or Styler.

    Returns:
        The text.

    Raises:
        IndexError: If no table with that number is remembered.
    """
    if isinstance(table, int):
        if not SHOWN:
            raise IndexError("No tables remembered; call slack_tables.track() first and rerun the cells")
        table = SHOWN[table][1]
    text = to_text(table)
    copied = False
    if shutil.which("xclip"):
        result = subprocess.run(
            ["xclip", "-selection", "clipboard"], input=text.encode(), stderr=subprocess.DEVNULL, check=False
        )
        copied = result.returncode == 0
    print(text)
    print("(copied to the clipboard)" if copied else "(not copied: no working xclip; copy the text above)")
    return text


def _demo() -> Any:
    """Small example table, for trying the module out."""
    index = pd.MultiIndex.from_tuples([("topic", "URL + text"), ("topic", "Text only")], names=["Label type", "Input"])
    frame = pd.DataFrame({"Documents": [362706, 362706], "Accuracy": [0.887, 0.861]}, index=index)
    return frame.style.format({"Documents": "{:,}", "Accuracy": "{:.1%}"})
