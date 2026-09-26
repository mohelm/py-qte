from typing import Any

import polars as pl
from rich import box
from rich.console import Group
from rich.jupyter import JupyterMixin
from rich.table import Table
from rich.text import Text

from qte.presentation.tables.common import (
    ATE_TABLE_SUB_HEADER,
    NICE_NAMES,
    QTE_TABLE_SUB_HEADER,
    _make_header,
)

_VERTICAL_SEPARATOR = "│"


class _JupyterGroup(Group, JupyterMixin):
    """A ``Group`` that also renders as HTML when returned from a notebook cell."""


def _make_estimates_table(
    data: pl.DataFrame,
    title: str | None,
    float_precision: int,
    group: tuple[str, ...] | None = None,
) -> Table:
    table = Table(
        title=title,
        box=box.SIMPLE,
        header_style="bold",
        expand=False,
    )

    # Draw a vertical rule right after the group columns to set them apart from
    # the estimated quantities (mirrors the Great Tables layout).
    separator_after = group[-1] if group else None

    for col_name, dtype in data.schema.items():
        justify = "right" if dtype.is_numeric() else "left"
        table.add_column(NICE_NAMES.get(col_name, col_name), justify=justify)
        if col_name == separator_after:
            table.add_column(_VERTICAL_SEPARATOR, justify="center", style="dim")

    separator_idx = data.columns.index(separator_after) if separator_after else None
    for row in data.iter_rows():
        formatted_row = []
        for i, val in enumerate(row):
            if isinstance(val, float):
                formatted_row.append(f"{val:.{float_precision}f}")
            elif val is None:
                formatted_row.append("[dim]null[/dim]")
            else:
                formatted_row.append(str(val))
            if separator_idx is not None and i == separator_idx:
                formatted_row.append(_VERTICAL_SEPARATOR)
        table.add_row(*formatted_row)
    return table


def make_rich_table(
    qtes: pl.DataFrame,
    atts: pl.DataFrame,
    hc: dict[str, Any],
    float_precision: int = 4,
    group: str | tuple[str, ...] | None = None,
) -> Group:
    """Dynamically creates a summary table using Rich."""
    if isinstance(group, str):
        group = (group,)

    header = Text()
    header.append("Quantile Treatment Effect\n", style="bold")
    for h in _make_header(hc, " ", "\n"):
        header.append(h)

    return _JupyterGroup(
        header,
        _make_estimates_table(
            qtes, title=QTE_TABLE_SUB_HEADER, float_precision=float_precision, group=group
        ),
        _make_estimates_table(
            atts, title=ATE_TABLE_SUB_HEADER, float_precision=float_precision, group=group
        ),
    )
