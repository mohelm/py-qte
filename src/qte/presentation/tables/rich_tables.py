from collections.abc import Iterable
from typing import Any

import polars as pl
from rich import box
from rich.console import Group
from rich.table import Table
from rich.text import Text

from qte.names import CI_LB_ID, CI_UB_ID, EFFECT_ID, QUANTILE_ID, SE_ID

NICE_NAMES: dict[str, str] = {
    EFFECT_ID: "Effect Estimate",
    QUANTILE_ID: "Quantile",
    SE_ID: "Std. Error",
    CI_LB_ID: "Lower CI",
    CI_UB_ID: "Upper CI",
}


def _make_header(
    content: dict[str, str],
    separator: str,
    new_line_char: str,
    additional_padding: int = 2,
) -> Iterable[str]:
    max_length_keys = max(len(k) for k in content)
    for k, v in content.items():
        padding = max_length_keys - len(k) + additional_padding
        yield f"{k}{separator * padding}: {v}{new_line_char}"


def _make_estimates_table(
    qtes: pl.DataFrame, title: str | None, float_precision: int
) -> Table:
    table = Table(
        title=title,
        box=box.SIMPLE,
        header_style="bold",
        expand=False,
    )

    for col_name, dtype in qtes.schema.items():
        justify = "right" if dtype.is_numeric() else "left"
        table.add_column(NICE_NAMES.get(col_name, col_name), justify=justify)

    for row in qtes.iter_rows():
        formatted_row = []
        for val in row:
            if isinstance(val, float):
                formatted_row.append(f"{val:.{float_precision}f}")
            elif val is None:
                formatted_row.append("[dim]null[/dim]")
            else:
                formatted_row.append(str(val))
        table.add_row(*formatted_row)
    return table


def _format_qte_result_for_console(
    qtes: pl.DataFrame,
    atts: pl.DataFrame,
    hc: dict[str, Any],
    float_precision: int = 4,
) -> Group:
    """Dynamically creates a summary table using Rich."""
    header = Text()
    header.append("Quantile Treatment Effect\n", style="bold")
    for h in _make_header(hc, " ", "\n"):
        header.append(h)

    return Group(
        header,
        _make_estimates_table(
            atts, title="Average Treatment Effects", float_precision=float_precision
        ),
        _make_estimates_table(
            qtes, title="Quantile Treatment Effects", float_precision=float_precision
        ),
    )
