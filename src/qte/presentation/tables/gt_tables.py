from typing import Any

import polars as pl
from great_tables import GT, loc, md, style

from qte.names import CI_LB_ID, CI_UB_ID, EFFECT_ID, QUANTILE_ID, SE_ID
from qte.presentation.tables.common import (
    ATE_TABLE_SUB_HEADER,
    NICE_NAMES,
    QTE_TABLE_SUB_HEADER,
    _make_header,
)


def make_great_table(
    qtes: pl.DataFrame,
    atts: pl.DataFrame,
    group: str | None | tuple[str, ...],
    header_content: dict[str, Any],
    float_precision: int = 2,
) -> GT:
    if isinstance(group, str):
        group = (group,)

    combined = pl.concat(
        (
            data.with_columns(pl.lit(est).alias("__kind"))
            for data, est in [(qtes, QTE_TABLE_SUB_HEADER), (atts, ATE_TABLE_SUB_HEADER)]
        ),
        how="diagonal",
    ).select(*(group or []), QUANTILE_ID, EFFECT_ID, SE_ID, CI_LB_ID, CI_UB_ID, "__kind")

    float_cols = [c for c, dtype in combined.schema.items() if dtype.is_float()]

    st_meat = "".join(_make_header(header_content, "&nbsp", "<br>"))
    subtitle = f"<div style='font-family: monospace;'>{st_meat}</div>"

    base = (
        GT(combined, groupname_col="__kind")
        .tab_header(
            title="Quantile & Average Treatment Effects",
            subtitle=md(subtitle),
        )
        .cols_label(**{col: NICE_NAMES.get(col, col) for col in combined.columns})
        .fmt_number(columns=float_cols, decimals=float_precision)
        .sub_missing(missing_text="-")
        .tab_options(
            heading_title_font_size="16px",
            column_labels_font_weight="bold",
            row_group_font_weight="bold",
            row_group_background_color="#f8f9fa",
        )
    )
    if group is not None:
        base = base.tab_style(
            style=style.borders(sides="right", style="solid", weight="2px", color="black"),
            locations=loc.body(columns=group[-1]),
        )
    return base
