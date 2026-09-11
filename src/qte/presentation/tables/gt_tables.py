from typing import Any

import polars as pl
from great_tables import GT, md

from qte.names import CI_LB_ID, CI_UB_ID, EFFECT_ID, QUANTILE_ID, SE_ID
from qte.presentation.tables.common import _make_header

NICE_NAMES: dict[str, str] = {
    "group": "Group",
    QUANTILE_ID: "Quantile",
    EFFECT_ID: "Effect Estimate",
    SE_ID: "Std. Error",
    CI_LB_ID: "Lower CI",
    CI_UB_ID: "Upper CI",
}


def format_qte_result_combined_gt(
    qtes: pl.DataFrame,
    atts: pl.DataFrame,
    group: str | None,
    hc: dict[str, Any],
    float_precision: int = 2,
) -> GT:

    combined = pl.concat(
        (
            d.with_columns(pl.lit(k).alias("__kind"))
            for d, k in [(qtes, "qte"), (atts, "att")]
        ),
        how="diagonal",
    ).select(
        *(group or []), QUANTILE_ID, EFFECT_ID, SE_ID, CI_LB_ID, CI_UB_ID, "__kind"
    )

    float_cols = [c for c, dtype in combined.schema.items() if dtype.is_float()]

    header_meat = "".join(_make_header(hc, "&nbsp", "<br>"))
    subtitle = f"<div style='font-family: monospace;'>{header_meat}</div>"

    return (
        GT(combined, groupname_col="__kind", rowname_col=group)
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
