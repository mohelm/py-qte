from collections.abc import Iterable
from typing import Any

from qte.names import (
    CI_LB_ID,
    CI_UB_ID,
    EFFECT_ID,
    ESTIMATE_ID,
    N_BOOT_ID,
    P_VALUE_ID,
    QUANTILE_ID,
    SE_ID,
    STATISTIC_ID,
    TERM_ID,
    VERDICT_ID,
    Z_ID,
)

NICE_NAMES: dict[str, str] = {
    "group": "Group",
    QUANTILE_ID: "Quantile",
    EFFECT_ID: "Effect Estimate",
    SE_ID: "Std. Error",
    CI_LB_ID: "Lower CI",
    CI_UB_ID: "Upper CI",
    TERM_ID: "Term",
    ESTIMATE_ID: "Coefficient",
    Z_ID: "z value",
    P_VALUE_ID: "p value",
    STATISTIC_ID: "Test Statistic",
    N_BOOT_ID: "Bootstrap Draws",
    VERDICT_ID: "Verdict",
}

QTE_TABLE_SUB_HEADER = "Quantile Treatment Effects"
ATE_TABLE_SUB_HEADER = "Average Treatment Effects"
COEFFICIENT_TABLE_TITLE = "Instrument Probability Model"
COEFFICIENT_TABLE_SUB_HEADER = "Coefficients"
VALIDITY_TABLE_TITLE = "Instrument Validity Test"


def _make_table_subtitle(header_content: dict[str, Any]) -> str:
    """Format header content as a monospace HTML subtitle."""
    st_meat = "".join(_make_header(header_content, "\u00a0", "<br>"))
    return f"<div style='font-family: monospace; text-align: left;'>{st_meat}</div>"


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
