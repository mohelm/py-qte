from collections.abc import Iterable

from qte.names import CI_LB_ID, CI_UB_ID, EFFECT_ID, QUANTILE_ID, SE_ID

NICE_NAMES: dict[str, str] = {
    "group": "Group",
    QUANTILE_ID: "Quantile",
    EFFECT_ID: "Effect Estimate",
    SE_ID: "Std. Error",
    CI_LB_ID: "Lower CI",
    CI_UB_ID: "Upper CI",
}

QTE_TABLE_SUB_HEADER = "Quantile Treatment Effects"
ATE_TABLE_SUB_HEADER = "Average Treatment Effects"


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
