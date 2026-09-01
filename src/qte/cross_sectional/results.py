import dataclasses
from collections.abc import Iterable
from dataclasses import dataclass, field
from io import StringIO
from typing import Any, ClassVar

import altair as alt
import numpy as np
import polars as pl
from great_tables import GT, md
from numpy.typing import NDArray
from rich import box
from rich.console import Console, Group
from rich.table import Table
from rich.text import Text
from scipy.stats import norm

from qte.custom_types import CausalTarget, Estimator, FormularRhs
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


def _format_qte_result_for_console(
    res: pl.DataFrame,
    hc: dict[str, Any],
    float_precision: int = 4,
) -> Group:
    """Dynamically creates a summary table using Rich."""
    header = Text()
    header.append("Quantile Treatment Effect\n", style="bold")
    for h in _make_header(hc, " ", "\n"):
        header.append(h)

    table = Table(
        box=box.SIMPLE,
        header_style="bold",
        expand=False,
    )

    for col_name, dtype in res.schema.items():
        justify = "right" if dtype.is_numeric() else "left"
        table.add_column(NICE_NAMES.get(col_name, col_name), justify=justify)

    for row in res.iter_rows():
        formatted_row = []
        for val in row:
            if isinstance(val, float):
                formatted_row.append(f"{val:.{float_precision}f}")
            elif val is None:
                formatted_row.append("[dim]null[/dim]")
            else:
                formatted_row.append(str(val))
        table.add_row(*formatted_row)

    return Group(header, table)


@dataclass()
class _QteIntermediateResult:
    qs: NDArray[np.float64]
    q_val_t: NDArray[np.float64]
    q_val_c: NDArray[np.float64]
    effects: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        self.effects = self.q_val_t - self.q_val_c

    def to_polars(self) -> pl.DataFrame:
        return pl.from_dict(dataclasses.asdict(self)).rename(
            {
                "effects": EFFECT_ID,
                "qs": QUANTILE_ID,
            }
        )


@dataclass
class QteResult:
    estimator: Estimator
    causal_target: CausalTarget
    outcome_variable: str
    _res: pl.DataFrame = field(repr=False)
    ps_x_formular: FormularRhs | None = None
    or_x_formular: FormularRhs | None = None

    _exclude_from_tables: ClassVar[tuple[str, ...]] = ("q_val_c", "q_val_t")

    def _make_table_header_content(self, alpha: float) -> dict[str, Any]:
        header_content = {
            "Estimator": self.estimator.upper(),
            "Causal Target": self.causal_target.upper(),
            "Outcome": self.outcome_variable,
        }
        if self.ps_x_formular is not None:
            header_content["Propensity Score Model"] = self.ps_x_formular
        if self.or_x_formular is not None:
            header_content["Outcome Model"] = self.or_x_formular
        header_content["Confidence Lvl"] = alpha
        return header_content

    def __post_init__(self) -> None:
        self._res = self._res.sort(QUANTILE_ID)

    def _add_ci(self, alpha: float) -> pl.DataFrame:
        alpha_half = (1 - alpha) / 2
        return self._res.with_columns(
            **{
                CI_LB_ID: pl.col(EFFECT_ID) + norm.ppf(alpha_half) * pl.col(SE_ID),
                CI_UB_ID: pl.col(EFFECT_ID) + norm.ppf(1 - alpha_half) * pl.col(SE_ID),
            },
        )

    def _make_table_for_console(self, alpha: float) -> Group:
        return _format_qte_result_for_console(
            self._add_ci(alpha).drop(self._exclude_from_tables),
            self._make_table_header_content(alpha),
        )

    def __str__(self) -> str:
        buf = StringIO()
        console = Console(file=buf, force_terminal=True)
        t = self._make_table_for_console(0.95)
        console.print(t)
        return buf.getvalue()

    def summarize(self, alpha: float = 0.95) -> Group:
        return self._make_table_for_console(alpha)

    def get_as_dataframe(self, alpha: float = 0.95) -> pl.DataFrame:
        return self._add_ci(alpha=alpha)

    def plot(self, alpha: float = 0.9) -> alt.LayerChart | alt.FacetChart:
        _ds = self._add_ci(alpha)
        zero = (
            alt.Chart()
            .encode(y=alt.datum(0))
            .mark_rule(color="grey", strokeDash=(4, 4))
        )
        tooltip = [
            alt.Tooltip("q", title="Quantile"),
            alt.Tooltip(
                CI_LB_ID,
                title="CI Lower Bound:",
                format=".2f",
            ),
            alt.Tooltip(
                "effect",
                title="Effect Size:",
                format=".2f",
            ),
            alt.Tooltip(
                CI_UB_ID,
                title="CI Upper Bound:",
                format=".2f",
            ),
            alt.Tooltip("q_val_c", title="Baseline:", format=".2f"),
            alt.Tooltip("q_val_t", title="Baseline + Effect:", format=".2f"),
        ]

        def _make_line_layer(
            y: str, stroke_dash: tuple[int, int] = (1, 0)
        ) -> alt.LayerChart | alt.FacetChart:
            x = alt.X("q:Q", title="Quantile")
            base = alt.Chart()
            line_chart = base.mark_line(
                color="black", strokeDash=list(stroke_dash)
            ).encode(x=x, y=alt.Y(y, title=""))  # type: ignore
            point_chart = base.mark_point(color="black", fill="black").encode(  # type: ignore
                x=x, y=alt.Y(y, title=""), tooltip=tooltip
            )
            return alt.layer(line_chart, point_chart, zero)  # type: ignore

        return alt.layer(
            _make_line_layer("effect"),
            _make_line_layer(CI_LB_ID, stroke_dash=(8, 8)),
            _make_line_layer(CI_UB_ID, stroke_dash=(8, 8)),
            data=_ds,
        )

    def tabulate(self, alpha: float = 0.95) -> GT:
        _ds = self._add_ci(alpha).drop(self._exclude_from_tables)
        header_meat = "".join(
            h
            for h in _make_header(
                self._make_table_header_content(alpha), "&nbsp", "<br>"
            )
        )

        subtitle_md = f"<div style='font-family: monospace;'>{header_meat}</div>"

        return (
            GT(_ds)
            .tab_header(
                title="Quantile Treatment Effect",
                subtitle=md(subtitle_md),
            )
            .tab_options(heading_align="left")
            .cols_label(
                **{id_: nn for id_, nn in NICE_NAMES.items() if id_ in _ds.columns}
            )
            .fmt_number(decimals=4)
        )
