import dataclasses
from dataclasses import dataclass, field
from io import StringIO
from typing import ClassVar

import numpy as np
import polars as pl
from altair import Chart
from great_tables import GT
from numpy.typing import NDArray
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from scipy.stats import norm

from qte.custom_types import CausalTarget, Estimator
from qte.names import EFFECT_ID, QUANTILE_ID, SE_ID


def _format_qte_result_for_console(
    res: pl.DataFrame,
    title: str = "Quantile Effects Summary",
    float_precision: int = 4,
) -> Table:
    """Dynamically creates a Rich Table from any Polars DataFrame."""
    table = Table(
        title=title,
        title_style="bold white",
        box=box.SIMPLE_HEAD,
        header_style="bold magenta",
        expand=True,
    )

    # 1. Dynamically add columns based on schema and data type
    for col_name, dtype in res.schema.items():
        # Right align numeric columns, left align strings/others
        justify = "right" if dtype.is_numeric() else "left"
        # Style float columns distinctly
        style = "cyan" if dtype in (pl.Float32, pl.Float64) else "white"

        table.add_column(col_name, justify=justify, style=style)

    # 2. Add rows with formatted values
    for row in res.iter_rows():
        formatted_row = []
        for val in row:
            if isinstance(val, float):
                # Format floats cleanly according to specified precision
                formatted_row.append(f"{val:.{float_precision}f}")
            elif val is None:
                formatted_row.append("[dim]null[/dim]")
            else:
                formatted_row.append(str(val))

        table.add_row(*formatted_row)

    return table


@dataclass()
class _QteIntermediateResult:
    qs: NDArray[np.float64]
    q_val_t: NDArray[np.float64]
    q_val_c: NDArray[np.float64]
    effects: None | NDArray[np.float64] = None

    def __post_init__(self) -> None:
        self.effects = self.q_val_t - self.q_val_c

    def to_polars(self) -> pl.DataFrame:
        return pl.from_dict(dataclasses.asdict(self)).rename({
            "effects": EFFECT_ID,
            "qs": QUANTILE_ID,
        })


@dataclass
class QteResult:
    estimator: Estimator
    causal_target: CausalTarget
    _res: pl.DataFrame = field(repr=False)

    _exclude_from_tables: ClassVar[tuple[str, ...]] = ("q_val_c", "q_val_t")

    def __post_init__(self) -> None:
        self._res = self._res.sort(QUANTILE_ID)

    def _add_ci(self, alpha: float) -> pl.DataFrame:
        alpha_half = (1 - alpha) / 2
        return self._res.with_columns(
            CI_LB_ID=pl.col(EFFECT_ID) + norm.ppf(1 - alpha_half) * pl.col(SE_ID),
            CI_UB_ID=pl.col(EFFECT_ID) + norm.ppf(alpha_half),
        )

    def plot(self, alpha: float = 0.95) -> GT:
        pass

    def tabulate(self, alpha: float = 0.95) -> Chart:
        pass

    def _make_table_for_console(self, alpha: float) -> Table:
        return _format_qte_result_for_console(
            self._add_ci(alpha).drop(self._exclude_from_tables)
        )

    def __str__(self) -> str:
        buf = StringIO()
        console = Console(file=buf, force_terminal=True)
        t = self._make_table_for_console(0.95)
        console.print(Panel(t, border_style="blue"))
        return buf.getvalue()

    def summarize(self, alpha: float = 0.95) -> Table:
        return self._make_table_for_console(alpha)

    def get_as_dataframe(self, alpha: float = 0.95) -> pl.DataFrame:
        return self._add_ci(alpha=alpha)
