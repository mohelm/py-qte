from dataclasses import dataclass, field
from typing import Any, ClassVar

import altair as alt
import polars as pl
from great_tables import GT
from rich.console import Group

from qte.names import (
    MEAN_CONTROL_ID,
    MEAN_TREATED_ID,
    QUANTILE_CONTROL_VAL_ID,
    QUANTILE_ID,
    QUANTILE_TREATED_VAL_ID,
)
from qte.presentation.altair_charts import make_plot
from qte.presentation.tables.gt_tables import make_great_table
from qte.presentation.tables.rich_tables import make_rich_table
from qte.stats import get_ci

type AltairChart = alt.Chart | alt.LayerChart | alt.FacetChart


@dataclass()
class _BasicQteResult:
    qtt: pl.DataFrame = field(repr=False)
    att: pl.DataFrame = field(repr=False)
    outcome: str
    group: tuple[str] | tuple[str, str] | None

    _exclude_from_qtt_tables: ClassVar[tuple[str, ...]] = (
        QUANTILE_TREATED_VAL_ID,
        QUANTILE_CONTROL_VAL_ID,
    )
    _exclude_from_att_tables: ClassVar[tuple[str, ...]] = (
        MEAN_CONTROL_ID,
        MEAN_TREATED_ID,
    )

    def __post_init__(self) -> None:
        self.qtt = self.qtt.sort(
            [
                *(
                    [self.group]
                    if isinstance(self.group, str)
                    else self.group
                    if self.group is not None
                    else []
                ),
                QUANTILE_ID,
            ]
        )
        self.att = self.att.sort(
            [self.group]
            if isinstance(self.group, str)
            else self.group
            if self.group is not None
            else []
        )

    def _make_table_header_content(self, alpha: float) -> dict[str, Any]:
        header_content = {"Outcome": self.outcome}
        header_content["Confidence Lvl"] = alpha
        return header_content

    def plot(self, alpha: float = 0.95) -> AltairChart:
        """Plot the quantile and average effects with confidence intervals.

        Parameters
        ----------
        alpha : float, default=0.95
            Confidence level of the intervals.

        Returns
        -------
        altair.Chart
            Altair chart of the estimates.
        """
        return make_plot(
            self.qtt.with_columns(get_ci(alpha)),
            self.att.with_columns(get_ci(alpha)),
            group=self.group,
        )

    def tabulate(self, alpha: float = 0.95) -> GT:
        """Build a Great Tables table of the estimates.

        Parameters
        ----------
        alpha : float, default=0.95
            Confidence level of the intervals.

        Returns
        -------
        great_tables.GT
            Formatted table of the estimates.
        """
        return make_great_table(
            self.qtt.drop(self._exclude_from_qtt_tables).with_columns(get_ci(alpha)),
            self.att.drop(self._exclude_from_att_tables).with_columns(get_ci(alpha)),
            self.group,
            self._make_table_header_content(alpha),
        )

    def get_as_dataframe(self, alpha: float = 0.95) -> pl.DataFrame:
        """Return the quantile effects with confidence interval columns.

        Parameters
        ----------
        alpha : float, default=0.95
            Confidence level of the intervals.

        Returns
        -------
        polars.DataFrame
            Quantile effects with ``ci_lb`` and ``ci_ub`` columns.
        """
        return self.qtt.with_columns(get_ci(alpha))

    def summarize(self, alpha: float = 0.95) -> Group:
        """Render a rich text summary of the estimates.

        Parameters
        ----------
        alpha : float, default=0.95
            Confidence level of the intervals.

        Returns
        -------
        rich.console.Group
            Rich renderable for terminals and notebooks.
        """
        return make_rich_table(
            self.qtt.drop(self._exclude_from_qtt_tables).with_columns(get_ci(alpha)),
            self.att.drop(self._exclude_from_att_tables).with_columns(get_ci(alpha)),
            self._make_table_header_content(alpha),
            float_precision=2,
            group=self.group,
        )
