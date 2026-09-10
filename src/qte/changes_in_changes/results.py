from dataclasses import dataclass
from typing import Any, NamedTuple

import polars as pl
from great_tables import GT
from numpy.typing import NDArray
from rich.console import Group
from scipy.stats import norm

from qte.changes_in_changes.custom_types import BasePeriod, ControlGroup
from qte.names import CI_LB_ID, CI_UB_ID, EFFECT_ID, SE_ID
from qte.presentation.altair_charts import make_plot
from qte.presentation.tables.gt_tables import format_qte_result_combined_gt
from qte.presentation.tables.rich_tables import _format_qte_result_for_console


class GroupTimeEffect(NamedTuple):
    group: int
    tp: int
    ecdf_observed: dict[str, NDArray]
    ecdf_counterfact: dict[str, NDArray]
    mean_observed: float
    mean_countfact: float
    group_size_observed: int
    group_size_counterfactual: int


def _add_ci(ds: pl.DataFrame, alpha: float) -> pl.DataFrame:
    alpha_half = (1 - alpha) / 2
    return ds.with_columns(
        **{
            CI_LB_ID: pl.col(EFFECT_ID) + norm.ppf(alpha_half) * pl.col(SE_ID),
            CI_UB_ID: pl.col(EFFECT_ID) + norm.ppf(1 - alpha_half) * pl.col(SE_ID),
        },
    )


@dataclass()
class CicResult:
    _qtt: pl.DataFrame
    _att: pl.DataFrame
    outcome: str
    group: str
    base_period: BasePeriod
    control_group: ControlGroup

    def _make_table_header_content(self, alpha: float) -> dict[str, Any]:
        header_content = {"Outcome": self.outcome}
        return header_content

    def plot(self, alpha: float = 0.95):
        return make_plot(
            _add_ci(self._qtt, alpha), _add_ci(self._att, alpha), group=self.group
        )

    def tabulate(self, alpha: float = 0.95) -> GT:
        return format_qte_result_combined_gt(
            _add_ci(self._qtt, alpha),
            _add_ci(self._att, alpha),
            self.group,
            self._make_table_header_content(alpha),
        )

    def summarize(self, alpha: float = 0.95) -> Group:
        return _format_qte_result_for_console(
            _add_ci(self._qtt, alpha),
            _add_ci(self._att, alpha),
            self._make_table_header_content(alpha),
            float_precision=2,
        )


@dataclass()
class CicResults:
    group: CicResult
    event_study: CicResult
    overall: CicResult
