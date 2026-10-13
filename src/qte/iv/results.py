"""Result container for the local quantile treatment effect estimator."""

from dataclasses import dataclass
from typing import Any

from qte.iv.custom_types import Estimator, WeightAdjustment
from qte.results import _BasicQteResult


@dataclass
class UnconditionalLocalQteResult(_BasicQteResult):
    """Result of a local quantile treatment effect estimation.

    Use `plot`, `tabulate`, `summarize` or `get_as_dataframe` to present the
    estimates. Instances are returned by `estimate_unconditional_local_effects`; the
    constructor is not part of the public API.

    Attributes
    ----------
    qtt : polars.DataFrame
        Quantile-specific effects, one row per quantile.
    att : polars.DataFrame
        Average effects for compliers (the local average treatment effect).
    outcome : ColumnName
        Outcome variable the effect was estimated for.
    group : tuple[str, ...] or None
        Columns the estimates are grouped by.
    complier_share : float
        Estimated share of compliers, ``E[kappa]``.
    first_stage : float
        Difference in treatment take-up between instrument arms,
        ``E[D | Z = 1] - E[D | Z = 0]``.
    estimator : `Estimator`
        Weighting scheme used to estimate the effects.
    """

    complier_share: float
    first_stage: float
    estimator: Estimator

    def _make_table_header_content(self, alpha: float) -> dict[str, Any]:
        """Add the weighting scheme to the shared table and chart header."""
        return {
            "Outcome": self.outcome,
            "Estimator": self.estimator.value,
            "Confidence Lvl": alpha,
        }


@dataclass
class ConditionalLocalQteResult(_BasicQteResult):
    """Result of a conditional local quantile treatment effect estimation.

    Instances are returned by `estimate_conditional_local_effects`; the
    constructor is not part of the public API.

    Attributes
    ----------
    qtt : polars.DataFrame
        Quantile-specific effects, one row per quantile. ``q_t``/``q_c`` are the
        counterfactual quantiles at the covariate reference (every covariate at
        its baseline), i.e. the ``Constant`` row of Abadie, Angrist & Imbens
        (2002), Table III, plus/minus the treatment effect.
    att : polars.DataFrame
        Average effects for compliers (the local average treatment effect), with
        ``m_t``/``m_c`` the complier treated/control mean outcomes.
    outcome : ColumnName
        Outcome variable the effect was estimated for.
    group : tuple[str, ...] or None
        Columns the estimates are grouped by.
    complier_share : float
        Estimated share of compliers.
    first_stage : float
        Difference in treatment take-up between instrument arms,
        ``E[D | Z = 1] - E[D | Z = 0]``.
    weight_adjustment : `WeightAdjustment`
        How the nonnegative complier weights were formed.
    """

    complier_share: float
    first_stage: float
    weight_adjustment: WeightAdjustment

    def _make_table_header_content(self, alpha: float) -> dict[str, Any]:
        """Add the weight adjustment to the shared table and chart header."""
        return {
            "Outcome": self.outcome,
            "Weights": self.weight_adjustment.value,
            "Confidence Lvl": alpha,
        }
