from dataclasses import dataclass

import polars as pl

from qte.cross_sectional.custom_types import CausalTarget, Estimator
from qte.custom_types import FormulaRhs
from qte.results import _BasicQteResult


@dataclass(frozen=True)
class _QteIntermediateResult:
    qtt: pl.DataFrame
    att: pl.DataFrame


@dataclass
class QteResult(_BasicQteResult):
    """Result of a cross-sectional quantile treatment effect estimation.

    Use `plot`, `tabulate`, `summarize` or `get_as_dataframe` to present the
    estimates. Instances are returned by the ``estimate_*`` functions; the
    constructor is not part of the public API.

    Attributes
    ----------
    qtt : polars.DataFrame
        Quantile-specific effects, one row per quantile.
    att : polars.DataFrame
        Average treatment effects.
    outcome : ColumnName
        Outcome variable the effect was estimated for.
    group : tuple[str, ...] or None
        Columns the estimates are grouped by.
    estimator : `Estimator`
        Estimator that produced the result.
    causal_target : `CausalTarget`
        Estimand that was targeted.
    propensity_score_formula : FormulaRhs, optional
        Propensity score formula, when applicable.
    outcome_regression_formula : FormulaRhs, optional
        Outcome regression formula, when applicable.
    """

    estimator: Estimator
    causal_target: CausalTarget
    propensity_score_formula: FormulaRhs | None = None
    outcome_regression_formula: FormulaRhs | None = None
