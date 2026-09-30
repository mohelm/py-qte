from dataclasses import dataclass

import polars as pl

from qte.cross_sectional.custom_types import CausalTarget, Estimator
from qte.custom_types import FormularRhs
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
    outcome : str
        Outcome variable the effect was estimated for.
    group : tuple[str, ...] or None
        Columns the estimates are grouped by.
    estimator : `Estimator`
        Estimator that produced the result.
    causal_target : `CausalTarget`
        Estimand that was targeted.
    ps_x_formular : str, optional
        Propensity score formula, when applicable.
    or_x_formular : str, optional
        Outcome regression formula, when applicable.
    """

    estimator: Estimator
    causal_target: CausalTarget
    ps_x_formular: FormularRhs | None = None
    or_x_formular: FormularRhs | None = None
