"""Abadie kappa-weighting estimator of the local quantile treatment effect.

The estimator targets the quantile treatment effect for *compliers* in the
Imbens-Angrist (1994) sense, using the weighting representation of Abadie
(2003). Under the LATE assumptions (independence, exclusion, monotonicity and a
relevant instrument) the complier potential-outcome distributions are identified
by

    F_{Y(d) | complier}(y) = E[w_d * 1{Y <= y}] / E[w_d],

with the Abadie / Froelich-Melly weights

    w_1 = D Z / p(X) - D (1 - Z) / (1 - p(X)),
    w_0 = (1 - D)(1 - Z) / (1 - p(X)) - (1 - D) Z / p(X),

where ``p(X) = P(Z = 1 | X)``. Dividing the treated weight by ``p(X)`` (rather
than using ``kappa * D``) is what makes the complier means unconditional on
``X``; the two coincide only when ``p(X)`` is constant. The local QTE is the
difference between the two complier quantiles. This module implements the point
estimate, its bootstrap standard errors, and the simulation helpers used to
validate it.

The weight algebra is assembled from Polars expressions; NumPy is only used for
the first-step propensity model and for inverting the signed weighted CDFs.
"""

from dataclasses import dataclass
from functools import partial

import numpy as np
import polars as pl
from numpy.typing import ArrayLike, NDArray

from qte.bootstrap import (
    BootstrapConfig,
    _attach_standard_errors,
    _make_bootstrap_config,
    _make_bootstrap_runs,
)
from qte.constants import MEDIAN
from qte.custom_types import ColumnName, FormulaRhs
from qte.helpers import _as_iterable_array, _get_mean_differences, _get_quantile_differences
from qte.iv.results import LocalQteResult
from qte.stats import estimate_propensity_score

_EPS = 1e-8
_DEFAULT_TRIM = 0.001
_PROPENSITY_ID = "_propensity"
_COMPLIER_WEIGHT_TREATED_ID = "_complier_weight_treated"
_COMPLIER_WEIGHT_CONTROL_ID = "_complier_weight_control"


@dataclass(frozen=True)
class _LocalQteIntermediateResult:
    """Point estimates from one (bootstrap) draw."""

    qtt: pl.DataFrame
    att: pl.DataFrame
    complier_share: float
    first_stage: float


def _compute_complier_weight_treated(
    treatment: ColumnName, instrument: ColumnName, propensity: ColumnName
) -> pl.Expr:
    """Polars expression for the treated complier weight ``D(Z-p)/(p(1-p))``."""
    return (
        pl.col(treatment)
        * (pl.col(instrument) - pl.col(propensity))
        / (pl.col(propensity) * (1.0 - pl.col(propensity)))
    )


def _compute_complier_weight_control(
    treatment: ColumnName, instrument: ColumnName, propensity: ColumnName
) -> pl.Expr:
    """Polars expression for the control complier weight ``(1-D)(p-Z)/(p(1-p))``."""
    return (
        (1.0 - pl.col(treatment))
        * (pl.col(propensity) - pl.col(instrument))
        / (pl.col(propensity) * (1.0 - pl.col(propensity)))
    )


def _weighted_mean(value: ColumnName, weight: ColumnName) -> pl.Expr:
    """Polars expression for the ``weight``-weighted mean of ``value``."""
    return (pl.col(weight) * pl.col(value)).sum() / pl.col(weight).sum()


def _estimate_instrument_propensity(
    ds: pl.DataFrame,
    instrument: ColumnName,
    instrument_probability_formula: FormulaRhs | None,
    weights: ColumnName | None,
) -> NDArray[np.float64]:
    """Estimate ``p(X) = P(Z = 1 | X)``, constant when no formula is given."""
    z = ds[instrument].to_numpy().astype(float)
    n = len(ds)
    base = ds[weights].to_numpy().astype(float) if weights is not None else np.ones(n)
    if instrument_probability_formula is None:
        p = np.full(n, np.average(z, weights=base))
    else:
        fit = estimate_propensity_score(ds, instrument, instrument_probability_formula, weights)
        p = np.asarray(fit.predict(), dtype=float)
    # Avoid division blow-ups at the boundary; observations this extreme only
    # arise under (near) deterministic take-up.
    return np.clip(p, _EPS, 1.0 - _EPS)


def _compute_quantiles(
    y: NDArray[np.float64], w: NDArray[np.float64], qs: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Quantiles of the signed-weighted empirical CDF of ``y``.

    The complier weights are signed (always-takers and never-takers in the
    "wrong" instrument arm enter with negative weight), so the weighted CDF is
    not guaranteed to be monotone. We invert it by the first crossing in
    outcome order, which reduces to the usual quantile when the CDF is monotone.
    """
    order = np.argsort(y, kind="mergesort")
    values, inverse = np.unique(y[order], return_inverse=True)
    weight_by_value = np.bincount(inverse, weights=w[order])
    total = weight_by_value.sum()
    if total <= _EPS:
        raise ValueError(
            "Non-positive total complier weight; the instrument is too weak to "
            "identify the local quantile treatment effect."
        )
    cdf = np.cumsum(weight_by_value) / total
    out = np.empty(len(qs), dtype=float)
    for i, tau in enumerate(qs):
        crossing = np.flatnonzero(cdf >= tau)
        out[i] = values[crossing[0]] if crossing.size else values[-1]
    return out


def _conditional_takeup_diff(
    ds: pl.DataFrame,
    treatment: ColumnName,
    instrument: ColumnName,
    weight: ColumnName,
) -> float:
    """``E[D | Z = 1] - E[D | Z = 0]`` (the first stage), computed in Polars."""
    d = pl.col(treatment)
    z = pl.col(instrument)
    w = pl.col(weight)
    at_one = (d * w).filter(z == 1).sum() / w.filter(z == 1).sum()
    at_zero = (d * w).filter(z == 0).sum() / w.filter(z == 0).sum()
    return float(ds.select((at_one - at_zero).alias("first_stage")).item())


def _compute_local_effects(
    ds: pl.DataFrame,
    outcome: ColumnName,
    treatment: ColumnName,
    instrument: ColumnName,
    qs: NDArray[np.float64],
    *,
    instrument_probability_formula: FormulaRhs | None = None,
    weights: ColumnName | None = None,
    trim: float = _DEFAULT_TRIM,
) -> _LocalQteIntermediateResult:
    """Compute complier quantiles and means without bootstrap inference.

    Parameters
    ----------
    ds : polars.DataFrame
        Input data.
    outcome : ColumnName
        Name of the outcome column.
    treatment : ColumnName
        Name of the binary treatment column (``1`` treated, ``0`` not).
    instrument : ColumnName
        Name of the binary instrument column (``1`` encouraged, ``0`` not).
    qs : ndarray
        Quantiles in ``(0, 1)`` at which to estimate the effect.
    instrument_probability_formula : FormulaRhs, optional
        Right-hand side of the logit model for ``P(Z = 1 | X)``. When omitted
        the instrument probability is taken to be constant.
    weights : ColumnName, optional
        Name of a column with sampling weights. When omitted, a column of ones
        named ``"_w"`` is added, matching the other estimators.
    trim : float, default=0.001
        Drop observations whose estimated instrument propensity lies outside
        ``[trim, 1 - trim]``. ``0`` keeps every observation.

    Returns
    -------
    `_LocalQteIntermediateResult`
        Complier quantiles, complier means and the first-stage diagnostics.
    """
    qs = _as_iterable_array(qs)
    if weights is None:
        weights = "_w"
        ds = ds.with_columns(pl.lit(1.0).alias(weights))
    propensity = _estimate_instrument_propensity(
        ds, instrument, instrument_probability_formula, weights
    )

    estimation_data = ds.with_columns(pl.Series(_PROPENSITY_ID, propensity))
    if trim > 0.0:
        estimation_data = estimation_data.filter(
            (pl.col(_PROPENSITY_ID) >= trim) & (pl.col(_PROPENSITY_ID) <= 1.0 - trim)
        )
    estimation_data = estimation_data.with_columns(
        (
            pl.col(weights)
            * _compute_complier_weight_treated(treatment, instrument, _PROPENSITY_ID)
        ).alias(_COMPLIER_WEIGHT_TREATED_ID),
        (
            pl.col(weights)
            * _compute_complier_weight_control(treatment, instrument, _PROPENSITY_ID)
        ).alias(_COMPLIER_WEIGHT_CONTROL_ID),
    )

    y = estimation_data[outcome].to_numpy().astype(float)

    qtt = _get_quantile_differences(
        qs,
        _compute_quantiles(y, estimation_data[_COMPLIER_WEIGHT_TREATED_ID].to_numpy(), qs),
        _compute_quantiles(y, estimation_data[_COMPLIER_WEIGHT_CONTROL_ID].to_numpy(), qs),
    )
    m_t, m_c = estimation_data.select(
        _weighted_mean(outcome, _COMPLIER_WEIGHT_TREATED_ID),
        _weighted_mean(outcome, _COMPLIER_WEIGHT_CONTROL_ID),
    ).row(0)
    att = _get_mean_differences(m_t, m_c)

    complier_share = float(
        estimation_data.select(
            pl.col(_COMPLIER_WEIGHT_TREATED_ID).sum() / pl.col(weights).sum()
        ).item()
    )
    first_stage = _conditional_takeup_diff(estimation_data, treatment, instrument, weights)

    return _LocalQteIntermediateResult(
        qtt=qtt,
        att=att,
        complier_share=complier_share,
        first_stage=first_stage,
    )


def estimate_local_effects(
    ds: pl.DataFrame,
    outcome: ColumnName,
    treatment: ColumnName,
    instrument: ColumnName,
    qs: ArrayLike = MEDIAN,
    *,
    instrument_probability_formula: FormulaRhs | None = None,
    weights: ColumnName | None = None,
    trim: float = _DEFAULT_TRIM,
    bootstrap_config: BootstrapConfig | int = 100,
) -> LocalQteResult:
    """Estimate the quantile treatment effect for compliers.

    Combines `_compute_local_effects` with bootstrap standard errors. The
    estimand is the difference between the complier quantiles of the treated and
    untreated potential outcomes; the average effect is the local average
    treatment effect (LATE). Identification requires independence and exclusion
    of the instrument, monotonicity and a relevant first stage.

    Parameters
    ----------
    ds : polars.DataFrame
        Input data.
    outcome : ColumnName
        Name of the outcome column.
    treatment : ColumnName
        Name of the binary treatment column (``1`` treated, ``0`` not).
    instrument : ColumnName
        Name of the binary instrument column (``1`` encouraged, ``0`` not).
    qs : array_like, default=0.5
        Quantiles in ``(0, 1)`` at which to estimate the effect.
    instrument_probability_formula : FormulaRhs, optional
        Right-hand side of the logit model for ``P(Z = 1 | X)``. When omitted
        the instrument probability is taken to be constant.
    weights : ColumnName, optional
        Name of a column with sampling weights.
    trim : float, default=0.001
        Drop observations whose estimated instrument propensity lies outside
        ``[trim, 1 - trim]``. ``0`` keeps every observation.
    bootstrap_config : `BootstrapConfig` or int, default=100
        Bootstrap settings, or the number of replications.

    Returns
    -------
    `LocalQteResult`
        Complier quantile effects and the LATE with bootstrap standard errors.

    See Also
    --------
    `qte.datasets.load_card`, `qte.datasets.load_jtpa`
    """
    bootstrap_config = _make_bootstrap_config(bootstrap_config)
    qs = _as_iterable_array(qs)
    fcn = partial(
        _compute_local_effects,
        outcome=outcome,
        treatment=treatment,
        instrument=instrument,
        qs=qs,
        instrument_probability_formula=instrument_probability_formula,
        weights=weights,
        trim=trim,
    )
    point = fcn(ds)
    runs = _make_bootstrap_runs(
        ds,
        fcn,
        n_iter=bootstrap_config.n_iter,
        seed=bootstrap_config.seed,
        n_workers=bootstrap_config.n_workers,
    )
    qtt, att = _attach_standard_errors(point, runs)
    return LocalQteResult(
        qtt=qtt,
        att=att,
        outcome=outcome,
        group=None,
        complier_share=point.complier_share,
        first_stage=point.first_stage,
    )
