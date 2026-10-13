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
the first-step propensity model and for inverting the weighted CDFs. Two
weighting schemes are available: Abadie's signed weights (`Estimator.SIGNED`)
and the positive-weight projection of Stata's ``ivqte, positive``
(`Estimator.POSITIVE`).
"""

from dataclasses import dataclass
from functools import partial

import numpy as np
import polars as pl
import statsmodels.api as sm
from formulaic import Formula
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
from qte.iv.custom_types import Estimator
from qte.iv.results import UnconditionalLocalQteResult
from qte.stats import estimate_propensity_score

_EPS = 1e-8
_DEFAULT_TRIM = 0.001
_PROPENSITY_ID = "_propensity"
_KAPPA_TREATED_ID = "_kappa_treated"
_COMPLIER_WEIGHT_TREATED_ID = "_complier_weight_treated"
_COMPLIER_WEIGHT_CONTROL_ID = "_complier_weight_control"
_POSITIVE_WEIGHT_ID = "_positive_weight"


@dataclass(frozen=True)
class _LocalQteIntermediateResult:
    """Point estimates from one (bootstrap) draw."""

    qtt: pl.DataFrame
    att: pl.DataFrame
    complier_share: float
    first_stage: float


def _compute_kappa_treated(instrument: ColumnName, propensity: ColumnName) -> pl.Expr:
    """Polars expression for the treated complier weight factor.

    ``(Z - p) / (p (1 - p))`` is Abadie's treated weight ``D (Z - p) / (p (1 - p))``
    with the ``D`` selection dropped; the control weight is ``(D - 1)`` times the
    same factor.
    """
    p = pl.col(propensity)
    return (pl.col(instrument) - p) / (p * (1.0 - p))


def _weighted_mean(value: ColumnName, weight: ColumnName) -> pl.Expr:
    """Polars expression for the ``weight``-weighted mean of ``value``."""
    return (pl.col(weight) * pl.col(value)).sum() / pl.col(weight).sum()


def _estimate_instrument_propensity(
    ds: pl.DataFrame,
    instrument: ColumnName,
    instrument_probability_formula: FormulaRhs | None,
    weights: ColumnName,
) -> NDArray[np.float64]:
    """Estimate ``p(X) = P(Z = 1 | X)``, constant when no formula is given."""
    p = (
        np.full(
            len(ds),
            fill_value=np.average(ds[instrument].to_numpy(), weights=ds[weights].to_numpy()),
        )
        if instrument_probability_formula is None  # For speed purposes.
        else estimate_propensity_score(
            ds, instrument, instrument_probability_formula, weights
        ).predict()
    )
    # Avoid division blow-ups at the boundary; observations this extreme only
    # arise under (near) deterministic take-up.
    return np.clip(p, _EPS, 1.0 - _EPS)


def _compute_quantiles(
    y: NDArray[np.float64],
    w: NDArray[np.float64],
    qs: NDArray[np.float64],
    *,
    first_crossing: bool = False,
) -> NDArray[np.float64]:
    """Quantiles of the weighted empirical CDF of ``y``.

    The complier weights can be signed, so the weighted CDF is not guaranteed to
    be monotone. By default we use the same inversion as Stata's ``ivqte``: the
    ``tau`` quantile is the value at rank ``#{F(y) <= tau}`` (clamped to at
    least one). With ``first_crossing`` we take the first outcome whose CDF
    reaches ``tau`` instead, which is the usual quantile when the CDF is
    monotone (used by the positive-weight estimator).
    """
    y = np.asarray(y, dtype=float)
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
    if first_crossing:
        return values[np.clip(np.searchsorted(cdf, qs, side="left"), 0, len(values) - 1)]
    counts = np.array([(cdf <= tau).sum() for tau in qs])
    return values[np.maximum(counts - 1, 0)]


def _project_complier_weights(
    estimation_data: pl.DataFrame,
    outcome: ColumnName,
    treatment: ColumnName,
    weights: ColumnName,
) -> pl.Series:
    """Project ``kappa`` on the treatment and outcome.

    Fits ``WLS(kappa ~ treatment * outcome)`` on the whole sample. The model is
    saturated in the binary treatment, hence identical to a separate regression
    per arm. Returns ``weights * fitted``; the caller applies the sign ``2 D - 1``
    and keeps the positive part, which are the asymptotically positive complier
    weights of Stata's ``ivqte, positive``.
    """
    response, design = Formula.from_spec(
        f"{_KAPPA_TREATED_ID} ~ {treatment} * {outcome}"
    ).get_model_matrix(estimation_data, output="numpy")
    fitted = (
        sm.WLS(response.ravel(), design, weights=estimation_data[weights].to_numpy())
        .fit()
        .fittedvalues
    )
    return pl.Series(estimation_data[weights].to_numpy() * fitted)


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
    return ds.select((at_one - at_zero).alias("first_stage")).item()


def _compute_effects_via_signed_weights(
    estimation_data: pl.DataFrame,
    qs: NDArray[np.float64],
    outcome: ColumnName,
    treatment: ColumnName,
    weights: ColumnName,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Complier quantiles and means from the signed Abadie weights."""
    # The treated weight is D * kappa_treated; the control factor is -kappa_treated,
    # so multiplying by (D - 1) gives the control weight and zero for the treated.
    estimation_data = estimation_data.with_columns(
        (pl.col(weights) * pl.col(treatment) * pl.col(_KAPPA_TREATED_ID)).alias(
            _COMPLIER_WEIGHT_TREATED_ID
        ),
        (pl.col(weights) * (pl.col(treatment) - 1.0) * pl.col(_KAPPA_TREATED_ID)).alias(
            _COMPLIER_WEIGHT_CONTROL_ID
        ),
    )
    y = estimation_data[outcome].to_numpy()
    qtt = _get_quantile_differences(
        qs,
        _compute_quantiles(y, estimation_data[_COMPLIER_WEIGHT_TREATED_ID].to_numpy(), qs),
        _compute_quantiles(y, estimation_data[_COMPLIER_WEIGHT_CONTROL_ID].to_numpy(), qs),
    )
    m_t, m_c = estimation_data.select(
        _weighted_mean(outcome, _COMPLIER_WEIGHT_TREATED_ID),
        _weighted_mean(outcome, _COMPLIER_WEIGHT_CONTROL_ID),
    ).row(0)
    return qtt, _get_mean_differences(m_t, m_c)


def _compute_effects_via_positive_weights(
    estimation_data: pl.DataFrame,
    qs: NDArray[np.float64],
    outcome: ColumnName,
    treatment: ColumnName,
    weights: ColumnName,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Complier quantiles and means from the projected positive weights."""
    projected = _project_complier_weights(estimation_data, outcome, treatment, weights)
    estimation_data = estimation_data.with_columns(
        (projected * (2.0 * pl.col(treatment) - 1.0)).alias(_POSITIVE_WEIGHT_ID)
    ).filter(pl.col(_POSITIVE_WEIGHT_ID) > 0)

    treated = estimation_data.filter(pl.col(treatment) == 1)
    control = estimation_data.filter(pl.col(treatment) == 0)
    y_trt = treated[outcome].to_numpy()
    weights_trt = treated[_POSITIVE_WEIGHT_ID].to_numpy()
    y_ctrl = control[outcome].to_numpy()
    weights_ctrl = control[_POSITIVE_WEIGHT_ID].to_numpy()

    return (
        _get_quantile_differences(
            qs,
            _compute_quantiles(y_trt, weights_trt, qs, first_crossing=True),
            _compute_quantiles(y_ctrl, weights_ctrl, qs, first_crossing=True),
        ),
        _get_mean_differences(
            np.average(y_trt, weights=weights_trt),
            np.average(y_ctrl, weights=weights_ctrl),
        ),
    )


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
    estimator: Estimator = Estimator.SIGNED,
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
    estimator : `Estimator`, default=`Estimator.SIGNED`
        Weighting scheme. ``SIGNED`` inverts the signed complier CDF;
        ``POSITIVE`` projects the weights and differences weighted quantiles.

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

    # Abadie's treated complier weight factor, centered as in Stata's `ivqte` so
    # that its sample mean is zero, as it is in the population. The treated weight
    # is `D * kappa_treated` and the control weight is `(D - 1) * kappa_treated`.
    estimation_data = estimation_data.with_columns(
        _compute_kappa_treated(instrument, _PROPENSITY_ID).alias(_KAPPA_TREATED_ID)
    ).with_columns(
        (pl.col(_KAPPA_TREATED_ID) - _weighted_mean(_KAPPA_TREATED_ID, weights)).alias(
            _KAPPA_TREATED_ID
        )
    )

    if estimator is Estimator.POSITIVE:
        qtt, att = _compute_effects_via_positive_weights(
            estimation_data, qs, outcome, treatment, weights
        )
    else:
        qtt, att = _compute_effects_via_signed_weights(
            estimation_data, qs, outcome, treatment, weights
        )

    complier_share = estimation_data.select(
        (pl.col(weights) * pl.col(treatment) * pl.col(_KAPPA_TREATED_ID)).sum()
        / pl.col(weights).sum()
    ).item()
    first_stage = _conditional_takeup_diff(estimation_data, treatment, instrument, weights)

    return _LocalQteIntermediateResult(
        qtt=qtt,
        att=att,
        complier_share=complier_share,
        first_stage=first_stage,
    )


def estimate_unconditional_local_effects(
    ds: pl.DataFrame,
    outcome: ColumnName,
    treatment: ColumnName,
    instrument: ColumnName,
    qs: ArrayLike = MEDIAN,
    *,
    instrument_probability_formula: FormulaRhs | None = None,
    weights: ColumnName | None = None,
    trim: float = _DEFAULT_TRIM,
    estimator: Estimator = Estimator.SIGNED,
    bootstrap_config: BootstrapConfig | int = 100,
) -> UnconditionalLocalQteResult:
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
    estimator : `Estimator`, default=`Estimator.SIGNED`
        Weighting scheme, see `_compute_local_effects`.
    bootstrap_config : `BootstrapConfig` or int, default=100
        Bootstrap settings, or the number of replications.

    Returns
    -------
    `UnconditionalLocalQteResult`
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
        estimator=estimator,
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
    return UnconditionalLocalQteResult(
        qtt=qtt,
        att=att,
        outcome=outcome,
        group=None,
        complier_share=point.complier_share,
        first_stage=point.first_stage,
        estimator=estimator,
    )
