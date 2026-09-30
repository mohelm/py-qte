from collections.abc import Iterable
from functools import partial

import numpy as np
import polars as pl
from numpy.typing import ArrayLike

from qte.bootstrap import BootstrapConfig, _make_bootstrap_config, _perform_bootstrap
from qte.constants import MEDIAN
from qte.cross_sectional.aipw import compute_aipw_effects
from qte.cross_sectional.custom_types import CausalTarget, Estimator
from qte.cross_sectional.ipw import compute_ipw_effects as compute_ipw_effects
from qte.cross_sectional.outcome_regression import compute_outcome_regression_effects
from qte.cross_sectional.results import QteResult
from qte.cross_sectional.results import _QteIntermediateResult as _QteIntermediateResult
from qte.cross_sectional.simulate import (
    simulate_covariate_data as simulate_covariate_data,
)
from qte.cross_sectional.simulate import simulate_simple_data as simulate_simple_data
from qte.cross_sectional.unadjusted import compute_unadjusted_effects
from qte.custom_types import (
    ColumnName,
    FormulaRhs,
)
from qte.names import EFFECT_ID, QUANTILE_ID, SE_ID


def get_statistics_from_bootstrap(
    boot_iter: Iterable[_QteIntermediateResult],
) -> _QteIntermediateResult:
    runs = list(boot_iter)
    agg = pl.col(EFFECT_ID).std().alias(SE_ID)
    qte = pl.concat([r.qtt for r in runs]).group_by(QUANTILE_ID).agg(agg)
    att = pl.concat([r.att for r in runs]).group_by([]).agg(agg)
    return _QteIntermediateResult(qte, att)


def estimate_unadjusted_effects(
    ds: pl.DataFrame,
    outcome: ColumnName,
    treatment: ColumnName,
    qs: ArrayLike = MEDIAN,
    *,
    weights: ColumnName | None = None,
    bootstrap_config: BootstrapConfig | int = 100,
) -> QteResult:
    """Estimate a quantile treatment effect by the unadjusted difference in quantiles.

    Compares the treated and control outcome distributions without covariate
    adjustment and attaches bootstrap standard errors.

    Parameters
    ----------
    ds : polars.DataFrame
        Input data.
    outcome : ColumnName
        Name of the outcome column.
    treatment : ColumnName
        Name of the binary treatment column (``1`` treated, ``0`` control).
    qs : array_like, default=0.5
        Quantiles in ``(0, 1)`` at which to estimate the effect.
    weights : ColumnName, optional
        Name of a column with sampling weights.
    bootstrap_config : `BootstrapConfig` or int, default=100
        Bootstrap settings, or the number of replications.

    Returns
    -------
    `QteResult`
        Estimated effects with bootstrap standard errors.

    See Also
    --------
    `estimate_ipw_effects`, `estimate_outcome_regression_effects`, `estimate_aipw_effects`
    """
    bootstrap_config = _make_bootstrap_config(bootstrap_config)
    qs = np.array(qs)
    fcn = partial(
        compute_unadjusted_effects,
        outcome=outcome,
        treatment=treatment,
        qs=qs,
        weights=weights,
    )
    estimate = fcn(ds)
    bs_it = _perform_bootstrap(
        ds,
        fcn=fcn,
        n_iter=bootstrap_config.n_iter,
        seed=bootstrap_config.seed,
        n_workers=bootstrap_config.n_workers,
    )
    bs_stats = get_statistics_from_bootstrap(bs_it)
    return QteResult(
        qtt=estimate.qtt.join(bs_stats.qtt, on=QUANTILE_ID),
        att=estimate.att.with_columns(bs_stats.att[SE_ID]),
        causal_target=CausalTarget.QTE,
        estimator=Estimator.UNADJUSTED,
        outcome=outcome,
        group=None,
    )


def estimate_ipw_effects(
    ds: pl.DataFrame,
    outcome: ColumnName,
    treatment: ColumnName,
    qs: ArrayLike = MEDIAN,
    *,
    propensity_score_formula: FormulaRhs,
    target: CausalTarget = CausalTarget.QTE,
    weights: ColumnName | None = None,
    bootstrap_config: BootstrapConfig | int = 100,
) -> QteResult:
    """Estimate a quantile treatment effect by inverse probability weighting.

    Reweights the outcome distribution with inverse propensity scores and
    attaches bootstrap standard errors.

    Parameters
    ----------
    ds : polars.DataFrame
        Input data.
    outcome : ColumnName
        Name of the outcome column.
    treatment : ColumnName
        Name of the binary treatment column (``1`` treated, ``0`` control).
    qs : array_like, default=0.5
        Quantiles in ``(0, 1)`` at which to estimate the effect.
    propensity_score_formula : FormulaRhs
        Right-hand side of the propensity score formula.
    target : `CausalTarget`, default=`CausalTarget.QTE`
        Estimand to target, ``QTE`` (population) or ``QTT`` (treated).
    weights : ColumnName, optional
        Name of a column with sampling weights.
    bootstrap_config : `BootstrapConfig` or int, default=100
        Bootstrap settings, or the number of replications.

    Returns
    -------
    `QteResult`
        Estimated effects with bootstrap standard errors.

    See Also
    --------
    `estimate_unadjusted_effects`, `estimate_outcome_regression_effects`, `estimate_aipw_effects`
    """
    qs = np.array(qs)
    bootstrap_config = _make_bootstrap_config(bootstrap_config)
    fcn = partial(
        compute_ipw_effects,
        outcome=outcome,
        treatment=treatment,
        propensity_score_formula=propensity_score_formula,
        qs=qs,
        weights=weights,
        target=target,
    )
    estimate = fcn(ds)
    bs_it = _perform_bootstrap(
        ds,
        fcn=fcn,
        n_iter=bootstrap_config.n_iter,
        seed=bootstrap_config.seed,
        n_workers=bootstrap_config.n_workers,
    )
    bs_stats = get_statistics_from_bootstrap(bs_it)
    return QteResult(
        qtt=estimate.qtt.join(bs_stats.qtt, on=QUANTILE_ID),
        att=estimate.att.with_columns(bs_stats.att[SE_ID]),
        causal_target=target,
        estimator=Estimator.IPW,
        outcome=outcome,
        group=None,
        propensity_score_formula=propensity_score_formula,
    )


def estimate_outcome_regression_effects(
    ds: pl.DataFrame,
    outcome: ColumnName,
    treatment: ColumnName,
    qs: ArrayLike = MEDIAN,
    *,
    outcome_regression_formula: FormulaRhs,
    target: CausalTarget = CausalTarget.QTE,
    weights: ColumnName | None = None,
    bootstrap_config: BootstrapConfig | int = 100,
) -> QteResult:
    """Estimate a quantile treatment effect by outcome regression.

    Predicts the counterfactual outcome distribution from quantile regressions
    and attaches bootstrap standard errors.

    Parameters
    ----------
    ds : polars.DataFrame
        Input data.
    outcome : ColumnName
        Name of the outcome column.
    treatment : ColumnName
        Name of the binary treatment column (``1`` treated, ``0`` control).
    qs : array_like, default=0.5
        Quantiles in ``(0, 1)`` at which to estimate the effect.
    outcome_regression_formula : FormulaRhs
        Right-hand side of the outcome regression formula.
    target : `CausalTarget`, default=`CausalTarget.QTE`
        Estimand to target, ``QTE`` (population) or ``QTT`` (treated).
    weights : ColumnName, optional
        Name of a column with sampling weights.
    bootstrap_config : `BootstrapConfig` or int, default=100
        Bootstrap settings, or the number of replications.

    Returns
    -------
    `QteResult`
        Estimated effects with bootstrap standard errors.

    See Also
    --------
    `estimate_unadjusted_effects`, `estimate_ipw_effects`, `estimate_aipw_effects`
    """
    qs = np.array(qs)
    fcn = partial(
        compute_outcome_regression_effects,
        outcome=outcome,
        treatment=treatment,
        outcome_regression_formula=outcome_regression_formula,
        qs=qs,
        weights=weights,
        target=target,
    )
    estimate = fcn(ds)
    bootstrap_config = _make_bootstrap_config(bootstrap_config)
    bs_it = _perform_bootstrap(
        ds,
        fcn=fcn,
        n_iter=bootstrap_config.n_iter,
        seed=bootstrap_config.seed,
        n_workers=bootstrap_config.n_workers,
    )
    bs_stats = get_statistics_from_bootstrap(bs_it)
    return QteResult(
        qtt=estimate.qtt.join(bs_stats.qtt, on=QUANTILE_ID),
        att=estimate.att.with_columns(bs_stats.att[SE_ID]),
        causal_target=target,
        estimator=Estimator.OR,
        outcome=outcome,
        group=None,
        outcome_regression_formula=outcome_regression_formula,
    )


def estimate_aipw_effects(
    ds: pl.DataFrame,
    outcome: ColumnName,
    treatment: ColumnName,
    qs: ArrayLike = (0.5,),
    *,
    propensity_score_formula: FormulaRhs,
    outcome_regression_formula: FormulaRhs,
    target: CausalTarget = CausalTarget.QTE,
    weights: ColumnName | None = None,
    bootstrap_config: BootstrapConfig | int = 100,
) -> QteResult:
    """Estimate a quantile treatment effect by augmented inverse probability weighting.

    Combines the propensity score and outcome regression into a doubly robust
    estimator and attaches bootstrap standard errors.

    Parameters
    ----------
    ds : polars.DataFrame
        Input data.
    outcome : ColumnName
        Name of the outcome column.
    treatment : ColumnName
        Name of the binary treatment column (``1`` treated, ``0`` control).
    qs : array_like, default=0.5
        Quantiles in ``(0, 1)`` at which to estimate the effect.
    propensity_score_formula : FormulaRhs
        Right-hand side of the propensity score formula.
    outcome_regression_formula : FormulaRhs
        Right-hand side of the outcome regression formula.
    target : `CausalTarget`, default=`CausalTarget.QTE`
        Estimand to target, ``QTE`` (population) or ``QTT`` (treated).
    weights : ColumnName, optional
        Name of a column with sampling weights.
    bootstrap_config : `BootstrapConfig` or int, default=100
        Bootstrap settings, or the number of replications.

    Returns
    -------
    `QteResult`
        Estimated effects with bootstrap standard errors.

    See Also
    --------
    `estimate_unadjusted_effects`, `estimate_ipw_effects`, `estimate_outcome_regression_effects`
    """
    if weights is None:
        weights = "_w"
        ds = ds.with_columns(pl.lit(1).alias(weights))
    qs = np.array(qs)
    fcn = partial(
        compute_aipw_effects,
        outcome=outcome,
        treatment=treatment,
        outcome_regression_formula=outcome_regression_formula,
        propensity_score_formula=propensity_score_formula,
        qs=qs,
        weights=weights,
        target=target,
    )
    estimate = fcn(ds)
    bootstrap_config = _make_bootstrap_config(bootstrap_config)
    bs_it = _perform_bootstrap(
        ds,
        fcn=fcn,
        n_iter=bootstrap_config.n_iter,
        seed=bootstrap_config.seed,
        n_workers=bootstrap_config.n_workers,
    )
    bs_stats = get_statistics_from_bootstrap(bs_it)
    return QteResult(
        qtt=estimate.qtt.join(bs_stats.qtt, on=QUANTILE_ID),
        att=estimate.att.with_columns(bs_stats.att[SE_ID]),
        causal_target=target,
        estimator=Estimator.AIPW,
        outcome=outcome,
        group=None,
        propensity_score_formula=propensity_score_formula,
        outcome_regression_formula=outcome_regression_formula,
    )
