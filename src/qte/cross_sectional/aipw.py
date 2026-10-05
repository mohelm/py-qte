import numpy as np
import polars as pl
from numpy.typing import ArrayLike, NDArray

from qte.cross_sectional.custom_types import (
    CausalTarget,
    OutcomeRegressionConfig,
    resolve_grid,
)
from qte.cross_sectional.or_helpers import make_weights, predict_outcome_model
from qte.cross_sectional.results import _QteIntermediateResult
from qte.custom_types import ColumnName, FormulaRhs
from qte.names import (
    EFFECT_ID,
    MEAN_CONTROL_ID,
    MEAN_TREATED_ID,
    QUANTILE_CONTROL_VAL_ID,
    QUANTILE_ID,
    QUANTILE_TREATED_VAL_ID,
)
from qte.stats import estimate_outcome_model, estimate_propensity_score, get_quantiles


def _compute_aipw_quantiles(
    qs: NDArray,
    outcome_grid: NDArray,
    or_preds: NDArray,
    outcomes: NDArray,
    treatment_status: NDArray,
    propensity_scores: NDArray,
    sampling_weights: NDArray,
) -> NDArray:

    n_qs = or_preds.shape[1]
    prop_score_weights = sampling_weights * treatment_status / propensity_scores
    w = np.repeat((sampling_weights - prop_score_weights) / n_qs, n_qs)
    or_preds_flat = or_preds.flatten()
    sorter = np.argsort(or_preds_flat)
    or_preds_flat_sorted = or_preds_flat[sorter]
    cdf_or = np.concatenate(([0.0], np.cumsum(w[sorter])))

    sorter = np.argsort(outcomes)
    outcome_sorted = outcomes[sorter]
    cdf_outcome = np.concatenate(([0.0], np.cumsum(prop_score_weights[sorter])))

    idx_or = np.searchsorted(or_preds_flat_sorted, outcome_grid, side="right")
    idx_outcome = np.searchsorted(outcome_sorted, outcome_grid, side="right")
    f0 = (cdf_or[idx_or] + cdf_outcome[idx_outcome]) / np.sum(sampling_weights)

    f0 = np.maximum.accumulate(np.maximum(np.minimum(1, f0), 0))
    u, indices = np.unique(f0, return_index=True)
    return np.interp(qs, u, outcome_grid[indices])


def _compute_aipw_mean(
    or_quantile_preds: NDArray,
    outcomes: NDArray,
    treatment_status: NDArray,
    propensity_scores: NDArray,
    sampling_weights: NDArray,
) -> float:
    mu_hat = or_quantile_preds.mean(axis=1)
    return np.average(
        mu_hat + (treatment_status / propensity_scores) * (outcomes - mu_hat),
        weights=sampling_weights,
    )


def compute_aipw_effects(
    ds: pl.DataFrame,
    outcome: ColumnName,
    treatment: ColumnName,
    qs: ArrayLike = (0.5,),
    *,
    propensity_score_formula: FormulaRhs,
    outcome_regression_config: OutcomeRegressionConfig,
    weights: ColumnName,
    target: CausalTarget = CausalTarget.QTE,
) -> _QteIntermediateResult:

    qs = np.array(qs)
    grid = resolve_grid(outcome_regression_config.grid)
    treated, control = (ds.filter(pl.col(treatment) == 1), ds.filter(pl.col(treatment) == 0))
    ps = estimate_propensity_score(ds, treatment, propensity_score_formula, weights).predict()
    ors_control = estimate_outcome_model(
        control,
        outcome,
        outcome_regression_config.formula,
        grid,
        weights,
        algorithm=outcome_regression_config.algorithm,
    )
    preds = predict_outcome_model(ors_control, ds, flatten=False)

    outcome_grid = ds[outcome].unique().sort().to_numpy()
    w_t = treated[weights].to_numpy()
    y_t = treated[outcome].to_numpy()

    w_all = make_weights(weights, ds, 1)
    if w_all is None:
        w_all = np.ones(ds.shape[0])
    d = ds[treatment].to_numpy()

    if target == CausalTarget.QTE:
        q_c = _compute_aipw_quantiles(
            qs,
            outcome_grid,
            preds,
            ds[outcome].to_numpy(),
            treatment_status=1 - d,
            propensity_scores=1 - ps,
            sampling_weights=w_all,
        )
        m_c = _compute_aipw_mean(preds, ds[outcome].to_numpy(), 1 - d, 1 - ps, w_all)

        ors_treated = estimate_outcome_model(
            treated,
            outcome,
            outcome_regression_config.formula,
            grid,
            weights,
            algorithm=outcome_regression_config.algorithm,
        )
        preds_treated = predict_outcome_model(ors_treated, ds, flatten=False)
        q_t = _compute_aipw_quantiles(
            qs,
            outcome_grid,
            preds_treated,
            ds[outcome].to_numpy(),
            treatment_status=d,
            propensity_scores=ps,
            sampling_weights=w_all,
        )
        m_t = _compute_aipw_mean(preds_treated, ds[outcome].to_numpy(), d, ps, w_all)

    if target == CausalTarget.QTT:
        q_c = _compute_aipw_quantiles(
            qs,
            outcome_grid,
            preds,
            ds[outcome].to_numpy(),
            treatment_status=1 - d,
            propensity_scores=1 - ps,
            sampling_weights=w_all * ps,
        )
        m_c = _compute_aipw_mean(preds, ds[outcome].to_numpy(), 1 - d, 1 - ps, w_all * ps)

        q_t = get_quantiles(qs, treated[outcome].to_numpy(), w=make_weights(weights, treated))
        m_t = np.average(y_t, weights=w_t)
    return _QteIntermediateResult(
        qtt=pl.DataFrame(
            {
                QUANTILE_ID: qs,
                QUANTILE_TREATED_VAL_ID: q_t,
                QUANTILE_CONTROL_VAL_ID: q_c,
            }
        ).with_columns(
            (pl.col(QUANTILE_TREATED_VAL_ID) - pl.col(QUANTILE_CONTROL_VAL_ID)).alias(EFFECT_ID)
        ),
        att=pl.DataFrame(
            {
                MEAN_CONTROL_ID: m_c,
                MEAN_TREATED_ID: m_t,
            }
        ).with_columns((pl.col(MEAN_TREATED_ID) - pl.col(MEAN_CONTROL_ID)).alias(EFFECT_ID)),
    )
