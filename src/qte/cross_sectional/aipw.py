import numpy as np
import polars as pl
from numpy.typing import ArrayLike, NDArray

from qte.constants import PERCENTILES
from qte.cross_sectional.or_helpers import make_weights, predict_outcome_model
from qte.cross_sectional.results import _QteIntermediateResult
from qte.custom_types import CausalTarget
from qte.stats import estimate_outcome_model, estimate_propensity_score, get_quantiles


def _compute_f(outcome_grid_val, preds, outcome, weight1, weight2, adj) -> float:
    f_cond = (preds <= outcome_grid_val).mean(axis=1)
    return (
        np.mean(weight1 * f_cond + weight2 * ((outcome <= outcome_grid_val) - f_cond))
        / adj
    )


def _compute_aipw_term_for_qte(
    qs: NDArray,
    grid: NDArray,
    or_preds: NDArray,
    outcome: NDArray,
    weight2: NDArray,
    weight1: ArrayLike,
    adj: float = 1,
) -> NDArray:
    n_obs, n_qs = or_preds.shape
    weights = np.repeat(
        (np.broadcast_to(weight1, n_obs) - np.broadcast_to(weight2, n_obs)) / n_qs, n_qs
    )  # (n_obs x n_qs times 1)
    or_preds_flat = or_preds.flatten()
    sorter = np.argsort(or_preds_flat)
    or_preds_flat_sorted = or_preds_flat[sorter]
    cdf_or = np.concatenate(([0.0], np.cumsum(weights[sorter])))

    sorter = np.argsort(outcome)
    outcome_sorted = outcome[sorter]
    cdf_outcome = np.concatenate((
        [0.0],
        np.cumsum(np.broadcast_to(weight2, n_obs)[sorter]),
    ))
    idx_or = np.searchsorted(or_preds_flat_sorted, grid, side="right")
    idx_outcome = np.searchsorted(outcome_sorted, grid, side="right")
    f0 = (cdf_or[idx_or] + cdf_outcome[idx_outcome]) / (n_obs * adj)

    f0 = np.maximum.accumulate(np.maximum(np.minimum(1, f0), 0))
    u, indices = np.unique(f0, return_index=True)
    return np.interp(qs, u, grid[indices])


def compute_aipw_qte(
    ds: pl.DataFrame,
    outcome_c: str,
    treatment_c: str,
    qs: NDArray[np.float64] = (0.5,),
    *,
    ps_x_formular: str | None = None,
    or_x_formular: str | None = None,
    weights_c: str | None = None,
    target: CausalTarget = CausalTarget.QTE,
    or_quantiles: NDArray = PERCENTILES,
):
    treated, control = (
        ds.filter(pl.col(treatment_c) == 1),
        ds.filter(pl.col(treatment_c) == 0),
    )
    ps = estimate_propensity_score(ds, treatment_c, ps_x_formular).predict()
    ors_control = estimate_outcome_model(
        control, outcome_c, or_x_formular, or_quantiles
    )
    preds = predict_outcome_model(ors_control, ds, flatten=False)

    outcome_grid = ds[outcome_c].unique().sort().to_numpy()

    if target == CausalTarget.QTE:
        q0 = _compute_aipw_term_for_qte(
            qs,
            outcome_grid,
            preds,
            ds[outcome_c].to_numpy(),
            weight1=1,
            weight2=(1 - ds[treatment_c].to_numpy()) / (1 - ps),
        )

        ors_treated = estimate_outcome_model(
            treated, outcome_c, or_x_formular, or_quantiles
        )
        preds_treated = predict_outcome_model(ors_treated, ds, flatten=False)
        q1 = _compute_aipw_term_for_qte(
            qs,
            outcome_grid,
            preds_treated,
            ds[outcome_c].to_numpy(),
            weight1=1,
            weight2=ds[treatment_c].to_numpy() / ps,
        )

    if target == CausalTarget.QTT:
        q0 = _compute_aipw_term_for_qte(
            qs,
            outcome_grid,
            preds,
            ds[outcome_c].to_numpy(),
            weight1=ps,
            weight2=((1 - ds[treatment_c].to_numpy()) * ps) / (1 - ps),
            adj=ps.mean(),
        )
        q1 = get_quantiles(
            qs, treated[outcome_c].to_numpy(), w=make_weights(weights_c, treated)
        )
    return _QteIntermediateResult(qs, q1, q0)
