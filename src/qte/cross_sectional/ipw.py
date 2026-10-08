import numpy as np
import polars as pl
from numpy.typing import NDArray

from qte.cross_sectional.custom_types import CausalTarget
from qte.cross_sectional.results import _QteIntermediateResult
from qte.custom_types import ColumnName, FormulaRhs
from qte.helpers import _get_mean_differences, _get_quantile_differences
from qte.stats import estimate_propensity_score, get_quantiles


def compute_ipw_effects(
    ds: pl.DataFrame,
    outcome: ColumnName,
    treatment: ColumnName,
    propensity_score_formula: FormulaRhs,
    qs: NDArray[np.float64],
    weights: ColumnName | None = None,
    target: CausalTarget = CausalTarget.QTE,
) -> _QteIntermediateResult:
    ps = estimate_propensity_score(ds, treatment, propensity_score_formula, weights).predict()

    treated, control = (
        ds.filter(pl.col(treatment) == 1),
        ds.filter(pl.col(treatment) == 0),
    )
    y_t, y_c = treated[outcome].to_numpy(), control[outcome].to_numpy()
    if target == CausalTarget.QTE:
        sample_weights = 1 if weights is None else ds[weights].to_numpy()
        bw_treated = sample_weights * ds[treatment].to_numpy() / ps
        bw_control = sample_weights * (1 - ds[treatment]).to_numpy() / (1 - ps)
        y_all = ds[outcome].to_numpy()
        q_t = get_quantiles(qs, y_all, bw_treated)
        q_c = get_quantiles(qs, y_all, bw_control)
        mean_t, mean_c = (
            np.average(y_all, weights=bw_treated),
            np.average(y_all, weights=bw_control),
        )

    if target == CausalTarget.QTT:
        # TODO: lookhere.
        w_t = treated[weights].to_numpy() if weights is not None else None
        q_t = get_quantiles(qs, y_t, w_t)
        control_obs_selector = ds[treatment].to_numpy() == 0
        sample_weights = 1 if weights is None else control[weights].to_numpy()
        ps_c = ps[control_obs_selector]
        bw_control = sample_weights * ps_c / (1 - ps_c)
        q_c = get_quantiles(qs, y_c, bw_control)
        mean_t, mean_c = (
            np.average(y_t, weights=w_t),
            np.average(y_c, weights=bw_control),
        )

    return _QteIntermediateResult(
        qtt=_get_quantile_differences(qs, q_t, q_c),
        att=_get_mean_differences(mean_t, mean_c),
    )
