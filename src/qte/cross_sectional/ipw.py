import numpy as np
import polars as pl
from numpy.typing import NDArray

from qte.cross_sectional.results import _QteIntermediateResult
from qte.custom_types import CausalTarget
from qte.names import (
    EFFECT_ID,
    MEAN_CONTROL_ID,
    MEAN_TREATED_ID,
    QUANTILE_CONTROL_VAL_ID,
    QUANTILE_ID,
    QUANTILE_TREATED_VAL_ID,
)
from qte.stats import estimate_propensity_score, get_quantiles


def compute_ipw_qte(
    ds: pl.DataFrame,
    outcome_c: str,
    treatment_c: str,
    ps_x_formular: str,
    qs: NDArray[np.float64],
    weight_c: str | None = None,
    target: CausalTarget = CausalTarget.QTE,
) -> _QteIntermediateResult:
    ps = estimate_propensity_score(ds, treatment_c, ps_x_formular).predict()

    treated, control = (
        ds.filter(pl.col(treatment_c) == 1),
        ds.filter(pl.col(treatment_c) == 0),
    )
    y_t, y_c = treated[outcome_c].to_numpy(), control[outcome_c].to_numpy()
    if target == CausalTarget.QTE:
        weights = 1 if weight_c is None else ds[weight_c].to_numpy()
        bw_treated = weights * ds[treatment_c].to_numpy() / ps
        bw_control = weights * (1 - ds[treatment_c]).to_numpy() / (1 - ps)
        y_all = ds[outcome_c].to_numpy()
        q_t = get_quantiles(qs, y_all, bw_treated)
        q_c = get_quantiles(qs, y_all, bw_control)
        mean_t, mean_c = (
            np.average(y_all, weights=bw_treated),
            np.average(y_all, weights=bw_control),
        )

    if target == CausalTarget.QTT:
        # TODO: lookhere.
        w_t = treated[weight_c].to_numpy() if weight_c is not None else None
        q_t = get_quantiles(qs, y_t, w_t)
        control_obs_selector = ds[treatment_c].to_numpy() == 0
        weights = 1 if weight_c is None else control[weight_c].to_numpy()
        ps_c = ps[control_obs_selector]
        bw_control = weights * ps_c / (1 - ps_c)
        q_c = get_quantiles(qs, y_c, bw_control)
        mean_t, mean_c = (
            np.average(y_t, weights=w_t),
            np.average(y_c, weights=bw_control),
        )

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
                MEAN_TREATED_ID: mean_t,
                MEAN_CONTROL_ID: mean_c,
            }
        ).with_columns((pl.col(MEAN_TREATED_ID) - pl.col(MEAN_CONTROL_ID)).alias(EFFECT_ID)),
    )
