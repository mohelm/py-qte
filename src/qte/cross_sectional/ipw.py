import numpy as np
import polars as pl
from numpy.typing import NDArray

from qte.cross_sectional.results import _QteIntermediateResult
from qte.custom_types import CausalTarget
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

    if target == CausalTarget.QTE:
        weights = 1 if weight_c is None else ds[weight_c].to_numpy()
        bw_treated = weights * ds[treatment_c].to_numpy() / ps
        bw_control = weights * (1 - ds[treatment_c]).to_numpy() / (1 - ps)
        q_t = get_quantiles(qs, ds[outcome_c].to_numpy(), bw_treated)
        q_c = get_quantiles(qs, ds[outcome_c].to_numpy(), bw_control)

    if target == CausalTarget.QTT:
        treated, control = (
            ds.filter(pl.col(treatment_c) == 1),
            ds.filter(pl.col(treatment_c) == 0),
        )
        q_t = get_quantiles(
            qs,
            treated[outcome_c].to_numpy(),
            treated[weight_c].to_numpy() if weight_c is not None else None,
        )
        control_obs_selector = ds[treatment_c].to_numpy() == 0
        weights = (
            1 if weight_c is None else ds[weight_c].to_numpy()[:, control_obs_selector]
        )
        ps_c = ps[control_obs_selector]
        bw_control = weights * ps_c / (1 - ps_c)
        q_c = get_quantiles(qs, control[outcome_c].to_numpy(), bw_control)

    return _QteIntermediateResult(qs, q_t, q_c)
