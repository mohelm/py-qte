import numpy as np
import polars as pl
from numpy.typing import NDArray

from qte.constants import PERCENTILES
from qte.cross_sectional.or_helpers import make_weights, predict_outcome_model
from qte.cross_sectional.results import _QteIntermediateResult
from qte.custom_types import CausalTarget, ColumnName, DataFrame
from qte.stats import estimate_outcome_model, get_quantiles


def compute_or_qte(
    ds: DataFrame,
    outcome_c: ColumnName,
    treatment_c: ColumnName,
    qs: NDArray[np.float64] = (0.5,),  # type: ignore
    *,
    weights_c: ColumnName | None = None,
    or_x_formular: str | None = None,
    target: CausalTarget = CausalTarget.QTE,
    or_quantiles: NDArray = PERCENTILES,
) -> _QteIntermediateResult:
    treated, control = (
        ds.filter(pl.col(treatment_c) == 1),
        ds.filter(pl.col(treatment_c) == 0),
    )
    ors_control = estimate_outcome_model(
        control, outcome_c, or_x_formular, or_quantiles
    )

    if target == CausalTarget.QTE:
        preds_control = predict_outcome_model(ors_control, ds)

        weights = make_weights(weights_c, ds, or_quantiles.shape[0])
        q_c = get_quantiles(qs, preds_control, weights)

        ors_treated = estimate_outcome_model(
            treated, outcome_c, or_x_formular, or_quantiles
        )
        preds_treated = predict_outcome_model(ors_treated, ds)
        q_t = get_quantiles(qs, preds_treated, weights)

    if target == CausalTarget.QTT:
        preds_control = predict_outcome_model(ors_control, treated)
        weights = make_weights(weights_c, treated, or_quantiles.shape[0])
        q_c = get_quantiles(qs, preds_control, weights)
        q_t = get_quantiles(
            qs, treated[outcome_c].to_numpy(), w=make_weights(weights_c, treated)
        )

    return _QteIntermediateResult(qs, q_t, q_c)
