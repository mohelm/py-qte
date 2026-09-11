import numpy as np
import polars as pl
from numpy.typing import NDArray

from qte.constants import PERCENTILES
from qte.cross_sectional.or_helpers import make_weights, predict_outcome_model
from qte.cross_sectional.results import _QteIntermediateResult
from qte.custom_types import CausalTarget, ColumnName, DataFrame
from qte.names import (
    EFFECT_ID,
    MEAN_CONTROL_ID,
    MEAN_TREATED_ID,
    QUANTILE_CONTROL_VAL_ID,
    QUANTILE_ID,
    QUANTILE_TREATED_VAL_ID,
)
from qte.stats import estimate_outcome_model, get_quantiles


def compute_or_qte(
    ds: DataFrame,
    outcome_c: ColumnName,
    treatment_c: ColumnName,
    qs: NDArray[np.float64] = (0.5,),  # type: ignore
    *,
    or_x_formular: str,
    weights_c: ColumnName | None = None,
    target: CausalTarget = CausalTarget.QTE,
    or_quantiles: NDArray = PERCENTILES,
) -> _QteIntermediateResult:
    treated, control = (
        ds.filter(pl.col(treatment_c) == 1),
        ds.filter(pl.col(treatment_c) == 0),
    )
    ors_control = estimate_outcome_model(control, outcome_c, or_x_formular, or_quantiles)

    if target == CausalTarget.QTE:
        preds_control = predict_outcome_model(ors_control, ds)

        weights = make_weights(weights_c, ds, or_quantiles.shape[0])
        q_c = get_quantiles(qs, preds_control, weights)

        ors_treated = estimate_outcome_model(treated, outcome_c, or_x_formular, or_quantiles)
        preds_treated = predict_outcome_model(ors_treated, ds)
        q_t = get_quantiles(qs, preds_treated, weights)
        mean_t, mean_c = (
            np.average(preds_treated, weights=weights),
            np.average(preds_control, weights=weights),
        )

    if target == CausalTarget.QTT:
        preds_control = predict_outcome_model(ors_control, treated)
        w_c = make_weights(weights_c, treated, or_quantiles.shape[0])
        q_c = get_quantiles(qs, preds_control, w_c)
        w_t = make_weights(weights_c, treated)
        y_t = treated[outcome_c].to_numpy()
        q_t = get_quantiles(qs, y_t, w=w_t)
        mean_t, mean_c = (
            np.average(y_t, weights=w_t),
            np.average(preds_control, weights=w_c),
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
