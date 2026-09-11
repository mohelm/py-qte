import numpy as np
import polars as pl
from numpy.typing import ArrayLike

from qte.cross_sectional.results import _QteIntermediateResult
from qte.custom_types import ColumnName
from qte.names import (
    EFFECT_ID,
    MEAN_CONTROL_ID,
    MEAN_TREATED_ID,
    QUANTILE_CONTROL_VAL_ID,
    QUANTILE_ID,
    QUANTILE_TREATED_VAL_ID,
)
from qte.stats import get_quantiles


def compute_simple_qte(
    ds: pl.DataFrame,
    outcome_c: ColumnName,
    treatment_c: ColumnName,
    qs: ArrayLike,
    *,
    weight_c: str | None = None,
) -> _QteIntermediateResult:
    qs = np.array(qs)

    # TODO: look into that
    treated, control = (
        ds.filter(pl.col(treatment_c) == 1.0),
        ds.filter(pl.col(treatment_c) == 0.0),
    )
    w_t = weight_c if weight_c is None else treated[weight_c].to_numpy()
    w_c = weight_c if weight_c is None else control[weight_c].to_numpy()
    y_t, y_c = treated[outcome_c].to_numpy(), control[outcome_c].to_numpy()
    return _QteIntermediateResult(
        qtt=pl.DataFrame(
            {
                QUANTILE_ID: qs,
                QUANTILE_TREATED_VAL_ID: get_quantiles(qs, y_t, w_t),
                QUANTILE_CONTROL_VAL_ID: get_quantiles(qs, y_c, w_c),
            }
        ).with_columns(
            (pl.col(QUANTILE_TREATED_VAL_ID) - pl.col(QUANTILE_CONTROL_VAL_ID)).alias(
                EFFECT_ID
            )
        ),
        att=pl.DataFrame(
            {
                MEAN_TREATED_ID: np.average(y_t, weights=w_t),
                MEAN_CONTROL_ID: np.average(y_c, weights=w_c),
            }
        ).with_columns(
            (pl.col(MEAN_TREATED_ID) - pl.col(MEAN_CONTROL_ID)).alias(EFFECT_ID)
        ),
    )
