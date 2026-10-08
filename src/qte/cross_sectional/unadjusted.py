import numpy as np
import polars as pl
from numpy.typing import ArrayLike

from qte.cross_sectional.results import _QteIntermediateResult
from qte.custom_types import ColumnName
from qte.helpers import _as_iterable_array, _get_mean_differences, _get_quantile_differences
from qte.stats import get_quantiles


def compute_unadjusted_effects(
    ds: pl.DataFrame,
    outcome: ColumnName,
    treatment: ColumnName,
    qs: ArrayLike,
    *,
    weights: ColumnName | None = None,
) -> _QteIntermediateResult:
    qs = _as_iterable_array(qs)

    treated, control = (
        ds.filter(pl.col(treatment) == 1.0),
        ds.filter(pl.col(treatment) == 0.0),
    )
    w_t = weights if weights is None else treated[weights].to_numpy()
    w_c = weights if weights is None else control[weights].to_numpy()
    y_t, y_c = treated[outcome].to_numpy(), control[outcome].to_numpy()
    return _QteIntermediateResult(
        qtt=_get_quantile_differences(qs, get_quantiles(qs, y_t, w_t), get_quantiles(qs, y_c, w_c)),
        att=_get_mean_differences(np.average(y_t, weights=w_t), np.average(y_c, weights=w_c)),
    )
