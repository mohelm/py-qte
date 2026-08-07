import numpy as np
import polars as pl
from numpy.typing import ArrayLike

from qte.cross_sectional.results import _QteIntermediateResult
from qte.custom_types import ColumnName
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

    treated, control = (
        ds.filter(pl.col(treatment_c) == 1.0),
        ds.filter(pl.col(treatment_c) == 0.0),
    )
    q_t = get_quantiles(
        qs,
        treated[outcome_c].to_numpy(),
        weight_c if weight_c is None else treated[weight_c].to_numpy(),
    )
    q_c = get_quantiles(
        qs,
        control[outcome_c].to_numpy(),
        weight_c if weight_c is None else control[weight_c].to_numpy(),
    )
    return _QteIntermediateResult(qs, q_t, q_c)
