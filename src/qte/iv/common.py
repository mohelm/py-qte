"""Internals shared by the conditional and unconditional IV estimators."""

import numpy as np
import polars as pl
from numpy.typing import NDArray

from qte.custom_types import ColumnName, FormulaRhs
from qte.stats import estimate_propensity_score

_EPS = 1e-8
_DEFAULT_TRIM = 0.001
_PROPENSITY_ID = "_propensity"


def _estimate_instrument_propensity(
    ds: pl.DataFrame,
    instrument: ColumnName,
    instrument_probability_formula: FormulaRhs | None,
    weights: ColumnName,
) -> NDArray[np.float64]:
    """Estimate ``p(X) = P(Z = 1 | X)``, constant when no formula is given."""
    p = (
        np.full(
            len(ds),
            fill_value=np.average(ds[instrument].to_numpy(), weights=ds[weights].to_numpy()),
        )
        if instrument_probability_formula is None  # For speed purposes.
        else estimate_propensity_score(
            ds, instrument, instrument_probability_formula, weights
        ).predict()
    )
    # Avoid division blow-ups at the boundary; observations this extreme only
    # arise under (near) deterministic take-up.
    return np.clip(p, _EPS, 1.0 - _EPS)


def _conditional_takeup_diff(
    ds: pl.DataFrame,
    treatment: ColumnName,
    instrument: ColumnName,
    weight: ColumnName,
) -> float:
    """``E[D | Z = 1] - E[D | Z = 0]`` (the first stage), computed in Polars."""
    d = pl.col(treatment)
    z = pl.col(instrument)
    w = pl.col(weight)
    at_one = (d * w).filter(z == 1).sum() / w.filter(z == 1).sum()
    at_zero = (d * w).filter(z == 0).sum() / w.filter(z == 0).sum()
    return ds.select((at_one - at_zero).alias("first_stage")).item()
