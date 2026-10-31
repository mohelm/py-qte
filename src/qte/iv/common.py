"""Internals shared by the conditional and unconditional IV estimators."""

from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import NDArray

from qte.custom_types import ColumnName, FormulaRhs
from qte.results import InstrumentProbabilityResult
from qte.stats import estimate_propensity_score

_EPS = 1e-8
_DEFAULT_TRIM = 0.001
_PROPENSITY_ID = "_propensity"


@dataclass(frozen=True)
class _InstrumentPropensity:
    """Estimated instrument probabilities and, optionally, the fitted logit."""

    values: NDArray[np.float64]
    model_result: InstrumentProbabilityResult | None = None


def _estimate_instrument_propensity(
    ds: pl.DataFrame,
    instrument: ColumnName,
    instrument_probability_formula: FormulaRhs | None,
    weights: ColumnName,
    *,
    with_model: bool = False,
) -> _InstrumentPropensity:
    """Estimate ``p(X) = P(Z = 1 | X)``, constant when no formula is given."""
    model_result = None
    if instrument_probability_formula is None and not with_model:
        # The constant probability is just the weighted take-up rate. Skip the
        # GLM here so bootstrap draws stay fast.
        p = np.full(
            len(ds),
            fill_value=np.average(ds[instrument].to_numpy(), weights=ds[weights].to_numpy()),
        )
    else:
        # An omitted formula is still an intercept-only logit, which has the same
        # fitted probability as the weighted mean above but gives us a model
        # result to present.
        formula = instrument_probability_formula or "1"
        fit = estimate_propensity_score(ds, instrument, formula, weights)
        p = fit.predict()
        if with_model:
            model_result = InstrumentProbabilityResult.from_statsmodels(
                fit, instrument=instrument, formula=instrument_probability_formula
            )
    # Avoid division blow-ups at the boundary; observations this extreme only
    # arise under (near) deterministic take-up.
    return _InstrumentPropensity(values=np.clip(p, _EPS, 1.0 - _EPS), model_result=model_result)


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
