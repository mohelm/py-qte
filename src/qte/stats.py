import numpy as np
import polars as pl
import statsmodels.formula.api as smf
from numpy.typing import NDArray
from statsmodels.discrete.discrete_model import BinaryResultsWrapper

from qte.custom_types import ColumnName, Series
from qte.quantile_regression import QuantileRegression, QuantileRegressionResult


def get_quantiles(
    qs: NDArray[np.float64] | float,
    data: Series | NDArray,
    w: NDArray | None = None,
) -> NDArray | float:
    if isinstance(data, pl.Series):
        data = data.to_numpy()
    if w is None:
        return np.quantile(data, qs)
    return np.quantile(data, qs, method="inverted_cdf", weights=w)


def estimate_propensity_score(
    ds: pl.DataFrame, treatment_c: str, x_formular: str
) -> BinaryResultsWrapper:
    ds_as_dict = {col.name: col.to_numpy() for col in ds.iter_columns()}
    return smf.logit(formula=f"{treatment_c}~{x_formular}", data=ds_as_dict).fit(disp=0)


def estimate_outcome_model(
    ds: pl.DataFrame, outcome_c: ColumnName, x_formular: str, qs: NDArray[np.float64]
) -> QuantileRegressionResult:
    return QuantileRegression(f"{outcome_c}~{x_formular}", data=ds).fit(qs)
