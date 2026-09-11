from dataclasses import dataclass

import numpy as np
import polars as pl
import statsmodels.formula.api as smf
from numpy.typing import NDArray
from scipy.stats import norm
from statsmodels.discrete.discrete_model import BinaryResultsWrapper

from qte.custom_types import ColumnName, FormularRhs, Series
from qte.names import CI_LB_ID, CI_UB_ID, EFFECT_ID, SE_ID
from qte.quantile_regression import QuantileRegression, QuantileRegressionResult


def get_quantiles(
    qs: NDArray[np.float64] | float,
    data: Series | NDArray,
    w: NDArray | None = None,
) -> NDArray:
    if isinstance(data, pl.Series):
        data = data.to_numpy()
    if w is None:
        return np.quantile(data, qs)
    return np.quantile(data, qs, method="inverted_cdf", weights=w)


def estimate_propensity_score(
    ds: pl.DataFrame, treatment_c: str, x_formular: FormularRhs
) -> BinaryResultsWrapper:
    ds_as_dict = {col.name: col.to_numpy() for col in ds.iter_columns()}
    return smf.logit(formula=f"{treatment_c}~{x_formular}", data=ds_as_dict).fit(disp=0)


def estimate_outcome_model(
    ds: pl.DataFrame, outcome_c: ColumnName, x_formular: str, qs: NDArray[np.float64]
) -> QuantileRegressionResult:
    return QuantileRegression(f"{outcome_c}~{x_formular}", data=ds).fit(qs)


@dataclass
class Ecdf:
    values: NDArray
    probs: NDArray
    weights: NDArray | None = None

    @classmethod
    def make(cls: type[Ecdf], y: NDArray, w: NDArray) -> Ecdf:
        sorter = np.argsort(y)
        y_sorted = y[sorter]
        w_sorted = w[sorter]

        # Handle exact ties: accumulate weights for identical outcome values
        y_unique, indices = np.unique(y_sorted, return_inverse=True)
        w_unique = np.bincount(indices, weights=w_sorted)

        w_norm = w_unique / w_unique.sum()
        return Ecdf(y_unique, np.cumsum(w_norm), w_unique)

    def evaluate_inverse(self, qs: NDArray) -> NDArray:
        idx = np.searchsorted(self.probs, qs, side="left")
        idx = np.clip(idx, 0, len(self.values) - 1)
        return self.values[idx]

    def evaluate(self, grid: NDArray) -> NDArray:
        idx = np.searchsorted(self.values, grid, side="right")
        return np.concatenate(([0.0], self.probs))[idx]


def get_ci(alpha: float) -> list[pl.Expr]:
    alpha_half = (1 - alpha) / 2
    return [
        (pl.col(EFFECT_ID) + norm.ppf(alpha_half) * pl.col(SE_ID)).alias(CI_LB_ID),
        (pl.col(EFFECT_ID) + norm.ppf(1 - alpha_half) * pl.col(SE_ID)).alias(CI_UB_ID),
    ]
