import numpy as np
import polars as pl
from numpy.typing import NDArray

from qte.custom_types import ColumnName
from qte.quantile_regression import QuantileRegressionResult


def predict_outcome_model(
    or_: QuantileRegressionResult, ds: pl.DataFrame | None = None, *, flatten: bool = True
) -> NDArray[np.float64]:

    preds = np.sort(or_.predict(ds), axis=1)
    return preds.flatten() if flatten else preds


def make_weights(
    weights: ColumnName | None, ds: pl.DataFrame, rep: int | None = None
) -> NDArray[np.float64] | None:
    if weights is None:
        return None
    sample_weights = ds[weights].to_numpy()
    return sample_weights if rep is None else np.tile(sample_weights, rep)  # TODO: really tile?
