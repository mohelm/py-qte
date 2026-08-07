import numpy as np
import polars as pl
from numpy.typing import NDArray

from qte.quantile_regression import QuantileRegressionResult


def predict_outcome_model(
    or_: QuantileRegressionResult, ds: pl.DataFrame | None = None, flatten: bool = True
) -> NDArray[np.float64]:
    preds = np.sort(or_.predict(ds), axis=1)
    return preds.flatten() if flatten else preds


def make_weights(
    weights_c: str | None, ds: pl.DataFrame, rep: int | None = None
) -> NDArray[np.float64] | None:
    if weights_c is None:
        return None
    weights = ds[weights_c].to_numpy()
    return weights if rep is None else np.tile(weights, rep)
