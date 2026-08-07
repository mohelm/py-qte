import numpy as np
import polars as pl
from numpy.typing import NDArray

from qte.custom_types import Series


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
