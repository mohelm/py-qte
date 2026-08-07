from functools import partial

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar

# Note maybe consider numpy.quantile for all this...


def check_function(data: NDArray, q: float) -> NDArray:
    # This gives how much each observations contributes to the loss function.
    return data * (q - (data <= 0))


def weighted_check_function(val: float, data: NDArray, q: float, w: NDArray) -> NDArray:
    # This computes the value of the loss function for quantile q at value val.
    return (w * check_function(data - val, q)).mean()


def get_quantile(
    q: float, data: NDArray, w: NDArray | None = None, normalize: bool = True
) -> float:
    if w is None:
        w = np.ones((data.shape[0], 1))
    if normalize:
        w = w / w.sum()

    return minimize_scalar(
        partial(weighted_check_function, data=data, q=q, w=w),
        bounds=(data.min(), data.max()),
    ).x


def get_quantiles(
    qs: NDArray[np.float32],
    data: NDArray,
    w: NDArray | None = None,
    normalize: bool = True,
) -> NDArray:
    get_quantile_specialized = partial(
        get_quantile, data=data, w=w, normalize=normalize
    )
    return np.array([get_quantile_specialized(q) for q in qs])
