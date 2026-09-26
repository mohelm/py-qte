import numpy as np
from numpy.typing import NDArray


def _make_q(n: int) -> NDArray:
    return np.linspace(0, 1, n + 1)[1:-1]


QUARTILES = _make_q(4)
QUANTILES = _make_q(5)
DECILES = _make_q(10)
PERCENTILES = _make_q(100)
MEDIAN = _make_q(2)
