import numpy as np

from qte.stats import get_quantile, get_quantiles


def test_get_quantile():
    data = np.array([-1, 0, 1])
    res = get_quantile(0.5, data)
    assert np.isclose(res, 0)


def test_get_quantiles():
    data = np.arange(-50, +51)
    expected_result = np.array([-25, 0, 25])
    quantiles = np.array([0.25, 0.5, 0.75])
    res = get_quantiles(quantiles, data)
    assert np.isclose(res, expected_result).all()
