from collections.abc import Iterable, Iterator

import numpy as np

from qte.helpers import (
    _as_iterable_array,
    _get_mean_differences,
    _get_quantile_differences,
)
from qte.names import (
    EFFECT_ID,
    MEAN_CONTROL_ID,
    MEAN_TREATED_ID,
    QUANTILE_CONTROL_VAL_ID,
    QUANTILE_ID,
    QUANTILE_TREATED_VAL_ID,
)


def test_scalar_becomes_reiterable_length_one_array():
    result = _as_iterable_array(0.5)

    assert len(result) == 1
    assert isinstance(result, Iterable)
    assert not isinstance(result, Iterator)


def test_get_quantile_differences_subtracts_control_from_treated():
    frame = _get_quantile_differences(
        np.array([0.25, 0.5, 0.75]),
        np.array([2.0, 3.0, 4.0]),
        np.array([1.0, 1.5, 1.0]),
    )

    assert frame.columns == [
        QUANTILE_ID,
        QUANTILE_TREATED_VAL_ID,
        QUANTILE_CONTROL_VAL_ID,
        EFFECT_ID,
    ]
    assert frame[EFFECT_ID].to_list() == [1.0, 1.5, 3.0]


def test_get_mean_differences_subtracts_control_from_treated():
    frame = _get_mean_differences(3.0, 1.0)

    assert frame.columns == [MEAN_TREATED_ID, MEAN_CONTROL_ID, EFFECT_ID]
    assert frame[EFFECT_ID].to_list() == [2.0]
