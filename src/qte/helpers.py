"""Small shared helpers."""

import numpy as np
import polars as pl
from numpy.typing import ArrayLike, NDArray

from qte.names import (
    EFFECT_ID,
    MEAN_CONTROL_ID,
    MEAN_TREATED_ID,
    QUANTILE_CONTROL_VAL_ID,
    QUANTILE_ID,
    QUANTILE_TREATED_VAL_ID,
)


def _as_iterable_array(values: ArrayLike) -> NDArray:
    # Basically, we ensure that the array we create from values is always iterable and has a
    # length. For instance values=0.5 is ArrayLike, but np.asarray(0.5) has no length and is not iterable.
    return np.atleast_1d(np.asarray(values, dtype=float))


def _get_quantile_differences(qs: ArrayLike, q_t: ArrayLike, q_c: ArrayLike) -> pl.DataFrame:
    """Assemble the ``q`` / ``q_t`` / ``q_c`` / ``effect`` frame.

    ``effect`` is the treated-minus-control quantile difference.
    """
    return pl.DataFrame(
        {
            QUANTILE_ID: qs,
            QUANTILE_TREATED_VAL_ID: q_t,
            QUANTILE_CONTROL_VAL_ID: q_c,
        }
    ).with_columns(
        (pl.col(QUANTILE_TREATED_VAL_ID) - pl.col(QUANTILE_CONTROL_VAL_ID)).alias(EFFECT_ID)
    )


def _get_mean_differences(m_t: ArrayLike, m_c: ArrayLike) -> pl.DataFrame:
    """Assemble the ``m_t`` / ``m_c`` / ``effect`` frame.

    ``effect`` is the treated-minus-control mean difference.
    """
    return pl.DataFrame(
        {
            MEAN_TREATED_ID: m_t,
            MEAN_CONTROL_ID: m_c,
        }
    ).with_columns((pl.col(MEAN_TREATED_ID) - pl.col(MEAN_CONTROL_ID)).alias(EFFECT_ID))
