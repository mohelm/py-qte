"""Shared helpers for the instrument-validity tests."""

import numpy as np
import polars as pl

_MIN_ARM_SIZE = 2


def _compute_outcome_grid(out_vals: pl.Series, max_y_grid_size: int) -> pl.Series:
    """Return the sorted candidate interval endpoints of the outcome."""
    if out_vals.n_unique() <= max_y_grid_size:
        return out_vals.unique().sort()
    quantiles = np.linspace(0.0, 1.0, max_y_grid_size).tolist()
    return pl.Series(out_vals.quantile(quantiles, "linear")).unique().sort()
