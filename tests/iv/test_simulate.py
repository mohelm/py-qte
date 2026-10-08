"""Tests for the local-QTE data simulation and its known-truth effects."""

import numpy as np
import polars as pl
import pytest

from qte.iv.simulate import get_true_local_effects, simulate_data
from qte.names import EFFECT_ID

# E[min(1, u / takeup_scale)] for u ~ U(0, 1) with the default takeup_scale of 0.75.
COMPLIER_SHARE = 0.625


def test_simulate_data_shape_and_reproducibility():
    ds = simulate_data(n=500, seed=0)

    assert ds.columns == ["x", "instrument", "treatment", "y", "y_0", "y_1", "complier"]
    assert set(ds["instrument"].unique()) <= {0, 1}
    assert set(ds["treatment"].unique()) <= {0, 1}
    # One-sided non-compliance: no always-takers, so D = 0 whenever Z = 0.
    assert ds.filter(pl.col("instrument") == 0)["treatment"].eq(0).all()
    assert simulate_data(n=100, seed=42).equals(simulate_data(n=100, seed=42))
    assert not simulate_data(n=100, seed=42).equals(simulate_data(n=100, seed=43))


def test_simulate_data_complier_share_and_first_stage():
    ds = simulate_data(n=200_000, seed=0)

    assert ds["complier"].mean() == pytest.approx(COMPLIER_SHARE, abs=0.01)
    first_stage = (
        ds.filter(pl.col("instrument") == 1)["treatment"].mean()
        - ds.filter(pl.col("instrument") == 0)["treatment"].mean()
    )
    assert first_stage == pytest.approx(COMPLIER_SHARE, abs=0.01)


def test_true_local_effects_returns_quantiles_and_late():
    qs = np.array([0.25, 0.5, 0.75])
    qtt, late = get_true_local_effects(qs, n_oracle=200_000, seed=1)
    assert qtt.height == qs.size
    assert qtt[EFFECT_ID].is_finite().all()
    assert np.isfinite(late)
    assert late > 0
