"""Tests for the RD simulation helpers."""

import numpy as np
import polars as pl
import pytest

from qte.rdd import get_true_first_stage, get_true_local_effects, simulate_rdd_data


def test_simulate_rdd_data_columns_and_reproducibility():
    ds = simulate_rdd_data(n=500, seed=0)

    assert ds.columns == [
        "running",
        "treatment",
        "y",
        "y_0",
        "y_1",
        "d_0",
        "d_1",
        "complier",
    ]
    assert ds.equals(simulate_rdd_data(n=500, seed=0))
    assert not ds.equals(simulate_rdd_data(n=500, seed=1))
    assert ds["treatment"].is_in([0, 1]).all()
    assert ds["d_0"].le(ds["d_1"]).all()
    assert ds["complier"].eq((ds["d_1"] == 1) & (ds["d_0"] == 0)).all()


def test_simulated_first_stage_matches_population_value():
    ds = simulate_rdd_data(n=200_000, alpha=2.5, seed=0)
    first_stage = (
        ds.filter(pl.col("running") >= 0)["treatment"].mean()
        - ds.filter(pl.col("running") < 0)["treatment"].mean()
    )
    assert first_stage == pytest.approx(get_true_first_stage(alpha=2.5), abs=0.02)
    assert ds["complier"].mean() == pytest.approx(get_true_first_stage(alpha=2.5), abs=0.01)


def test_true_local_effects_are_finite():
    qs = np.array([0.25, 0.5, 0.75])
    qtt, late = get_true_local_effects(qs, n_oracle=200_000, seed=1)

    assert qtt.height == qs.size
    assert qtt["effect"].is_finite().all()
    assert np.isfinite(late)
    # The treatment effect in the simulation is -eps_1, so its mean is negative.
    assert late < 0
