"""Tests for the RD distribution-regression estimator."""

import numpy as np
import polars as pl
import pytest
from great_tables import GT
from rich.console import Group

from qte.names import CI_LB_ID, CI_UB_ID, EFFECT_ID, QUANTILE_ID, SE_ID
from qte.rdd import (
    Kernel,
    LocalPolynomialConfig,
    RunningVariableConfig,
    estimate_rdd_distribution_effects,
    get_true_first_stage,
    get_true_local_effects,
    simulate_rdd_data,
)
from qte.rdd.distribution import _compute_local_distribution_effects


@pytest.fixture(scope="module")
def fuzzy_data() -> pl.DataFrame:
    return simulate_rdd_data(n=100_000, alpha=2.5, seed=1)


@pytest.fixture(scope="module")
def fuzzy_result(fuzzy_data):
    return estimate_rdd_distribution_effects(
        fuzzy_data,
        "y",
        "treatment",
        RunningVariableConfig("running", 0.0),
        qs=[0.25, 0.5, 0.75],
        local_polynomial_config=LocalPolynomialConfig(bandwidth=0.5),
        outcome_grid=100,
        bootstrap_config=10,
    )


def test_fuzzy_result_matches_simulated_truth(fuzzy_data, fuzzy_result):
    qs = [0.25, 0.5, 0.75]
    truth, true_late = get_true_local_effects(qs, alpha=2.5, n_oracle=500_000, seed=1)

    assert fuzzy_result.qtt[QUANTILE_ID].to_list() == qs
    assert fuzzy_result.qtt[EFFECT_ID].is_finite().all()
    assert fuzzy_result.qtt[SE_ID].is_finite().all()
    assert np.allclose(
        fuzzy_result.qtt[EFFECT_ID].to_numpy(),
        truth[EFFECT_ID].to_numpy(),
        atol=0.12,
    )
    assert fuzzy_result.att[EFFECT_ID][0] == pytest.approx(true_late, abs=0.10)
    assert fuzzy_result.first_stage == pytest.approx(get_true_first_stage(alpha=2.5), abs=0.05)
    assert fuzzy_result.n_obs == fuzzy_data.height


def test_result_presentation_methods(fuzzy_data, fuzzy_result):
    frame = fuzzy_result.get_as_dataframe()
    assert {CI_LB_ID, CI_UB_ID, EFFECT_ID, SE_ID}.issubset(frame.columns)
    assert isinstance(fuzzy_result.tabulate(), GT)
    assert isinstance(fuzzy_result.summarize(), Group)
    assert fuzzy_result.plot() is not None
    assert fuzzy_result.plot_design(fuzzy_data) is not None
    assert fuzzy_result.plot_treatment(fuzzy_data) is not None


def test_sharp_design_recovers_unconditional_quantile_jump(fuzzy_data):
    sharp = fuzzy_data.with_columns(
        (pl.col("running") >= 0.0).cast(pl.Int64).alias("treatment"),
        pl.when(pl.col("running") >= 0.0).then(pl.col("y_1")).otherwise(pl.col("y_0")).alias("y"),
    )
    qs = np.array([0.25, 0.5, 0.75])
    truth = np.quantile(sharp["y_1"].to_numpy(), qs) - np.quantile(sharp["y_0"].to_numpy(), qs)

    result = estimate_rdd_distribution_effects(
        sharp,
        "y",
        "treatment",
        RunningVariableConfig("running", 0.0),
        qs=qs,
        local_polynomial_config=LocalPolynomialConfig(bandwidth=0.8),
        outcome_grid=100,
        bootstrap_config=10,
    )

    assert result.first_stage == pytest.approx(1.0, abs=1e-8)
    assert np.allclose(result.qtt[EFFECT_ID].to_numpy(), truth, atol=0.10)


def test_rejects_non_binary_treatment(fuzzy_data):
    bad = fuzzy_data.with_columns((pl.col("treatment") + 1).alias("treatment"))
    with pytest.raises(Exception, match="must be 0/1"):
        estimate_rdd_distribution_effects(
            bad,
            "y",
            "treatment",
            RunningVariableConfig("running", 0.0),
            local_polynomial_config=LocalPolynomialConfig(bandwidth=0.5),
            bootstrap_config=1,
        )


def test_rejects_invalid_bandwidth():
    with pytest.raises(ValueError, match="bandwidth"):
        LocalPolynomialConfig(bandwidth=0.0)


def test_rejects_missing_first_stage(fuzzy_data):
    no_jump = fuzzy_data.with_columns(
        pl.lit(0).alias("treatment"),
        pl.lit(1.0).alias("_w"),
    )
    with pytest.raises(ValueError, match="does not jump"):
        _compute_local_distribution_effects(
            no_jump,
            "y",
            "treatment",
            RunningVariableConfig("running", 0.0),
            np.array([0.5]),
            local_polynomial_config=LocalPolynomialConfig(bandwidth=0.5),
            outcome_grid=30,
            weights="_w",
        )


def test_accepts_explicit_outcome_grid(fuzzy_data):
    grid = np.linspace(-3.0, 3.0, 40)
    result = estimate_rdd_distribution_effects(
        fuzzy_data.head(5_000),
        "y",
        "treatment",
        RunningVariableConfig("running", 0.0),
        qs=[0.5],
        local_polynomial_config=LocalPolynomialConfig(bandwidth=0.8),
        outcome_grid=grid,
        bootstrap_config=2,
    )
    assert result.outcome_grid_size == grid.size
    assert result.qtt[EFFECT_ID].is_finite().all()


def test_accepts_string_kernel(fuzzy_data):
    result = estimate_rdd_distribution_effects(
        fuzzy_data.head(2_000),
        "y",
        "treatment",
        RunningVariableConfig("running", 0.0),
        qs=[0.5],
        local_polynomial_config=LocalPolynomialConfig(bandwidth=0.75, kernel="uniform"),
        outcome_grid=30,
        bootstrap_config=2,
    )
    assert result.local_polynomial_config.kernel is Kernel.UNIFORM
    assert result.qtt[EFFECT_ID].is_finite().all()
