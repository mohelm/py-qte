import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose

from qte.bootstrap import BootstrapConfig
from qte.cross_sectional import (
    estimate_aipw_effects,
    estimate_ipw_effects,
    estimate_unadjusted_effects,
    simulate_covariate_data,
    simulate_simple_data,
)
from qte.cross_sectional.custom_types import CausalTarget
from qte.cross_sectional.simulate import get_true_means, get_true_quantiles
from qte.names import EFFECT_ID

QS = (0.25, 0.5, 0.75)
CFG = BootstrapConfig(n_iter=8, seed=1)

N_USED_COVARIATES = 5
SIMPLE_COLUMNS = ["treat", "y"]
COVARIATE_COLUMNS = [f"x{i}" for i in range(N_USED_COVARIATES)] + ["treat", "y", "y_0", "y_1"]


def _add_correct_features(ds: pl.DataFrame) -> pl.DataFrame:
    return ds.with_columns(
        ind1=(pl.col("x1") > 0.1).cast(pl.Float64),
        ind2=((pl.col("x0") * pl.col("x2")) > 0).cast(pl.Float64),
        sq=(2.5 + 0.3 * pl.col("x1")).sqrt(),
    )


def test_simulate_simple_shape_and_columns():
    ds = simulate_simple_data(n=500, treatment_share=0.4, seed=0)

    assert isinstance(ds, pl.DataFrame)
    assert ds.columns == SIMPLE_COLUMNS
    assert ds.shape == (500, 2)
    assert set(ds["treat"].unique()) <= {0, 1}


def test_simulate_covariate_shape_and_columns():
    ds = simulate_covariate_data(n=500, seed=0)

    assert isinstance(ds, pl.DataFrame)
    assert ds.columns == COVARIATE_COLUMNS
    assert ds.shape == (500, len(COVARIATE_COLUMNS))


def test_simulate_noise_covariates_are_appended():
    ds = simulate_covariate_data(n=100, n_noise_covariates=3, seed=0)

    used = [f"x{i}" for i in range(N_USED_COVARIATES)]
    noise = [f"x{i}" for i in range(N_USED_COVARIATES, N_USED_COVARIATES + 3)]
    assert ds.columns == used + noise + ["treat", "y", "y_0", "y_1"]


def test_simulate_is_reproducible():
    assert simulate_simple_data(n=100, seed=42).equals(simulate_simple_data(n=100, seed=42))
    assert not simulate_simple_data(n=100, seed=42).equals(simulate_simple_data(n=100, seed=43))
    assert simulate_covariate_data(n=100, seed=42).equals(simulate_covariate_data(n=100, seed=42))


def test_estimators_recover_the_simple_truth():
    ds = simulate_simple_data(n=20_000, treatment_effect=1.5, seed=0)
    res = estimate_unadjusted_effects(ds, "y", "treat", qs=QS, bootstrap_config=CFG)

    assert_allclose(res.qtt[EFFECT_ID].to_numpy(), 1.5, atol=0.05)
    assert res.att[EFFECT_ID].item() == pytest.approx(1.5, abs=0.05)


def test_covariate_data_is_confounded_but_aipw_recovers_the_truth():
    ds = _add_correct_features(simulate_covariate_data(n=20_000, effect_scale=1.0, seed=0))
    truth = get_true_quantiles(QS, n_oracle=200_000, effect_scale=1.0, seed=0)[EFFECT_ID].to_numpy()

    naive = estimate_unadjusted_effects(ds, "y", "treat", qs=QS, bootstrap_config=CFG)
    aipw = estimate_aipw_effects(
        ds,
        "y",
        "treat",
        qs=QS,
        propensity_score_formula="x1 + x3 + I(x0 > 0)",
        outcome_regression_config="x3 + x4 + ind1 + ind2 + sq",
        bootstrap_config=CFG,
    )

    # Confounding biases the unadjusted estimator away from the true effect.
    assert not np.allclose(naive.qtt[EFFECT_ID].to_numpy(), truth, atol=0.1)
    assert_allclose(aipw.qtt[EFFECT_ID].to_numpy(), truth, atol=0.1)


def test_covariate_data_att_matches_true_ate():
    ds = simulate_covariate_data(n=20_000, effect_scale=1.0, seed=0)
    truth = get_true_means(n_oracle=200_000, effect_scale=1.0, seed=0)[EFFECT_ID].item()

    res = estimate_ipw_effects(
        ds,
        "y",
        "treat",
        qs=QS,
        propensity_score_formula="x1 + x3 + I(x0 > 0)",
        target=CausalTarget.QTT,
        bootstrap_config=CFG,
    )

    assert res.att[EFFECT_ID].item() == pytest.approx(truth, abs=0.1)
