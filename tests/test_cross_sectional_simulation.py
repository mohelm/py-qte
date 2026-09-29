import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose

from qte.bootstrap import BootstrapConfig
from qte.cross_sectional import (
    estimate_aipw_qte,
    estimate_ipw_qte,
    estimate_simple_qte,
    simulate_data,
)
from qte.custom_types import CausalTarget
from qte.names import EFFECT_ID

QS = (0.25, 0.5, 0.75)
CFG = BootstrapConfig(n_iter=8, seed=1)


def test_simulate_without_covariates_shape_and_columns():
    ds = simulate_data(n=500, treatment_share=0.4, seed=0)

    assert isinstance(ds, pl.DataFrame)
    assert ds.columns == ["treat", "y"]
    assert ds.shape == (500, 2)
    assert ds["treat"].sum() == 200
    assert set(ds["treat"].unique()) == {0, 1}


def test_simulate_with_covariates_shape_and_columns():
    ds = simulate_data(n=500, with_covariates=True, seed=0)

    assert isinstance(ds, pl.DataFrame)
    assert ds.columns == ["x1", "x2", "treat", "y"]
    assert ds.shape == (500, 4)
    assert set(ds["x2"].unique()) == {0.0, 1.0}
    assert set(ds["treat"].unique()) == {0, 1}


def test_simulate_is_reproducible():
    first = simulate_data(n=100, seed=42)
    second = simulate_data(n=100, seed=42)
    third = simulate_data(n=100, seed=43)
    cov_first = simulate_data(n=100, with_covariates=True, seed=42)
    cov_second = simulate_data(n=100, with_covariates=True, seed=42)

    assert first.equals(second)
    assert not first.equals(third)
    assert cov_first.equals(cov_second)


def test_estimators_recover_the_simple_truth():
    ds = simulate_data(n=20_000, treatment_effect=1.5, seed=0)
    res = estimate_simple_qte(ds, "y", "treat", qs=QS, bootstrap_config=CFG)

    assert_allclose(res.qtt[EFFECT_ID].to_numpy(), 1.5, atol=0.05)
    assert res.att[EFFECT_ID].item() == pytest.approx(1.5, abs=0.05)


def test_covariate_data_is_confounded_but_ipw_recovers_the_truth():
    ds = simulate_data(n=20_000, with_covariates=True, treatment_effect=1.0, seed=0)
    truth = 1.0

    naive = estimate_simple_qte(ds, "y", "treat", qs=QS, bootstrap_config=CFG)
    ipw = estimate_ipw_qte(ds, "y", "treat", qs=QS, ps_x_formular="x1 + x2", bootstrap_config=CFG)
    aipw = estimate_aipw_qte(
        ds,
        "y",
        "treat",
        qs=QS,
        ps_x_formular="x1 + x2",
        or_x_formular="x1 + x2",
        bootstrap_config=CFG,
    )

    # Confounding biases the unadjusted estimator away from the true effect.
    assert not np.allclose(naive.qtt[EFFECT_ID].to_numpy(), truth, atol=0.2)
    assert_allclose(ipw.qtt[EFFECT_ID].to_numpy(), truth, atol=0.15)
    assert_allclose(aipw.qtt[EFFECT_ID].to_numpy(), truth, atol=0.15)


def test_covariate_data_att_matches_qte():
    # The treatment effect is constant, so ATT == ATE == QTE at every quantile.
    ds = simulate_data(n=10_000, with_covariates=True, treatment_effect=2.0, seed=0)
    res = estimate_ipw_qte(
        ds,
        "y",
        "treat",
        qs=QS,
        ps_x_formular="x1 + x2",
        target=CausalTarget.QTT,
        bootstrap_config=CFG,
    )

    assert_allclose(res.qtt[EFFECT_ID].to_numpy(), 2.0, atol=0.2)
    assert res.att[EFFECT_ID].item() == pytest.approx(2.0, abs=0.2)


@pytest.mark.parametrize("with_covariates", [False, True])
def test_validation(with_covariates):
    with pytest.raises(ValueError, match="n must be positive"):
        simulate_data(n=0, with_covariates=with_covariates)
