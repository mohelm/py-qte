from dataclasses import replace

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal, assert_series_equal

from qte.bootstrap import BootstrapConfig
from qte.constants import QUARTILES
from qte.cross_sectional import (
    estimate_aipw_effects,
    estimate_ipw_effects,
    estimate_outcome_regression_effects,
    estimate_unadjusted_effects,
)
from qte.cross_sectional.custom_types import CausalTarget
from qte.cross_sectional.results import QteResult
from qte.names import EFFECT_ID, QUANTILE_ID


def make_data(n_treated: int = 500, n_control: int | None = None) -> pl.DataFrame:
    if n_control is None:
        n_control = n_treated
    rng = np.random.default_rng()
    treated_outcome = rng.normal(0, 1, n_treated)
    control_outcome = rng.normal(1, 1, n_control)
    return pl.from_dict(
        {
            "outcome": np.r_[treated_outcome, control_outcome],
            "treated": np.r_[np.ones((n_treated,)), np.zeros((n_control,))],
        }
    )


def test_estimate_unadjusted_effects():
    ds = make_data(5000)
    assert isinstance(ds, pl.DataFrame)

    res = estimate_unadjusted_effects(ds, "outcome", "treated", qs=(0.05, 0.5, 0.95))
    assert isinstance(res, QteResult)


def test_estimate_unadjusted_effects_parallel_matches_sequential():
    ds = make_data(500)
    cfg = BootstrapConfig(n_iter=8, seed=42)
    serial = estimate_unadjusted_effects(
        ds, "outcome", "treated", qs=(0.25, 0.5, 0.75), bootstrap_config=cfg
    )
    parallel = estimate_unadjusted_effects(
        ds,
        "outcome",
        "treated",
        qs=(0.25, 0.5, 0.75),
        bootstrap_config=replace(cfg, n_workers=2),
    )
    assert_frame_equal(serial.qtt, parallel.qtt)
    assert_frame_equal(serial.att, parallel.att)


def test_estimate_ipw_effects():
    ds = make_data(5000)
    res = estimate_ipw_effects(
        ds, "outcome", "treated", propensity_score_formula="1", qs=(0.05, 0.5, 0.95)
    )
    assert isinstance(res, QteResult)


IPW_LALONDE_TEST_CASE = [
    (
        {"target": CausalTarget.QTE, "qs": [0.25, 0.5, 0.75]},
        {
            "q": [0.25, 0.5, 0.75],
            "effect_q": [-8754.131, -13667.879, -16545.341],
            "effect_m": [-13194.78],
        },
    ),
    (
        {"target": CausalTarget.QTT, "qs": [0.25, 0.5, 0.75]},
        {
            "q": [0.25, 0.5, 0.75],
            "effect_q": [-1879.133, -4634.049, -6416.931],
            "effect_m": [-4685.583],
        },
    ),
]


@pytest.mark.parametrize("estimate_params,expected_results", IPW_LALONDE_TEST_CASE)
def test_estimate_ipw_effects_with_lalonde(lalonde_psid, estimate_params, expected_results):
    xf = "age + I(age**2) + education + black + hispanic + married + nodegree"
    res = estimate_ipw_effects(
        lalonde_psid, "re78", "treat", propensity_score_formula=xf, **estimate_params
    )
    assert_series_equal(pl.Series("q", expected_results["q"]), res.qtt[QUANTILE_ID])
    assert_series_equal(pl.Series("effect", expected_results["effect_q"]), res.qtt[EFFECT_ID])
    assert_series_equal(pl.Series("effect", expected_results["effect_m"]), res.att[EFFECT_ID])


OR_TEST_CASES = [
    (
        {"target": CausalTarget.QTE, "qs": QUARTILES, "bootstrap_config": 10},
        {"q": QUARTILES, "effect_q": [-7389.094, -12340.600, -15976.407], "effect_m": [-11673.0]},
    ),
    (
        {"target": CausalTarget.QTT, "qs": QUARTILES, "bootstrap_config": 10},
        {"q": QUARTILES, "effect_q": [-3271.908, -6025.094, -7481.486], "effect_m": [-5179.468]},
    ),
]


@pytest.mark.parametrize("estimate_params,expected_results", OR_TEST_CASES)
def test_estimate_outcome_regression_effects_with_lalonde(
    lalonde_psid, estimate_params, expected_results
):
    xf = "age + I(age**2) + education + black + hispanic + married + nodegree"
    res = estimate_outcome_regression_effects(
        lalonde_psid, "re78", "treat", outcome_regression_formula=xf, **estimate_params
    )
    assert_series_equal(pl.Series("q", expected_results["q"]), res.get_as_dataframe()["q"])
    assert_series_equal(
        pl.Series("effect", expected_results["effect_q"]),
        res.qtt["effect"],
        rel_tol=0.01,
    )
    assert_series_equal(
        pl.Series("effect", expected_results["effect_m"]),
        res.att["effect"],
        rel_tol=0.01,
    )


AIPW_TEST_CASES = [
    (
        {"target": CausalTarget.QTE, "qs": QUARTILES, "bootstrap_config": 10},
        {"q": QUARTILES, "effect_q": [-7646.724, -12684.516, -16522.675], "effect_m": [-12535.523]},
    ),
    (
        {"target": CausalTarget.QTT, "qs": [0.25, 0.5, 0.75, 0.9], "bootstrap_config": 10},
        {
            "q": [0.25, 0.5, 0.75, 0.9],
            "effect_q": [
                -1866.290,
                -4602.606,
                -6202.56798,
                -10517.062,
            ],  # Q75: should be -6202.56789
            "effect_m": [-4543.927],
        },
    ),
]


@pytest.mark.parametrize("estimate_params,expected_results", AIPW_TEST_CASES)
def test_estimate_aipw_effects_with_lalonde(lalonde_psid, estimate_params, expected_results):
    xf = "age + I(age**2) + education + black + hispanic + married + nodegree"
    res = estimate_aipw_effects(
        lalonde_psid,
        "re78",
        "treat",
        outcome_regression_formula=xf,
        propensity_score_formula=xf,
        **estimate_params,
    )
    assert_series_equal(pl.Series("q", expected_results["q"]), res.get_as_dataframe()["q"])
    assert_series_equal(
        pl.Series("effect", expected_results["effect_q"]),
        res.qtt["effect"],
        rel_tol=0.01,
    )
    assert_series_equal(
        pl.Series("effect", expected_results["effect_m"]),
        res.att["effect"],
        rel_tol=0.01,
    )
