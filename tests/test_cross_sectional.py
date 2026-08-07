import numpy as np
import polars as pl
import pytest
from polars.testing import assert_series_equal

from qte.constants import QUARTILES
from qte.cross_sectional import (
    estimate_aipw_qte,
    estimate_ipw_qte,
    estimate_or_qte,
    estimate_simple_qte,
)
from qte.cross_sectional.results import QteResult
from qte.custom_types import CausalTarget


def make_data(n_treated: int = 500, n_control: int | None = None) -> pl.DataFrame:
    if n_control is None:
        n_control = n_treated
    rng = np.random.default_rng()
    treated_outcome = rng.normal(0, 1, n_treated)
    control_outcome = rng.normal(1, 1, n_control)
    return pl.from_dict({
        "outcome": np.r_[treated_outcome, control_outcome],
        "treated": np.r_[np.ones((n_treated,)), np.zeros((n_control,))],
    })


def test_estimate_qte():
    ds = make_data(5000)
    assert isinstance(ds, pl.DataFrame)

    res = estimate_simple_qte(ds, "outcome", "treated", qs=(0.05, 0.5, 0.95))
    assert isinstance(res, QteResult)


def test_estimate_ipw_qte():
    ds = make_data(5000)
    res = estimate_ipw_qte(
        ds, "outcome", "treated", ps_x_formular="1", qs=(0.05, 0.5, 0.95)
    )
    assert isinstance(res, QteResult)


IPW_LALONDE_TEST_CASE = [
    (
        {"target": CausalTarget.QTE, "qs": [0.25, 0.5, 0.75]},
        {"q": [0.25, 0.5, 0.75], "effect": [-8754.131, -13667.879, -16545.341]},
    ),
    (
        {"target": CausalTarget.QTT, "qs": [0.25, 0.5, 0.75]},
        {"q": [0.25, 0.5, 0.75], "effect": [-1879.133, -4634.049, -6416.931]},
    ),
]


@pytest.mark.parametrize("estimate_params,expected_results", IPW_LALONDE_TEST_CASE)
def test_estimate_ipw_qte_with_lalonde(lalonde_psid, estimate_params, expected_results):
    xf = "age + I(age**2) + education + black + hispanic + married + nodegree"
    res = estimate_ipw_qte(
        lalonde_psid, "re78", "treat", ps_x_formular=xf, **estimate_params
    )
    assert_series_equal(
        pl.Series("q", expected_results["q"]), res.get_as_dataframe()["q"]
    )
    assert_series_equal(
        pl.Series("effect", expected_results["effect"]),
        res.get_as_dataframe()["effect"],
    )


OR_TEST_CASES = [
    (
        {"target": CausalTarget.QTE, "qs": QUARTILES, "n_bootstrap_iter": 10},
        {"q": QUARTILES, "effect": [-7389.094, -12340.600, -15976.407]},
    ),
    (
        {"target": CausalTarget.QTT, "qs": QUARTILES, "n_bootstrap_iter": 10},
        {"q": QUARTILES, "effect": [-3196.046, -5933.218, -7265.786]},
    ),
]


@pytest.mark.parametrize("estimate_params,expected_results", OR_TEST_CASES)
def test_estimate_or_qte_with_lalonde(lalonde_psid, estimate_params, expected_results):
    xf = "age + I(age**2) + education + black + hispanic + married + nodegree"
    res = estimate_or_qte(
        lalonde_psid, "re78", "treat", or_x_formular=xf, **estimate_params
    )
    assert_series_equal(
        pl.Series("q", expected_results["q"]), res.get_as_dataframe()["q"]
    )
    assert_series_equal(
        pl.Series("effect", expected_results["effect"]),
        res.get_as_dataframe()["effect"],
        rel_tol=0.01,
    )


AIPW_TEST_CASES = [
    (
        {"target": CausalTarget.QTE, "qs": QUARTILES, "n_bootstrap_iter": 10},
        {"q": QUARTILES, "effect": [-7646.724, -12684.516, -16522.675]},
    ),
    (
        {"target": CausalTarget.QTT, "qs": QUARTILES, "n_bootstrap_iter": 10},
        {"q": QUARTILES, "effect": [-1866.290, -4602.606, -6367.724]},
    ),
]


@pytest.mark.parametrize("estimate_params,expected_results", AIPW_TEST_CASES)
def test_estimate_aipw_qte_with_lalonde(
    lalonde_psid, estimate_params, expected_results
):
    xf = "age + I(age**2) + education + black + hispanic + married + nodegree"
    res = estimate_aipw_qte(
        lalonde_psid,
        "re78",
        "treat",
        or_x_formular=xf,
        ps_x_formular=xf,
        **estimate_params,
    )
    assert_series_equal(
        pl.Series("q", expected_results["q"]), res.get_as_dataframe()["q"]
    )
    assert_series_equal(
        pl.Series("effect", expected_results["effect"]),
        res.get_as_dataframe()["effect"],
        rel_tol=0.05,
    )
