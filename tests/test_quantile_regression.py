import numpy as np
import polars as pl
import statsmodels.formula.api as smf
from pytest import fixture

from qte.constants import MEDIAN
from qte.datasets import load_engel_with_with_weights
from qte.quantile_regression import QuantileRegression, QuantileRegressionResult


def test_quantile_regression_results(lalonde_psid):
    xf = "re78 ~ treat+age + I(age**2) + education + black + hispanic + married + nodegree"
    fit = QuantileRegression(xf, lalonde_psid).fit(qs=MEDIAN)
    assert isinstance(fit, QuantileRegressionResult)
    # Size of coefficients
    assert fit.coefficients.shape[1] == MEDIAN.shape[0]
    # Against statsmodels


def test_quantile_regression_prediction(lalonde_psid):
    xf = "re78 ~ treat+age + I(age**2) + education + black + hispanic + married + nodegree"
    fit = QuantileRegression(xf, lalonde_psid).fit(qs=MEDIAN)
    preds = fit.predict()
    assert preds.shape[0] == lalonde_psid.shape[0]


def test_quantile_regression_prediction_with_new_data(lalonde_psid):
    xf = "re78 ~ treat+age + I(age**2) + education + black + hispanic + married + nodegree"
    fit = QuantileRegression(xf, lalonde_psid).fit(qs=MEDIAN)
    preds = fit.predict(lalonde_psid.head(5))
    assert preds.shape[0] == 5


@fixture()
def data_with_int_weights():
    rng = np.random.default_rng(0)
    n = 60
    return pl.DataFrame(
        {
            "x": rng.normal(size=n),
            "w": rng.integers(1, 4, size=n).astype(float),
        }
    ).with_columns(y=1.0 + 2.0 * pl.col("x") + rng.normal(size=n))


def test_weighted_quantile_regression_matches_replication_on_exploded_data(data_with_int_weights):
    qs = np.array([0.25, 0.5, 0.75])
    formular = "y~x"

    weighted = QuantileRegression(formular, ds=data_with_int_weights).fit(qs, weights_c="w")
    # Integer weights are equivalent to replicating every row w_i times.
    replicated = data_with_int_weights.select(pl.all().repeat_by("w").explode()).drop("w")
    unweighted = QuantileRegression(formular, ds=replicated).fit(qs, weights_c=None)

    assert np.allclose(weighted.coefficients, unweighted.coefficients, atol=1e-6)

    # Compare to statsmmodels
    mod = smf.quantreg(formular, replicated)

    sm_coeffs = np.c_[*[mod.fit(q=q).params.values for q in qs]]
    assert np.allclose(weighted.coefficients, sm_coeffs, atol=1e-4)


def test_weighted_quantile_regression_matches_statsmodels_on_scaled_data(data_with_int_weights):
    qs = np.array([0.25, 0.5, 0.75])

    weighted = QuantileRegression("y ~ x", ds=data_with_int_weights).fit(qs, weights_c="w")

    scaled_data = data_with_int_weights.with_columns(
        [(pl.col(c) * pl.col("w")).alias(f"{c}_w") for c in ["y", "x"]]
    )
    mod = smf.quantreg("y_w ~ -1 + w + x_w", scaled_data)
    sm_coeffs = np.c_[*[mod.fit(q=q).params.values for q in qs]]
    assert np.allclose(weighted.coefficients, sm_coeffs, atol=1e-4)


def test_quantile_regression_matches_r_quantregpackage_results():

    qs = [0.1, 0.5, 0.9]

    unweighted_expected_coeffs = np.array(
        [
            [0.6983779, 0.4183258, 0.4777900],
            [0.8041082, 0.8765921, 0.8908642],
        ]
    )
    weighted_expected_coeffs = np.array(
        [
            [0.8360724, 1.9414444, 3.9609537],
            [0.7844268, 0.6545163, 0.4171704],
        ]
    )

    ds = load_engel_with_with_weights()

    weighted = QuantileRegression("log_foodexp ~ log_income", ds=ds).fit(qs, weights_c="w")
    assert np.allclose(weighted.coefficients, weighted_expected_coeffs, atol=1e-6)

    weighted = QuantileRegression("log_foodexp ~ log_income", ds=ds).fit(qs, weights_c=None)
    assert np.allclose(weighted.coefficients, unweighted_expected_coeffs, atol=1e-6)
