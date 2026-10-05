import numpy as np
import polars as pl
import statsmodels.formula.api as smf
from pytest import fixture, mark

import qte.quantile_regression as qr
from qte.constants import MEDIAN
from qte.datasets import load_engel_with_weights
from qte.quantile_regression import (
    QuantileRegression,
    QuantileRegressionAlgorithms,
    QuantileRegressionResult,
)

QS = [0.1, 0.5, 0.9]
UNWEIGHTED_EXPECTED_COEFFS = np.array(
    [
        [0.6983779, 0.4183258, 0.4777900],
        [0.8041082, 0.8765921, 0.8908642],
    ]
)
WEIGHTED_EXPECTED_COEFFS = np.array(
    [
        [0.8360724, 1.9414444, 3.9609537],
        [0.7844268, 0.6545163, 0.4171704],
    ]
)


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
    formula = "y~x"

    weighted = QuantileRegression(formula, ds=data_with_int_weights).fit(qs, weights="w")
    # Integer weights are equivalent to replicating every row w_i times.
    replicated = data_with_int_weights.select(pl.all().repeat_by("w").explode()).drop("w")
    unweighted = QuantileRegression(formula, ds=replicated).fit(qs, weights=None)

    # The statsmodels fallback is less precise than the Fortran solver.
    atol = 1e-6 if qr.rq_fortran is not None else 5e-5
    assert np.allclose(weighted.coefficients, unweighted.coefficients, atol=atol)

    # Compare to statsmmodels
    mod = smf.quantreg(formula, replicated)

    sm_coeffs = np.c_[*[mod.fit(q=q).params.values for q in qs]]
    assert np.allclose(weighted.coefficients, sm_coeffs, atol=1e-4)


def test_weighted_quantile_regression_matches_statsmodels_on_scaled_data(data_with_int_weights):
    qs = np.array([0.25, 0.5, 0.75])

    weighted = QuantileRegression("y ~ x", ds=data_with_int_weights).fit(qs, weights="w")

    scaled_data = data_with_int_weights.with_columns(
        [(pl.col(c) * pl.col("w")).alias(f"{c}_w") for c in ["y", "x"]]
    )
    mod = smf.quantreg("y_w ~ -1 + w + x_w", scaled_data)
    sm_coeffs = np.c_[*[mod.fit(q=q).params.values for q in qs]]
    assert np.allclose(weighted.coefficients, sm_coeffs, atol=1e-4)


@mark.skipif(qr.rq_fortran is None, reason="requires the compiled Fortran extension")
def test_quantile_regression_matches_r_quantregpackage_results():
    ds = load_engel_with_weights()

    weighted = QuantileRegression("log_foodexp ~ log_income", ds=ds).fit(QS, weights="w")
    assert np.allclose(weighted.coefficients, WEIGHTED_EXPECTED_COEFFS, atol=1e-6)

    unweighted = QuantileRegression("log_foodexp ~ log_income", ds=ds).fit(QS, weights=None)
    assert np.allclose(unweighted.coefficients, UNWEIGHTED_EXPECTED_COEFFS, atol=1e-6)


def test_quantile_regression_falls_back_to_statsmodels(monkeypatch):
    ds = load_engel_with_weights()

    monkeypatch.setattr(qr, "rq_fortran", None)

    weighted = QuantileRegression("log_foodexp ~ log_income", ds=ds).fit(QS, weights="w")
    assert np.allclose(weighted.coefficients, WEIGHTED_EXPECTED_COEFFS, atol=5e-5)

    unweighted = QuantileRegression("log_foodexp ~ log_income", ds=ds).fit(QS, weights=None)
    assert np.allclose(unweighted.coefficients, UNWEIGHTED_EXPECTED_COEFFS, atol=5e-5)


@mark.skipif(qr.rq_fortran is None, reason="requires the compiled Fortran extension")
def test_process_preprocessing_matches_independent_solves():
    rng = np.random.default_rng(0)
    n = 6_000
    ds = pl.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)}).with_columns(
        y=1.0 + 2.0 * pl.col("x1") - pl.col("x2") + rng.normal(size=n)
    )
    qs = np.linspace(0.05, 0.95, 10)

    independent = QuantileRegression("y ~ x1 + x2", ds).fit(
        qs, algorithm=QuantileRegressionAlgorithms.FRISCH_NEWTON
    )
    processed = QuantileRegression("y ~ x1 + x2", ds).fit(
        qs, algorithm=QuantileRegressionAlgorithms.PREPROCESSING
    )

    assert np.allclose(independent.coefficients, processed.coefficients, atol=1e-4)


def test_process_preprocessing_falls_back_for_small_samples():
    rng = np.random.default_rng(0)
    n = 100
    ds = pl.DataFrame({"x": rng.normal(size=n)}).with_columns(
        y=1.0 + 2.0 * pl.col("x") + rng.normal(size=n)
    )
    qs = np.array([0.25, 0.5, 0.75])

    independent = QuantileRegression("y ~ x", ds).fit(
        qs, algorithm=QuantileRegressionAlgorithms.FRISCH_NEWTON
    )
    processed = QuantileRegression("y ~ x", ds).fit(
        qs, algorithm=QuantileRegressionAlgorithms.PREPROCESSING
    )

    assert np.array_equal(independent.coefficients, processed.coefficients)


def test_process_methods_preserve_quantile_order():
    """Regression test: sorting qs internally must not reorder the output."""
    rng = np.random.default_rng(0)
    n = 6_000
    ds = pl.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)}).with_columns(
        y=1.0 + 2.0 * pl.col("x1") - pl.col("x2") + rng.normal(size=n)
    )
    qs = np.array([0.9, 0.25, 0.5, 0.1])
    order = np.argsort(qs)
    inverse = np.argsort(order)

    model = QuantileRegression("y ~ x1 + x2", ds)
    unsorted = model.fit(qs, algorithm=QuantileRegressionAlgorithms.PREPROCESSING).coefficients
    sorted_ = model.fit(
        qs[order], algorithm=QuantileRegressionAlgorithms.PREPROCESSING
    ).coefficients
    assert np.allclose(unsorted, sorted_[:, inverse], atol=1e-10)
