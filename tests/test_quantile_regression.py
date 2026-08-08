from qte.constants import MEDIAN
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
