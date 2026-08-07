import numpy as np
import polars as pl
from formulaic import Formula
from numpy.typing import NDArray

from qte.quantile_regression import rq_fortran


def fast_quantreg(
    X: NDArray[np.float64], y: NDArray[np.float64], q: float
) -> NDArray[np.float64]:
    n_obs, n_coeffs = X.shape

    a = np.asfortranarray(X.T)
    y_input = -y
    rhs = (1 - q) * X.sum(axis=0)

    d = np.ones(n_obs)
    u = np.ones(n_obs)
    beta = 0.99995
    eps = 1e-6

    wn = np.zeros((n_obs, 9), order="F")
    wn[:, 0] = 1 - q

    wp = np.zeros((n_coeffs, n_coeffs + 3), order="F")
    nit = np.zeros(3, dtype=np.int32)
    info = 0

    rq_fortran.rqfnb(a, y_input, rhs, d, u, beta, eps, wn, wp, nit, info)
    return -wp[:, 0]


class QuantileRegressionResult:
    def __init__(self, coefficients: NDArray, x: NDArray[np.float64], formula: Formula):
        self.coefficients = coefficients
        self._x = x
        self.formula = formula

    def predict(self, ds: pl.DataFrame | None = None) -> NDArray:
        if ds is not None:
            _, X = self.formula.get_model_matrix(ds, output="numpy")
            return X @ self.coefficients
        return self._x @ self.coefficients


class QuantileRegression:
    def __init__(self, formula: str, data: pl.DataFrame) -> None:
        self.formula = formula
        self.ds = data

    def fit(self, qs: NDArray) -> QuantileRegressionResult:
        fml = Formula(self.formula)
        y, X = fml.get_model_matrix(self.ds, output="numpy")
        coeffs = np.column_stack([fast_quantreg(X, y.ravel(), q) for q in qs])
        return QuantileRegressionResult(coeffs, x=X, formula=fml)
