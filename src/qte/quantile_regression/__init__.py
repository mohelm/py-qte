"""Quantile regression on a Fortran Frisch-Newton interior-point solver."""

import numpy as np
import polars as pl
from formulaic import Formula
from numpy.typing import NDArray

from qte.custom_types import ColumnName
from qte.quantile_regression import rq_fortran  # type: ignore


def _fast_quantreg(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    q: float,
    weights: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Fit a single quantile via the Fortran solver.

    Parameters
    ----------
    X : NDArray
        Design matrix, shape ``(n_obs, n_coeffs)``.
    y : NDArray
        Response, shape ``(n_obs,)``.
    q : float
        Quantile in ``(0, 1)``.
    weights : NDArray, optional
        Positive observation weights, shape ``(n_obs,)``.

    Returns
    -------
    NDArray
        Coefficients, shape ``(n_coeffs,)``.
    """
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        w = w / w.mean()
        X = X * w[:, None]
        y = y * w

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
    """Fitted coefficients of a :class:`QuantileRegression`.

    Parameters
    ----------
    coefficients : NDArray
        Fitted coefficients, shape ``(n_coeffs, n_quantiles)``.
    x_fit : NDArray
        Design matrix used for fitting, shape ``(n_obs, n_coeffs)``.
    formula : Formula
        Formula the model was fit with.

    Attributes
    ----------
    coefficients : NDArray
        Fitted coefficients, shape ``(n_coeffs, n_quantiles)``.
    formula : Formula
        Formula the model was fit with.
    """

    def __init__(self, coefficients: NDArray, x_fit: NDArray[np.float64], formula: Formula) -> None:
        self.coefficients = coefficients
        self._x_fit = x_fit
        self.formula = formula

    def predict(self, ds: pl.DataFrame | None = None) -> NDArray:
        """Predict fitted quantiles.

        Parameters
        ----------
        ds : DataFrame, optional
            Data to predict on. Defaults to the fitting data.

        Returns
        -------
        NDArray
            Predictions, shape ``(n_obs, n_quantiles)``.
        """
        return (
            self.formula.rhs.get_model_matrix(ds, output="numpy")  # type: ignore
            if ds is not None
            else self._x_fit
        ) @ self.coefficients


class QuantileRegression:
    """Quantile regression on a formula and a Polars frame.

    Parameters
    ----------
    formula : str
        Model formula, e.g. ``"y ~ x"``.
    ds : DataFrame
        Data the formula is evaluated against.
    """

    def __init__(self, formula: str, ds: pl.DataFrame) -> None:
        self.formula = formula
        self._ds = ds

    def fit(self, qs: NDArray, *, weights_c: ColumnName | None = None) -> QuantileRegressionResult:
        """Fit the model at one or more quantiles.

        Parameters
        ----------
        qs : NDArray
            Quantiles to fit, each in ``(0, 1)``.
        weights_c : str, optional
            Column of ``ds`` holding positive observation weights.

        Returns
        -------
        QuantileRegressionResult
            The fitted result.
        """
        fml = Formula(self.formula)
        y, X = fml.get_model_matrix(self._ds, output="numpy")
        weights = self._ds[weights_c].to_numpy() if weights_c is not None else None
        coeffs = np.column_stack([_fast_quantreg(X, y.ravel(), q, weights) for q in qs])
        return QuantileRegressionResult(coeffs, x_fit=X, formula=fml)
