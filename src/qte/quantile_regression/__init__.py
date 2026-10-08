"""Quantile regression using a Fortran interior-point solver.

The Fortran ``rq_fortran`` extension implements the Frisch-Newton
interior-point algorithm. When the extension is not available (for example on
platforms without a Fortran toolchain), estimation falls back to
:class:`statsmodels.regression.quantile_regression.QuantReg`.
"""

from enum import StrEnum

import numpy as np
import polars as pl
from formulaic import Formula
from numpy.typing import NDArray

from qte.custom_types import ColumnName

try:
    from qte.quantile_regression import rq_fortran  # type: ignore
except ImportError:  # pragma: no cover - depends on the build environment
    rq_fortran = None  # type: ignore[assignment]


def _solve(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    q: float,
) -> NDArray[np.float64]:
    """Solve one quantile on an already weighted design."""
    if rq_fortran is None:
        from statsmodels.regression.quantile_regression import QuantReg

        return np.asarray(QuantReg(y, X).fit(q=q).params, dtype=np.float64)

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

    return _solve(X, y, q)


def _preprocess_sorted(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    qs: NDArray[np.float64],
    weights: NDArray[np.float64] | None = None,
    *,
    m_factor: float = 0.8,
    eps: float = 1e-6,
) -> NDArray[np.float64]:
    """Fit a sequence of already sorted quantiles with preprocessing.

    Port of ``quantreg::rq.fit.ppro`` (Chernozhukov, Fernandez-Val and Melly,
    2020): the coefficient from the previous quantile classifies the
    observations whose residual sign is already known, so they can be collapsed
    into two aggregated observations and the interior-point solve runs on a
    reduced problem. The sign checks keep the estimates numerically equal to a
    fresh solve.

    Parameters
    ----------
    X : NDArray
        Design matrix, shape ``(n_obs, n_coeffs)``.
    y : NDArray
        Response, shape ``(n_obs,)``.
    qs : NDArray
        Quantiles in ``(0, 1)``, sorted ascending.
    weights : NDArray, optional
        Positive observation weights, shape ``(n_obs,)``.
    m_factor : float, default=0.8
        Multiplier for the number of observations kept between the two
        quantile cut-offs (``Mm.factor`` in ``quantreg``).
    eps : float, default=1e-6
        Floor for the estimated residual standard errors.

    Returns
    -------
    NDArray
        Coefficients, shape ``(n_coeffs, n_quantiles)``.
    """
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        w = w / w.mean()
        X = X * w[:, None]
        y = y * w

    n_obs, n_coeffs = X.shape
    coef = np.empty((n_coeffs, qs.shape[0]))

    # If there is just a single quantile the pre-processing algorithm does not help.
    if qs.shape[0] < 2:
        return _solve(X, y, qs[0])[:, None]
    # The reduced problem keeps about n sqrt(k) dtau observations. For small
    # samples or few coefficients it becomes ill-conditioned (and the overhead
    # is not worth it), so fall back to independent solves.
    initial_effective_sample_size = m_factor * n_obs * np.sqrt(n_coeffs) * np.max(np.diff(qs))

    # TODO: take that out/expose.
    if n_obs < 5_000 or initial_effective_sample_size < 5 * n_coeffs:
        return np.column_stack([_solve(X, y, q) for q in qs])

    # Get initial b
    b = _solve(X, y, qs[0])
    coef[:, 0] = b

    # Conservative estimate of the residual standard errors, see Portnoy and
    # Koenker (1997). ``chol(X'X)`` is upper triangular in R, so in numpy we
    # transpose the lower Cholesky factor.
    try:
        upper = np.linalg.cholesky(X.T @ X).T
        x_upper_inv = np.linalg.solve(upper.T, X.T).T
    except np.linalg.LinAlgError:
        return np.column_stack([_solve(X, y, q) for q in qs])

    # z_i from the paper.
    band = np.maximum(eps, np.sqrt((x_upper_inv**2).sum(axis=1)))

    for j, tau in enumerate(qs[1:], start=1):
        tau = float(tau)
        r = y - X @ b
        not_optimal = True
        mm = 1.0
        while not_optimal:
            effective_sample_size = mm * initial_effective_sample_size
            lo_q = max(1.0 / n_obs, tau - effective_sample_size / (2 * n_obs))
            hi_q = min(tau + effective_sample_size / (2 * n_obs), (n_obs - 1) / n_obs)

            # This is relative directly in algorithm 2 it is how the sets J_H and J_L are
            # calculated.
            kappa = np.quantile(r / band, [lo_q, hi_q])
            sl = r < band * kappa[0]
            su = r > band * kappa[1]
            while True:
                effective_sample_identifer = ~su & ~sl
                xx = X[effective_sample_identifer]
                yy = y[effective_sample_identifer]
                if sl.any():
                    xx = np.vstack([xx, X[sl].sum(axis=0)])
                    yy = np.append(yy, y[sl].sum())
                if su.any():
                    xx = np.vstack([xx, X[su].sum(axis=0)])
                    yy = np.append(yy, y[su].sum())
                if xx.shape[0] < n_coeffs or np.linalg.matrix_rank(xx) < n_coeffs:
                    # The reduced problem is rank deficient (too few kept rows
                    # or a degenerate design), which the Fortran solver does not
                    # report. Fall back to a full solve for this quantile.
                    b = _solve(X, y, tau)
                    r = y - X @ b
                    not_optimal = False
                    break
                b = _solve(xx, yy, tau)
                r = y - X @ b
                su_bad = (r < 0) & su
                sl_bad = (r > 0) & sl
                bad_signs = int((su_bad | sl_bad).sum())
                if bad_signs > 0:
                    if bad_signs > 0.1 * effective_sample_size:
                        mm *= 2
                        break
                    su = su & ~su_bad
                    sl = sl & ~sl_bad
                else:
                    not_optimal = False
                    break
        coef[:, j] = b

    return coef


def _solve_qr_with_preprocessing(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    qs: NDArray[np.float64],
    weights: NDArray[np.float64] | None = None,
    *,
    m_factor: float = 0.8,
    eps: float = 1e-6,
) -> NDArray[np.float64]:

    # For the preprocessing algorithm, we need ordered quantiles since quantile order is exploited.
    # We return the old order exploiting that argsorts sorts and argsort (argsorts) restores.
    order = np.argsort(qs, kind="stable")
    return _preprocess_sorted(X, y, qs[order], weights, m_factor=m_factor, eps=eps)[
        :, np.argsort(order)
    ]


class QuantileRegressionResult:
    """Fitted coefficients of a `QuantileRegression`.

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


class QuantileRegressionAlgorithms(StrEnum):
    """Algorithm used to fit the quantile regression coefficient process.

    Attributes
    ----------
    FRISCH_NEWTON
        Exact per-quantile Frisch-Newton interior point (the baseline).
    PREPROCESSING
        Exact preprocessing of the process (Chernozhukov, Fernandez-Val and
        Melly, 2020, Algorithm 2); falls back to `FRISCH_NEWTON`.
    """

    FRISCH_NEWTON = "frisch_newton"
    PREPROCESSING = "preprocessing"


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

    def fit(
        self,
        qs: NDArray,
        *,
        weights: ColumnName | None = None,
        algorithm: QuantileRegressionAlgorithms = QuantileRegressionAlgorithms.FRISCH_NEWTON,
    ) -> QuantileRegressionResult:
        """Fit the model at one or more quantiles.

        Parameters
        ----------
        qs : NDArray
            Quantiles to fit, each in ``(0, 1)``.
        weights : ColumnName, optional
            Column of ``ds`` holding positive observation weights.
        algorithm : `QuantileRegressionAlgorithms`, default=`QuantileRegressionAlgorithms.FRISCH_NEWTON`
            ``FRISCH_NEWTON`` solves each quantile independently;
            ``PREPROCESSING`` preprocesses the quantile process using the
            previous coefficients (Algorithm 2) and falls back to independent
            solves when the reduced problem is too small or rank deficient.

        Returns
        -------
        QuantileRegressionResult
            The fitted result.
        """
        fml = Formula.from_spec(self.formula)
        y, X = fml.get_model_matrix(self._ds, output="numpy")
        sample_weights = self._ds[weights].to_numpy() if weights is not None else None
        coeffs = (
            _solve_qr_with_preprocessing(X, y.ravel(), np.asarray(qs, dtype=float), sample_weights)
            if algorithm == QuantileRegressionAlgorithms.PREPROCESSING
            else np.column_stack([_fast_quantreg(X, y.ravel(), q, sample_weights) for q in qs])
        )
        return QuantileRegressionResult(coeffs, x_fit=X, formula=fml)
