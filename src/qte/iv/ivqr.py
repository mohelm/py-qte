"""Instrumental-variable quantile regression (Chernozhukov & Hansen).

The estimator fits the structural quantile function

    q(tau, d, x) = d' alpha(tau) + x' beta(tau)

by inverse quantile regression. For a candidate value of ``alpha`` the outcome
is shifted to ``y - d' alpha`` and a conventional quantile regression is run on
the projected endogenous variables and the covariates. The candidate whose
projected-instrument coefficients are closest to zero in the Wald metric is the
estimate; the remaining coefficients are read off the final regression. This is
the estimation part of the R package ``IVQR`` (``ivqr()``) and of Chernozhukov
and Hansen (2005, 2006).

Unlike the local-QTE estimators in this package, IVQR does not target compliers
and does not need a binary treatment or instrument.
"""

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import polars as pl
import statsmodels.api as sm
from formulaic import Formula, ModelMatrix
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm
from statsmodels.sandbox.regression.gmm import IV2SLS

from qte.constants import MEDIAN
from qte.custom_types import ColumnName, FormulaRhs
from qte.helpers import _as_iterable_array
from qte.iv.custom_types import IvqrTreatment
from qte.iv.results import IvqrResult
from qte.names import EFFECT_ID, ESTIMATE_ID, QUANTILE_ID, SE_ID, TERM_ID
from qte.quantile_regression import _fast_quantreg

_DAHAT_ID = "_ivqr_dhat"
_EPS = np.finfo(float).eps ** (2.0 / 3.0)


@dataclass(frozen=True)
class _IvqrQuantileResult:
    """Estimates from one IVQR quantile."""

    tau: float
    qte: float
    inverse_jacobian: NDArray[np.float64]
    covariate_estimates: NDArray[np.float64]


def _get_design_matrix_from_formula(formula: str, ds: pl.DataFrame) -> ModelMatrix:
    """Evaluate a right-hand-side formula and return its dense design matrix."""
    matrix = Formula.from_spec(formula).get_model_matrix(ds, output="numpy")
    if not isinstance(matrix, ModelMatrix):
        raise TypeError("Expected a single design matrix from a right-hand-side formula.")
    return matrix


def _get_first_stage_fitted_vals(
    ds: pl.DataFrame,
    treatment: ColumnName,
    instruments: tuple[ColumnName, ...],
    covariates: FormulaRhs,
) -> NDArray[np.float64]:
    """Fit the linear first stage and return the treatment fitted values."""
    formula = f"{treatment} ~ {' + '.join(instruments)} + {covariates}"
    response, design = Formula.from_spec(formula).get_model_matrix(ds, output="numpy")
    return sm.OLS(response.ravel(), design).fit().fittedvalues


def _compute_2sls(
    y: NDArray[np.float64],
    d: NDArray[np.float64],
    x_design: NDArray[np.float64],
    instrument_design: ModelMatrix,
) -> tuple[float, float]:
    """Estimate the average IV effect and its standard error by 2SLS."""
    x = np.column_stack((d, x_design))
    result = IV2SLS(y, x, instrument_design).fit()
    return result.params[0], result.bse[0]


def _rq_bandwidth(tau: float, n_obs: int) -> float:
    """Hall-Sheather bandwidth used by ``quantreg::summary.rq(se = "ker")``."""
    x = norm.ppf(tau)
    f = norm.pdf(x)
    return (
        n_obs ** (-1.0 / 3.0)
        * norm.ppf(0.975) ** (2.0 / 3.0)
        * ((1.5 * f**2) / (2.0 * x**2 + 1.0)) ** (1.0 / 3.0)
    )


def _quantile_regression_kernel_covariance(
    design: NDArray[np.float64],
    outcome: NDArray[np.float64],
    coefficients: NDArray[np.float64],
    tau: float,
) -> NDArray[np.float64]:
    """Kernel covariance matrix of a conventional quantile regression.

    This is the covariance returned by ``quantreg::summary.rq(..., se = "ker",
    covariance = TRUE)`` and is used to form the IVQR Wald objective.
    """
    bandwidth = _rq_bandwidth(tau, design.shape[0])
    while tau - bandwidth < 0.0 or tau + bandwidth > 1.0:
        bandwidth /= 2.0

    residuals = outcome - design @ coefficients
    bandwidth = (norm.ppf(tau + bandwidth) - norm.ppf(tau - bandwidth)) * min(
        np.sqrt(np.var(residuals, ddof=1)),
        (np.quantile(residuals, 0.75) - np.quantile(residuals, 0.25)) / 1.34,
    )
    # TODO: probably rather raise here.
    bandwidth = max(float(bandwidth), np.finfo(float).eps)
    density = norm.pdf(residuals / bandwidth) / bandwidth
    weighted_cross_product = design.T @ (density[:, None] * design)
    inverse = np.linalg.inv(weighted_cross_product)
    return tau * (1.0 - tau) * inverse @ (design.T @ design) @ inverse


def _silverman_bandwidth(residuals: NDArray[np.float64]) -> float:
    """Silverman bandwidth used for the IVQR sandwich covariance."""
    bandwidth = (
        1.364
        * (2.0 * np.sqrt(np.pi)) ** (-1.0 / 5.0)
        * np.std(residuals, ddof=1)
        * residuals.shape[0] ** (-1.0 / 5.0)
    )
    return max(float(bandwidth), np.finfo(float).eps)


def _compute_inverse_of_jacobian(
    design: NDArray[np.float64],
    dx: NDArray[np.float64],
    residuals: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute the inverse of the IVQR Jacobian."""
    bandwidth = _silverman_bandwidth(residuals)
    while True:
        kernel = (np.abs(residuals) < bandwidth).astype(float)
        # The Jacobian is [dim(alpha)=1,dim(beta)] times [dim(alpha)=1,dim(beta)].
        jacobian = (1.0 / (2.0 * design.shape[0] * bandwidth)) * (kernel[:, None] * design).T @ dx
        try:
            return np.linalg.inv(jacobian)
        except np.linalg.LinAlgError:
            bandwidth *= 1.1


def _compute_standard_error_at_quantile(
    tau: float,
    design: NDArray[np.float64],
    inverse_jacobian: NDArray[np.float64],
    n_obs: int,
) -> NDArray[np.float64]:
    """Compute pointwise standard errors for one IVQR quantile."""
    # This is the diagonal score covariance S(tau, tau) from the 2006 paper.
    score_covariance = tau * (1.0 - tau) / n_obs * (design.T @ design)
    covariance = (1.0 / n_obs) * inverse_jacobian @ score_covariance @ inverse_jacobian.T
    return np.sqrt(np.diag(covariance))


def _make_coefficient_frame(
    terms: Sequence[str],
    estimates: NDArray[np.float64],
    standard_errors: NDArray[np.float64],
    qs: NDArray[np.float64],
) -> pl.DataFrame:
    """Assemble a long coefficient frame with one row per term and quantile."""
    return pl.DataFrame(
        {
            TERM_ID: [term for _ in qs for term in terms],
            QUANTILE_ID: [float(tau) for tau in qs for _ in terms],
            ESTIMATE_ID: [float(value) for value in estimates],
            SE_ID: [float(value) for value in standard_errors],
        }
    )


def _compute_ivqr_at_quantile(
    tau: float,
    *,
    y: NDArray[np.float64],
    d: NDArray[np.float64],
    grid: NDArray[np.float64],
    design: NDArray[np.float64],
    dx: NDArray[np.float64],
    n_workers: int = 1,
) -> _IvqrQuantileResult:
    """Compute IVQR estimates and standard errors for one quantile."""

    # This is the Chernozhukov and Hansen algorithm. Fix alpha and run a regular quantile
    # regression (y-d*alpha) on the exogenous variables (instr index and covariates). Then call the
    # parameter on the instrument index gamma. Our eventual object for choosing alpha is a norm for
    # this parameter (which here is a function of alpha).
    def objective(alpha: float) -> NDArray[np.float64]:
        shifted_outcome = y - d * alpha
        coefficients = _fast_quantreg(design, shifted_outcome, tau)
        covariance = _quantile_regression_kernel_covariance(
            design, shifted_outcome, coefficients, tau
        )
        # First column is the Wald statistic; the rest are the QR coefficients.
        wald = coefficients[0] ** 2 / covariance[0, 0]
        return np.concatenate(([wald], coefficients))

    if n_workers == 1 or grid.shape[0] == 1:
        candidate_results = np.asarray([objective(alpha) for alpha in grid])
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            candidate_results = np.asarray(list(executor.map(objective, grid)))

    best_index = np.argmin(candidate_results[:, 0])  # 0 holds the Wald statistic
    alpha = grid[best_index]
    beta = candidate_results[best_index, 2:]  # 0 Wald, 1 instrument, remainder betas (covariates)

    # dx = [d, x_design]; np.r_[alpha,beta] is theta from the paper.
    residuals = y - dx @ np.r_[alpha, beta]

    return _IvqrQuantileResult(
        tau=tau,
        qte=alpha,
        inverse_jacobian=_compute_inverse_of_jacobian(design, dx, residuals),
        covariate_estimates=beta,
    )


def _compute_mean_qte_standard_error(
    influence_factor: NDArray[np.float64],
    qs: NDArray[np.float64],
) -> float:
    """Compute the analytical standard error of the average QTE."""
    n_quantiles, n_obs = influence_factor.shape
    bridge_weights = np.minimum.outer(qs, qs) - np.outer(qs, qs)
    covariance = (influence_factor @ influence_factor.T) * bridge_weights / n_obs**2
    return np.sqrt(covariance.sum() / n_quantiles**2)


def _compute_ivqr(
    ds: pl.DataFrame,
    outcome: ColumnName,
    treatment: IvqrTreatment,
    instruments: tuple[ColumnName, ...],
    qs: NDArray[np.float64],
    *,
    covariates: FormulaRhs,
    n_workers: int = 1,
) -> IvqrResult:
    """Compute IVQR coefficients and standard errors without resampling."""
    treatment_name = treatment.name
    grid = np.asarray(treatment.grid, dtype=np.float64)
    d = ds[treatment_name].to_numpy().astype(np.float64)
    y = ds[outcome].to_numpy().astype(np.float64)

    d_hat = _get_first_stage_fitted_vals(ds, treatment_name, instruments, covariates)
    estimation_data = ds.with_columns(pl.Series(_DAHAT_ID, d_hat))

    qr_design: ModelMatrix = _get_design_matrix_from_formula(
        f"~ {_DAHAT_ID} + {covariates}", estimation_data
    )
    if qr_design.model_spec is None:
        raise RuntimeError("Design matrix is missing its model specification.")
    qr_names = list(qr_design.model_spec.column_names)
    # Formulaic gives [intercept, dhat, X]. Reorder once to [dhat, intercept, X]
    # so alpha is column 0. This design is used for QR, J, SE, and influence.
    design = np.column_stack((qr_design[:, 1], qr_design[:, 0], qr_design[:, 2:]))
    dx = np.column_stack((d, design[:, 1:]))

    quantile_results = [
        _compute_ivqr_at_quantile(
            tau,
            y=y,
            d=d,
            grid=grid,
            design=design,
            dx=dx,
            n_workers=n_workers,
        )
        for tau in qs
    ]

    standard_errors = [
        _compute_standard_error_at_quantile(result.tau, design, result.inverse_jacobian, ds.height)
        for result in quantile_results
    ]

    # Now we are just interested in the parameter alpha which is always the first (at index 0)
    # parameter. We want to compute the mean (and its standard error) of this parameter across tau.
    # Stack the "influence factor" of alpha by tau in the horizontal direction so the shape is
    # n_quantiles times n_obs
    qte_influence_factor = np.vstack(
        [design @ result.inverse_jacobian[0, :] for result in quantile_results]
    )
    mean = np.mean(np.asarray([result.qte for result in quantile_results]))
    mean_standard_error = _compute_mean_qte_standard_error(qte_influence_factor, qs)

    projection_design = _get_design_matrix_from_formula(
        f"~ {' + '.join(instruments)} + {covariates}", ds
    )
    mean_from_tsls, standard_error_form_mean_from_tsls = _compute_2sls(
        y, d, dx[:, 1:], projection_design
    )

    return IvqrResult(
        qtt=pl.DataFrame(
            {
                QUANTILE_ID: qs,
                EFFECT_ID: [result.qte for result in quantile_results],
                SE_ID: [se[0] for se in standard_errors],
            }
        ),
        att=pl.DataFrame(
            {
                EFFECT_ID: [mean],
                SE_ID: [mean_standard_error],
            }
        ),
        covariate_coefficients=_make_coefficient_frame(
            [qr_names[0], *qr_names[2:]],
            np.concatenate([result.covariate_estimates for result in quantile_results]),
            np.concatenate([se[1:] for se in standard_errors]),
            qs,
        ),
        outcome=outcome,
        group=None,
        treatment=treatment_name,
        instruments=instruments,
        covariates=covariates,
        n_obs=ds.height,
        tsls=(mean_from_tsls, standard_error_form_mean_from_tsls),
    )


def estimate_ivqr_effects(
    ds: pl.DataFrame,
    outcome: ColumnName,
    treatment: IvqrTreatment,
    instruments: ColumnName | Sequence[ColumnName],
    qs: ArrayLike = MEDIAN,
    *,
    covariates: FormulaRhs = "1",
    n_workers: int = 1,
) -> IvqrResult:
    """Estimate an instrumental-variable quantile regression.

    The structural quantile function is ``q(tau, d, x) = d' alpha(tau) +
    x' beta(tau)``. ``alpha(tau)`` is the coefficient on the endogenous
    treatment and is the quantile treatment effect in the IVQR model;
    ``beta(tau)`` collects the intercept and covariate coefficients.

    Parameters
    ----------
    ds : polars.DataFrame
        Input data.
    outcome : ColumnName
        Name of the outcome column.
    treatment : `IvqrTreatment`
        Endogenous treatment variable with the grid for its structural
        coefficient.
    instruments : ColumnName or sequence of str
        Name(s) of the instrument(s). One or more instruments may be used.
    qs : array_like, default=0.5
        Quantiles in ``(0, 1)`` at which to estimate the coefficients.
    covariates : FormulaRhs, default="1"
        Right-hand side of the covariates. An intercept is always included.
    n_workers : int, default=1
        Number of worker threads used to evaluate the alpha grid within each
        quantile. Results are identical for any value because the grid order is
        preserved.

    Returns
    -------
    `IvqrResult`
        Treatment and covariate coefficient estimates with their standard
        errors, one row per coefficient and quantile.

    See Also
    --------
    `IvqrTreatment`
    `qte.iv.estimate_unconditional_local_effects`
    `qte.iv.estimate_conditional_local_effects`

    Examples
    --------
    Estimate the median effect of 401(k) participation on net financial assets,
    instrumenting participation by eligibility, as in Chernozhukov and Hansen
    (2004):

    >>> import numpy as np
    >>> from qte.datasets import load_pension
    >>> from qte.iv import IvqrTreatment, estimate_ivqr_effects
    >>> res = estimate_ivqr_effects(
    ...     load_pension(),
    ...     "net_tfa",
    ...     treatment=IvqrTreatment("p401", np.arange(0, 25_001, 100)),
    ...     instruments="e401",
    ...     qs=0.5,
    ...     covariates=(
    ...         "C(icat) + C(ecat) + a1 + a2 + a3 + a4 + marr + fsize "
    ...         "+ twoearn + db + pira + hown"
    ...     ),
    ... )
    """
    qs = _as_iterable_array(qs)
    instruments = (instruments,) if isinstance(instruments, str) else tuple(instruments)
    return _compute_ivqr(
        ds,
        outcome,
        treatment,
        instruments,
        qs,
        covariates=covariates,
        n_workers=n_workers,
    )
