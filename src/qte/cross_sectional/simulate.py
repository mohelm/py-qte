"""Simulate cross-sectional data for quantile treatment effect estimators.

Two independent designs are provided.

``simulate_simple_data``
    A randomised experiment without covariates. The outcome is
    ``y = treatment_effect * treat + N(0, error_var)``, so the treatment effect
    is the same at every quantile and equals ``treatment_effect``.

``simulate_covariate_data``
    A fixed observational setting. Five covariates always enter the design and
    ``n_noise_covariates`` additional covariates are noise. With
    ``x = (x0, ..., x4)`` uniform on ``[-1, 1]``::

        propensity(x) = logistic(x1 - x3 + 1.5 * 1{x0 > 0})
        D             ~ Bernoulli(propensity(x))
        Y(0) = f_loc(0, x) + sqrt(error_var) * f_scale(0, x) * eps
        Y(1) = f_loc(1, x) + sqrt(error_var) * f_scale(1, x) * eps

    where

        f_loc(d, x)   = 0.5*d + 2*d*x4 + 2*1{x1>0.1} - 1.7*1{x0*x2>0} - 3*x3
        f_scale(d, x) = sqrt(2 + 0.5*d + 0.3*d*x1)

    and a single ``eps ~ N(0, 1)`` is shared by both potential outcomes.
    Treatment is confounded through ``x0``, ``x1`` and ``x3`` and the treated
    outcome is more dispersed, so the quantile treatment effect increases with
    the quantile while the average effect is 0.5. The conditional quantile is
    linear in the features
    ``[x3, x4, 1{x1>0.1}, 1{x0*x2>0}, sqrt(2.5+0.3*x1)]``, so
    outcome-regression, IPW and doubly-robust estimators are correctly specified
    when given them.

The population effects of the covariate design are recovered by simulating a
large population with `get_true_quantiles` and `get_true_means`.
"""

import numpy as np
import polars as pl
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit

from qte.constants import MEDIAN
from qte.cross_sectional.custom_types import CausalTarget
from qte.helpers import _as_iterable_array, _get_quantile_differences
from qte.names import (
    EFFECT_ID,
    MEAN_CONTROL_ID,
    MEAN_TREATED_ID,
    OBSERVED_OUTCOME_ID,
    POTENTIAL_OUTCOME_CTRL_ID,
    POTENTIAL_OUTCOME_TREAT_ID,
)

TREATMENT_COL = "treat"
N_USED_COVARIATES = 5


def f_loc(
    d: NDArray[np.float64], x: NDArray[np.float64], effect_scale: float = 1.0
) -> NDArray[np.float64]:
    """Conditional location of the covariate design."""
    return (
        effect_scale * (0.5 * d + 2.0 * d * x[:, 4])
        + 2.0 * (x[:, 1] > 0.1)
        - 1.7 * (x[:, 0] * x[:, 2] > 0)
        - 3.0 * x[:, 3]
    )


def f_scale(
    d: NDArray[np.float64], x: NDArray[np.float64], effect_scale: float = 1.0
) -> NDArray[np.float64]:
    """Conditional scale of the covariate design."""
    return np.sqrt(effect_scale * (0.5 * d + 0.3 * d * x[:, 1]) + 2.0)


def propensity(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Logit propensity score of the covariate design."""
    return expit(x[:, 1] - x[:, 3] + 1.5 * (x[:, 0] > 0))


def simulate_simple_data(
    n: int = 1000,
    *,
    treatment_effect: float = 1.0,
    treatment_share: float = 0.5,
    error_var: float = 1.0,
    seed: int | None = None,
) -> pl.DataFrame:
    """Simulate a randomised experiment without covariates.

    Parameters
    ----------
    n : int
        Number of units.
    treatment_effect : float
        Constant shift of the treated outcome; the true QTE and ATT equal it.
    treatment_share : float
        Probability of treatment.
    error_var : float
        Variance of the error term.
    seed : int, optional
        Seed for `numpy.random.default_rng`.

    Returns
    -------
    polars.DataFrame
        Columns ``treat`` and ``y``.
    """
    rng = np.random.default_rng(seed)
    treat = rng.binomial(1, treatment_share, size=n)
    y = treatment_effect * treat + float(np.sqrt(error_var)) * rng.normal(size=n)
    return pl.DataFrame({TREATMENT_COL: treat, OBSERVED_OUTCOME_ID: y})


def simulate_covariate_data(
    n: int = 1000,
    *,
    n_noise_covariates: int = 0,
    effect_scale: float = 1.0,
    error_var: float = 1.0,
    seed: int | None = None,
) -> pl.DataFrame:
    """Simulate the fixed observational setting with covariates.

    Parameters
    ----------
    n : int
        Number of units.
    n_noise_covariates : int
        Number of additional noise covariates. The five covariates that enter
        the design are always generated, so the frame has
        ``5 + n_noise_covariates`` covariate columns.
    effect_scale : float
        Multiplies the treatment effect sizes (the coefficients of ``d`` in the
        location and scale components). ``1`` is the original design and ``0``
        removes the treatment effect.
    error_var : float
        Multiplies the conditional variance of both arms.
    seed : int, optional
        Seed for `numpy.random.default_rng`.

    Returns
    -------
    polars.DataFrame
        Columns ``x0, ..., x{p-1}``, ``treat``, ``y`` and the potential
        outcomes ``y_0`` and ``y_1``.
    """
    rng = np.random.default_rng(seed)
    p = N_USED_COVARIATES + n_noise_covariates
    x = rng.uniform(-1.0, 1.0, size=(n, p))
    treat = rng.binomial(1, propensity(x)).astype(int)
    eps = rng.normal(size=n)
    zeros = np.zeros(n)
    ones = np.ones(n)
    y_0 = (
        f_loc(zeros, x, effect_scale)
        + float(np.sqrt(error_var)) * f_scale(zeros, x, effect_scale) * eps
    )
    y_1 = (
        f_loc(ones, x, effect_scale)
        + float(np.sqrt(error_var)) * f_scale(ones, x, effect_scale) * eps
    )
    y = np.where(treat == 1, y_1, y_0)
    columns: dict[str, NDArray[np.float64]] = {f"x{i}": x[:, i] for i in range(p)}
    columns.update(
        {
            TREATMENT_COL: treat,
            OBSERVED_OUTCOME_ID: y,
            POTENTIAL_OUTCOME_CTRL_ID: y_0,
            POTENTIAL_OUTCOME_TREAT_ID: y_1,
        }
    )
    return pl.DataFrame(columns)


def _potential_outcomes(
    target: CausalTarget, n_oracle: int, effect_scale: float, error_var: float, seed: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Simulate potential outcomes of the covariate design for the given target."""
    ds = simulate_covariate_data(
        n_oracle, effect_scale=effect_scale, error_var=error_var, seed=seed
    )
    y_0 = ds[POTENTIAL_OUTCOME_CTRL_ID].to_numpy()
    y_1 = ds[POTENTIAL_OUTCOME_TREAT_ID].to_numpy()
    if target == CausalTarget.QTT:
        mask = ds[TREATMENT_COL].to_numpy() == 1
        y_0, y_1 = y_0[mask], y_1[mask]
    return y_0, y_1


def get_true_means(
    *,
    target: CausalTarget = CausalTarget.QTE,
    n_oracle: int = 1_000_000,
    effect_scale: float = 1.0,
    error_var: float = 1.0,
    seed: int = 0,
) -> pl.DataFrame:
    """Simulation-based potential-outcome means and the average effect.

    With ``target=QTE`` the means are over the whole population (``effect`` is
    the ATE); with ``target=QTT`` they are over the treated (``effect`` is the
    ATT). The truth is recovered by simulating ``n_oracle`` units.

    Parameters
    ----------
    target : `CausalTarget`, default=`CausalTarget.QTE`
        Estimand to recover, ``QTE`` (population) or ``QTT`` (treated).
    n_oracle : int, default=1_000_000
        Size of the simulated population.
    effect_scale : float, default=1.0
        Treatment effect scale used in `simulate_covariate_data`.
    error_var : float, default=1.0
        Error variance used in `simulate_covariate_data`.
    seed : int, default=0
        Seed for the simulated population.

    Returns
    -------
    polars.DataFrame
        Columns ``m_c``, ``m_t`` and ``effect`` (``m_t - m_c``).
    """
    y_0, y_1 = _potential_outcomes(target, n_oracle, effect_scale, error_var, seed)
    mean_0, mean_1 = float(np.mean(y_0)), float(np.mean(y_1))
    return pl.DataFrame(
        {
            MEAN_CONTROL_ID: [mean_0],
            MEAN_TREATED_ID: [mean_1],
            EFFECT_ID: [mean_1 - mean_0],
        }
    )


def get_true_quantiles(
    qs: ArrayLike = MEDIAN,
    *,
    target: CausalTarget = CausalTarget.QTE,
    n_oracle: int = 1_000_000,
    effect_scale: float = 1.0,
    error_var: float = 1.0,
    seed: int = 0,
) -> pl.DataFrame:
    """Simulation-based potential-outcome quantiles and the quantile effect.

    The truth is recovered by simulating ``n_oracle`` units and taking empirical
    quantiles. With ``target=QTT`` the treated subsample is used.

    Parameters
    ----------
    qs : array_like
        Probabilities in ``(0, 1)``.
    target : `CausalTarget`
        ``QTE`` for the population, ``QTT`` for the treated.
    n_oracle : int
        Size of the simulated population.
    effect_scale : float
        Treatment effect scale used in `simulate_covariate_data`.
    error_var : float
        Error variance used in `simulate_covariate_data`.
    seed : int
        Seed for the simulated population.

    Returns
    -------
    polars.DataFrame
        Columns ``q``, ``q_t``, ``q_c`` and ``effect`` (``q_t - q_c``).
    """
    qs = _as_iterable_array(qs)

    y_0, y_1 = _potential_outcomes(target, n_oracle, effect_scale, error_var, seed)
    return _get_quantile_differences(qs, np.quantile(y_1, qs), np.quantile(y_0, qs))
