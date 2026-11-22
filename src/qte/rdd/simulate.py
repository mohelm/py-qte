"""Simulate data for the local quantile treatment effect in an RD design.

The design follows the Monte Carlo model in Frandsen, Frölich and Melly (2012).
A normal running variable ``r`` indexes units. The potential outcomes are

    Y(0) = r + eps_0
    Y(1) = Y(0) - eps_1,

so the treatment effect is heterogeneous and equal to ``-eps_1``. Treatment
status is determined by

    D = 1{Y(1) - Y(0) + alpha * 1{r >= 0} >= eps_D},

where ``eps_D`` is a normal disturbance. Exceeding the cutoff lowers the cost of
treatment by ``alpha``. The design is fuzzy: treatment is possible on both sides
of the cutoff, and the local compliers are those with
``0 < eps_D + eps_1 <= alpha``. Because the running variable enters both
potential outcomes and treatment selection, the outcome difference at the
cutoff is not the treatment effect; the local distribution regression has to
control for ``r``.
"""

import numpy as np
import polars as pl
from numpy.typing import ArrayLike
from scipy.stats import norm

from qte.constants import MEDIAN
from qte.helpers import _as_iterable_array, _get_quantile_differences
from qte.names import (
    COMPLIER_ID,
    OBSERVED_OUTCOME_ID,
    POTENTIAL_OUTCOME_CTRL_ID,
    POTENTIAL_OUTCOME_TREAT_ID,
    RUNNING_ID,
    TREATMENT_ID,
)

_D0_ID = "d_0"
_D1_ID = "d_1"


def simulate_rdd_data(
    n: int = 10_000,
    *,
    alpha: float = 2.5,
    sigma_r: float = 1.0,
    sigma_0: float = 1.0,
    sigma_1: float = 1.0,
    sigma_d: float = 1.0,
    seed: int | None = None,
) -> pl.DataFrame:
    """Simulate a fuzzy RD design with heterogeneous quantile effects.

    Parameters
    ----------
    n : int, default=10_000
        Number of units.
    alpha : float, default=2.5
        Size of the treatment-cost reduction at the cutoff. Larger values
        produce a stronger first stage.
    sigma_r : float, default=1.0
        Standard deviation of the running variable.
    sigma_0 : float, default=1.0
        Standard deviation of the untreated outcome disturbance.
    sigma_1 : float, default=1.0
        Standard deviation of the treatment-effect disturbance. The treatment
        effect is ``-eps_1``.
    sigma_d : float, default=1.0
        Standard deviation of the treatment-selection disturbance.
    seed : int, optional
        Seed for `numpy.random.default_rng`.

    Returns
    -------
    polars.DataFrame
        Columns ``running``, ``treatment``, ``y``, ``y_0``, ``y_1``, ``d_0``,
        ``d_1`` and ``complier``.
    """
    rng = np.random.default_rng(seed)
    running = rng.normal(0.0, sigma_r, n)
    eps_0 = rng.normal(0.0, sigma_0, n)
    eps_1 = rng.normal(0.0, sigma_1, n)
    eps_d = rng.normal(0.0, sigma_d, n)

    y_0 = running + eps_0
    y_1 = y_0 - eps_1

    # Potential treatment status just below and just above the cutoff.
    selection_index = eps_d + eps_1
    d_0 = selection_index <= 0.0
    d_1 = selection_index <= alpha
    complier = d_1 & ~d_0

    treatment = np.where(running >= 0.0, d_1, d_0).astype(int)
    y = np.where(treatment == 1, y_1, y_0)

    return pl.DataFrame(
        {
            RUNNING_ID: running,
            TREATMENT_ID: treatment,
            OBSERVED_OUTCOME_ID: y,
            POTENTIAL_OUTCOME_CTRL_ID: y_0,
            POTENTIAL_OUTCOME_TREAT_ID: y_1,
            _D0_ID: d_0.astype(int),
            _D1_ID: d_1.astype(int),
            COMPLIER_ID: complier.astype(int),
        }
    )


def get_true_local_effects(
    qs: ArrayLike = MEDIAN,
    *,
    alpha: float = 2.5,
    sigma_1: float = 1.0,
    sigma_d: float = 1.0,
    n_oracle: int = 1_000_000,
    seed: int | None = None,
) -> tuple[pl.DataFrame, float]:
    """Compute the complier quantile effects and LATE of the simulated design.

    The running variable is independent of the disturbances, so at the cutoff
    the complier potential outcomes are ``Y(0) = eps_0`` and
    ``Y(1) = eps_0 - eps_1`` for units with ``0 < eps_D + eps_1 <= alpha``. A
    large oracle sample approximates their quantiles and mean difference.

    Parameters
    ----------
    qs : array_like, default=0.5
        Quantiles in ``(0, 1)`` at which to compute the true effect.
    alpha : float, default=2.5
        Treatment-cost reduction at the cutoff.
    sigma_1 : float, default=1.0
        Standard deviation of the treatment-effect disturbance.
    sigma_d : float, default=1.0
        Standard deviation of the treatment-selection disturbance.
    n_oracle : int, default=1_000_000
        Size of the oracle sample.
    seed : int, optional
        Seed for `numpy.random.default_rng`.

    Returns
    -------
    tuple of polars.DataFrame and float
        The true quantile effects and the true local average treatment effect.
    """
    qs = _as_iterable_array(qs)
    rng = np.random.default_rng(seed)
    eps_0 = rng.normal(0.0, 1.0, n_oracle)
    eps_1 = rng.normal(0.0, sigma_1, n_oracle)
    eps_d = rng.normal(0.0, sigma_d, n_oracle)

    selection_index = eps_d + eps_1
    complier = (selection_index > 0.0) & (selection_index <= alpha)

    y_0 = eps_0[complier]
    y_1 = (eps_0 - eps_1)[complier]
    qtt = _get_quantile_differences(
        qs,
        np.quantile(y_1, qs),
        np.quantile(y_0, qs),
    )
    late = float(y_1.mean() - y_0.mean())
    return qtt, late


def get_true_first_stage(
    *,
    alpha: float = 2.5,
    sigma_1: float = 1.0,
    sigma_d: float = 1.0,
) -> float:
    """Return the population share of local compliers.

    With the normal disturbances of `simulate_rdd_data`, the selection index is
    normal with variance ``sigma_1**2 + sigma_d**2``, so the complier share is
    ``Phi(alpha / sd) - Phi(0)``.

    Parameters
    ----------
    alpha : float, default=2.5
        Treatment-cost reduction at the cutoff.
    sigma_1 : float, default=1.0
        Standard deviation of the treatment-effect disturbance.
    sigma_d : float, default=1.0
        Standard deviation of the treatment-selection disturbance.

    Returns
    -------
    float
        Probability of being a local complier.
    """
    scale = np.sqrt(sigma_1 * sigma_1 + sigma_d * sigma_d)
    return float(norm.cdf(alpha / scale) - 0.5)
