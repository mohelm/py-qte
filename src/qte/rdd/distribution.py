from dataclasses import dataclass
from functools import partial

import numpy as np
import polars as pl
from numpy.typing import ArrayLike, NDArray

from qte._validation import (
    check_columns_are_in_dataset,
    check_missing_values_in_dataset,
    check_numeric,
    check_outcome,
    check_outcome_grid,
    check_quantiles,
    check_treatment_is_binary,
    check_weights,
)
from qte.bootstrap import (
    BootstrapConfig,
    _attach_standard_errors_to_estimate,
    _make_bootstrap_config,
    _make_bootstrap_runs,
)
from qte.constants import MEDIAN
from qte.custom_types import ColumnName
from qte.helpers import _as_iterable_array, _get_mean_differences, _get_quantile_differences
from qte.rdd._types import Side
from qte.rdd.custom_types import LocalPolynomialConfig, RunningVariableConfig
from qte.rdd.results import RddDistributionResult
from qte.stats import Ecdf, get_quantiles

_EPS = 1e-8
_GRID_CHUNK_SIZE = 64


@dataclass(frozen=True)
class _RddDistributionIntermediateResult:
    """Point estimates from one (bootstrap) draw."""

    qtt: pl.DataFrame
    att: pl.DataFrame
    first_stage: float
    outcome_grid_size: int
    n_obs: int


def _make_outcome_grid(
    y: NDArray[np.float64],
    weights: NDArray[np.float64],
    outcome_grid: int | ArrayLike,
) -> NDArray[np.float64]:
    """Build the outcome grid from weighted quantiles or an explicit sequence."""
    if not isinstance(outcome_grid, (int, np.integer)):
        return np.sort(np.asarray(outcome_grid, dtype=float))

    positive = weights > 0
    if positive.sum() >= 2:
        y = y[positive]
        weights = weights[positive]
    else:
        weights = np.ones_like(y)

    probs = np.linspace(0.0, 1.0, int(outcome_grid))
    return np.unique(get_quantiles(probs, y, weights))


def _make_local_linear_intercept_weights(
    r: NDArray[np.float64],
    weights: NDArray[np.float64],
    *,
    side: Side,
) -> NDArray[np.float64]:
    """Return the local-linear weights that extract the boundary intercept."""
    if r.size < 2:
        raise ValueError(f"Not enough observations on the {side.value} side of the cutoff.")

    design = np.column_stack((np.ones(r.size), r))
    if weights.max() <= 0.0:
        raise ValueError(f"All kernel weights are zero on the {side.value} side of the cutoff.")
    weights = weights / weights.max()

    weighted_design = weights[:, None] * design
    gram = design.T @ weighted_design
    return np.linalg.solve(gram, weighted_design.T)[0]


def _compute_complier_cdfs(
    y: NDArray[np.float64],
    treatment: NDArray[np.float64],
    grid: NDArray[np.float64],
    intercept_maker_above_thres: NDArray[np.float64],
    is_above_threshold: NDArray[np.bool_],
    intercept_maker_below_thres: NDArray[np.float64],
    first_stage: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute both complier CDFs in one pass over the outcome grid."""
    n_grid = grid.size
    treated_right = np.empty(n_grid)
    treated_left = np.empty(n_grid)
    control_right = np.empty(n_grid)
    control_left = np.empty(n_grid)

    # We do that chunkwise since the indiciator matrix is potentially large in memory (n_obs times
    # n_grid_poitns)
    for start in range(0, n_grid, _GRID_CHUNK_SIZE):
        stop = min(start + _GRID_CHUNK_SIZE, n_grid)
        indicators = y[:, None] <= grid[None, start:stop]
        treated = treatment[:, None] * indicators
        control = (1.0 - treatment)[:, None] * indicators

        treated_right[start:stop] = intercept_maker_above_thres @ treated[is_above_threshold]
        treated_left[start:stop] = intercept_maker_below_thres @ treated[~is_above_threshold]
        control_right[start:stop] = intercept_maker_above_thres @ control[is_above_threshold]
        control_left[start:stop] = intercept_maker_below_thres @ control[~is_above_threshold]

    treated_cdf = (treated_right - treated_left) / first_stage
    control_cdf = (control_left - control_right) / first_stage
    return treated_cdf, control_cdf


def _get_means(
    y: NDArray[np.float64],
    d: NDArray[np.float64],
    is_above_threshold: NDArray[np.bool_],
    intercept_maker_above_thres: NDArray[np.float64],
    intercept_maker_below_thres: NDArray[np.float64],
    first_stage: float,
) -> tuple[float, float]:
    """Return the complier mean outcomes for the treated and control states."""
    treated_outcome = d * y
    control_outcome = (1.0 - d) * y
    m_t = float(
        (
            intercept_maker_above_thres @ treated_outcome[is_above_threshold]
            - intercept_maker_below_thres @ treated_outcome[~is_above_threshold]
        )
        / first_stage
    )
    m_c = float(
        (
            intercept_maker_below_thres @ control_outcome[~is_above_threshold]
            - intercept_maker_above_thres @ control_outcome[is_above_threshold]
        )
        / first_stage
    )
    return m_t, m_c


def _compute_local_distribution_effects(
    ds: pl.DataFrame,
    outcome: ColumnName,
    treatment: ColumnName,
    running_variable: RunningVariableConfig,
    qs: NDArray[np.float64],
    *,
    local_polynomial_config: LocalPolynomialConfig,
    outcome_grid: int | ArrayLike,
    weights: ColumnName,
) -> _RddDistributionIntermediateResult:
    """Compute the RD complier quantiles and LATE without bootstrap inference."""
    y, d, sampling_weights = (
        ds[outcome].to_numpy(),
        ds[treatment].to_numpy(),
        ds[weights].to_numpy(),
    )
    r = ds[running_variable.name].to_numpy() - running_variable.cutoff

    total_weights = sampling_weights * local_polynomial_config.kernel._compute_weights(
        r / local_polynomial_config.bandwidth
    )

    is_above_threshold = r > 0.0

    # Intercept weights: A = e1'(X'WX)^-1 X'W, so A @ y is the local-linear
    # intercept at the cutoff for side outcomes y.
    intercept_maker_above_thres = _make_local_linear_intercept_weights(
        r[is_above_threshold], total_weights[is_above_threshold], side=Side.RIGHT
    )
    intercept_maker_below_thres = _make_local_linear_intercept_weights(
        r[~is_above_threshold], total_weights[~is_above_threshold], side=Side.LEFT
    )

    # Compute first stage
    first_stage = (
        intercept_maker_above_thres @ d[is_above_threshold]
        - intercept_maker_below_thres @ d[~is_above_threshold]
    )
    if first_stage <= _EPS:
        raise ValueError(
            "The treatment probability does not jump at the cutoff. The RD design "
            "needs a positive first stage to identify the complier distribution."
        )

    # The local average treatment effect uses the same local Wald ratios with the
    # outcome in place of the indicator.
    m_t, m_c = _get_means(
        y,
        d,
        is_above_threshold,
        intercept_maker_above_thres,
        intercept_maker_below_thres,
        first_stage,
    )

    # Compute distributions
    grid = _make_outcome_grid(y, total_weights, outcome_grid)
    treated_cdf, control_cdf = _compute_complier_cdfs(
        y,
        d,
        grid,
        intercept_maker_above_thres,
        is_above_threshold,
        intercept_maker_below_thres,
        first_stage,
    )

    # Rearrangement: sort the estimated CDF values to enforce monotonicity before
    # inversion, as in Chernozhukov et al. (2010) and Stata's `rddqte`.
    treated_cdf = np.sort(np.clip(treated_cdf, 0.0, 1.0))
    control_cdf = np.sort(np.clip(control_cdf, 0.0, 1.0))

    qtt = _get_quantile_differences(
        qs,
        Ecdf(grid, treated_cdf).evaluate_inverse(qs, side="right"),
        Ecdf(grid, control_cdf).evaluate_inverse(qs, side="right"),
    )
    att = _get_mean_differences(m_t, m_c)

    return _RddDistributionIntermediateResult(
        qtt=qtt,
        att=att,
        first_stage=first_stage,
        outcome_grid_size=grid.size,
        n_obs=ds.height,
    )


def estimate_rdd_distribution_effects(
    ds: pl.DataFrame,
    outcome: ColumnName,
    treatment: ColumnName,
    running_variable: RunningVariableConfig,
    qs: ArrayLike = MEDIAN,
    *,
    local_polynomial_config: LocalPolynomialConfig,
    outcome_grid: int | ArrayLike = 100,
    weights: ColumnName | None = None,
    bootstrap_config: BootstrapConfig | int = 100,
) -> RddDistributionResult:
    """Estimate local quantile treatment effects by local distribution regression.

    The estimator is the Frandsen, Frölich and Melly (2012) local-linear
    distribution regression. It targets the quantile treatment effect for
    compliers at the cutoff and handles both sharp and fuzzy designs; the sharp
    design is the special case with a first stage of one.

    A bandwidth has to be supplied because the estimator does not select one
    internally. The outcome grid for the distribution regressions is built from
    weighted quantiles of the local sample and defaults to 100 points.

    Parameters
    ----------
    ds : polars.DataFrame
        Input data.
    outcome : ColumnName
        Name of the outcome column.
    treatment : ColumnName
        Name of the binary treatment column (``1`` treated, ``0`` not).
    running_variable : `RunningVariableConfig`
        Running (forcing) variable and the cutoff at which the treatment
        probability jumps.
    qs : array_like, default=0.5
        Quantiles in ``(0, 1)`` at which to estimate the effect.
    local_polynomial_config : `LocalPolynomialConfig`
        Bandwidth and kernel for the local-linear distribution regressions.
    outcome_grid : int or array_like, default=100
        Outcome values at which the distribution regressions are evaluated. An
        integer ``k`` builds a grid from ``k`` weighted quantiles of the local
        sample; an explicit sequence is used as given.
    weights : ColumnName, optional
        Name of a column with sampling weights.
    bootstrap_config : `BootstrapConfig` or int, default=100
        Bootstrap settings, or the number of replications.

    Returns
    -------
    `RddDistributionResult`
        Complier quantile effects and the local average treatment effect, with
        bootstrap standard errors.

    See Also
    --------
    `qte.rdd.simulate.simulate_rdd_data`, `qte.rdd.simulate.get_true_local_effects`
    """
    check_columns_are_in_dataset(
        ds,
        [outcome, treatment, running_variable.name, *([weights] if weights is not None else [])],
    )
    check_missing_values_in_dataset(
        ds,
        [
            outcome,
            treatment,
            running_variable.name,
            *([weights] if weights is not None else []),
        ],
    )
    check_outcome(ds[outcome])
    check_treatment_is_binary(ds[treatment])
    check_numeric(ds[running_variable.name])
    if weights is not None:
        check_weights(ds[weights])

    qs = _as_iterable_array(qs)
    check_quantiles(qs)
    check_outcome_grid(outcome_grid)
    if weights is None:
        weights = "_w"
        ds = ds.with_columns(pl.lit(1.0).alias(weights))

    bootstrap_config = _make_bootstrap_config(bootstrap_config)
    fcn = partial(
        _compute_local_distribution_effects,
        outcome=outcome,
        treatment=treatment,
        running_variable=running_variable,
        qs=qs,
        local_polynomial_config=local_polynomial_config,
        outcome_grid=outcome_grid,
        weights=weights,
    )
    point = fcn(ds)
    runs = _make_bootstrap_runs(
        ds,
        fcn,
        n_iter=bootstrap_config.n_iter,
        seed=bootstrap_config.seed,
        n_workers=bootstrap_config.n_workers,
    )
    qtt, att = _attach_standard_errors_to_estimate(point, runs)
    return RddDistributionResult(
        qtt=qtt,
        att=att,
        outcome=outcome,
        group=None,
        treatment=treatment,
        running=running_variable.name,
        cutoff=running_variable.cutoff,
        local_polynomial_config=local_polynomial_config,
        outcome_grid_size=point.outcome_grid_size,
        first_stage=point.first_stage,
        n_obs=point.n_obs,
    )
