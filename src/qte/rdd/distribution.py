from collections.abc import Iterator
from dataclasses import dataclass
from functools import partial
from math import factorial
from typing import NamedTuple

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
    BootstrapMultiplierConfig,
    _attach_standard_errors_to_estimate,
    _make_bootstrap_config,
    _make_bootstrap_runs,
)
from qte.constants import MEDIAN
from qte.custom_types import ColumnName
from qte.helpers import _as_iterable_array, _get_mean_differences, _get_quantile_differences
from qte.names import (
    MEAN_CONTROL_ID,
    MEAN_TREATED_ID,
    QUANTILE_CONTROL_VAL_ID,
    QUANTILE_ID,
    QUANTILE_TREATED_VAL_ID,
    SE_ID,
)
from qte.rdd.custom_types import (
    LocalPolynomialConfig,
    RddInference,
    RunningVariableConfig,
    _Side,
)
from qte.rdd.results import RddDistributionResult
from qte.stats import Ecdf, get_quantiles

_EPS = 1e-8
_GRID_CHUNK_SIZE = 64
_DENSITY_BANDWIDTH_FACTOR = 1.5
_DENSITY_BANDWIDTH_EXPONENT = 1.0 / 7.0
_MIN_DENSITY_RATIO = 1e-6


class _RddSideFit(NamedTuple):
    design: NDArray[np.float64]
    coefficient_maker: NDArray[np.float64]

    @property
    def intercept_maker(self) -> NDArray[np.float64]:
        """Weights that extract the boundary intercept."""
        return self.coefficient_maker[0]

    def get_residuals(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Residuals from this side's local-polynomial fit."""
        return values - self.design @ (self.coefficient_maker @ values)


class _RddData(NamedTuple):
    y: NDArray[np.float64]
    d: NDArray[np.float64]
    n_obs: int
    is_above_threshold: NDArray[np.bool_]
    total_weights: NDArray[np.float64]
    below_thres_fit: _RddSideFit
    above_thres_fit: _RddSideFit


class _Residuals(NamedTuple):
    """Observation-level residuals split below and above the cutoff."""

    below: NDArray[np.float64]
    above: NDArray[np.float64]


@dataclass(frozen=True)
class _RddDistributionIntermediateResult:
    """Point estimates from one (bootstrap) draw."""

    qtt: pl.DataFrame
    att: pl.DataFrame
    first_stage: float
    outcome_grid_size: int
    grid: NDArray[np.float64]
    treated_cdf_monotone: NDArray[np.float64]
    control_cdf_monotone: NDArray[np.float64]
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


def _estimate_density_from_cdf(
    cdf: NDArray[np.float64],
    grid: NDArray[np.float64],
    value: float,
    bandwidth: float,
) -> float:
    """Kernel-smooth the CDF increments and evaluate the density at ``value``."""
    mass = np.diff(cdf, prepend=0.0)
    total = float(mass.sum())
    if total <= 0.0:
        return 0.0

    standardized = (value - grid) / bandwidth
    kernel = np.exp(-0.5 * standardized * standardized) / (np.sqrt(2.0 * np.pi) * bandwidth)
    return max(float(np.sum(mass * kernel) / total), 0.0)


def _make_gaussian_kernel_density(
    values: NDArray[np.float64],
    center: float,
    bandwidth: float,
) -> NDArray[np.float64]:
    """Evaluate a Gaussian kernel density contribution at ``center``."""
    standardized = (values - center) / bandwidth
    return np.exp(-0.5 * standardized * standardized) / (np.sqrt(2.0 * np.pi) * bandwidth)


def _get_density_bandwidth(
    y: NDArray[np.float64],
    total_weights: NDArray[np.float64],
    *,
    exponent: float,
    factor: float = 1.0,
) -> float:
    """Return a Silverman-type bandwidth for the outcome density."""
    positive_weights = total_weights > 0.0
    local_y = y[positive_weights]
    local_weights = total_weights[positive_weights]
    local_mean = float(np.average(local_y, weights=local_weights))
    local_sd = float(np.sqrt(np.average((local_y - local_mean) ** 2, weights=local_weights)))
    effective_n = max(2.0, float(np.count_nonzero(positive_weights)))
    if local_sd <= 0.0:
        local_sd = float(np.std(y))
    return max(factor * 1.06 * local_sd * effective_n ** (-exponent), _EPS)


def _make_local_polynomial_design(
    r: NDArray[np.float64],
    *,
    side: _Side,
    degree: int,
) -> NDArray[np.float64]:
    """Build the local-polynomial design ``[1, r, r^2 / 2, ...]``."""
    if r.size < 2:
        raise ValueError(f"Not enough observations on the {side.value} side of the cutoff.")

    columns = [np.ones(r.size)]
    for power in range(1, degree + 1):
        columns.append(r**power / float(factorial(power)))
    return np.column_stack(columns)


def _make_local_polynomial_operator(
    design: NDArray[np.float64],
    weights: NDArray[np.float64],
    *,
    side: _Side,
) -> NDArray[np.float64]:
    """Return the local-polynomial operator for one side's design."""
    if weights.max() <= 0.0:
        raise ValueError(f"All kernel weights are zero on the {side.value} side of the cutoff.")
    weights = weights / weights.max()

    weighted_design = weights[:, None] * design
    gram = design.T @ weighted_design
    return np.linalg.solve(gram, weighted_design.T)


def _compute_complier_cdfs(
    y: NDArray[np.float64],
    treatment: NDArray[np.float64],
    grid: NDArray[np.float64],
    is_above_threshold: NDArray[np.bool_],
    below_intercept_maker: NDArray[np.float64],
    above_intercept_maker: NDArray[np.float64],
    first_stage: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute both complier CDFs in one pass over the outcome grid."""
    n_grid = grid.size
    treated_above, treated_below = np.empty(n_grid), np.empty(n_grid)
    control_above, control_below = np.empty(n_grid), np.empty(n_grid)

    # We do that chunkwise since the indiciator matrix is potentially large in memory (n_obs times
    # n_grid_poitns)
    for start in range(0, n_grid, _GRID_CHUNK_SIZE):
        stop = min(start + _GRID_CHUNK_SIZE, n_grid)
        indicators = y[:, None] <= grid[None, start:stop]
        treated = treatment[:, None] * indicators
        control = (1.0 - treatment)[:, None] * indicators

        treated_above[start:stop] = above_intercept_maker @ treated[is_above_threshold]
        treated_below[start:stop] = below_intercept_maker @ treated[~is_above_threshold]
        control_above[start:stop] = above_intercept_maker @ control[is_above_threshold]
        control_below[start:stop] = below_intercept_maker @ control[~is_above_threshold]

    treated_cdf = np.clip((treated_above - treated_below) / first_stage, 0.0, 1.0)
    control_cdf = np.clip((control_below - control_above) / first_stage, 0.0, 1.0)
    return treated_cdf, control_cdf


def _get_means(
    y: NDArray[np.float64],
    d: NDArray[np.float64],
    is_above_threshold: NDArray[np.bool_],
    below_intercept_maker: NDArray[np.float64],
    above_intercept_maker: NDArray[np.float64],
    first_stage: float,
) -> tuple[float, float]:
    """Return the complier mean outcomes for the treated and control states."""
    treated_outcome = d * y
    control_outcome = (1.0 - d) * y
    m_t = float(
        (
            above_intercept_maker @ treated_outcome[is_above_threshold]
            - below_intercept_maker @ treated_outcome[~is_above_threshold]
        )
        / first_stage
    )
    m_c = float(
        (
            below_intercept_maker @ control_outcome[~is_above_threshold]
            - above_intercept_maker @ control_outcome[is_above_threshold]
        )
        / first_stage
    )
    return m_t, m_c


def _estimate_densities_at_jump_for_sharp_design(
    y: NDArray[np.float64],
    d: NDArray[np.float64],
    q1s: NDArray[np.float64],
    q0s: NDArray[np.float64],
    *,
    is_above_threshold: NDArray[np.bool_],
    below_thres_fit: _RddSideFit,
    above_thres_fit: _RddSideFit,
    first_stage: float,
    total_weights: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Estimate treated and control outcome densities at the cutoff."""
    density_bandwidth = _get_density_bandwidth(y, total_weights, exponent=1.0 / 5.0)

    def density_at(center: float, group: int) -> float:
        kernel = _make_gaussian_kernel_density(y, center, density_bandwidth)
        plus = above_thres_fit.intercept_maker @ (
            kernel[is_above_threshold] * (d[is_above_threshold] == group)
        )
        minus = below_thres_fit.intercept_maker @ (
            kernel[~is_above_threshold] * (d[~is_above_threshold] == group)
        )
        jump = plus - minus if group == 1 else minus - plus
        return max(float(jump / first_stage), 0.0)

    return (
        np.array([density_at(q1, 1) for q1 in q1s]),
        np.array([density_at(q0, 0) for q0 in q0s]),
    )


def _estimate_density_from_at_jump_for_fuzzy_design(
    y: NDArray[np.float64],
    point: _RddDistributionIntermediateResult,
    q1s: NDArray[np.float64],
    q0s: NDArray[np.float64],
    total_weights: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Estimate complier densities from the monotone CDFs."""
    density_bandwidth = _get_density_bandwidth(
        y,
        total_weights,
        exponent=_DENSITY_BANDWIDTH_EXPONENT,
        factor=_DENSITY_BANDWIDTH_FACTOR,
    )
    return (
        np.array(
            [
                _estimate_density_from_cdf(
                    point.treated_cdf_monotone, point.grid, q1, density_bandwidth
                )
                for q1 in q1s
            ]
        ),
        np.array(
            [
                _estimate_density_from_cdf(
                    point.control_cdf_monotone, point.grid, q0, density_bandwidth
                )
                for q0 in q0s
            ]
        ),
    )


def _compute_point_estimate(
    data: _RddData,
    qs: NDArray[np.float64],
    outcome_grid: int | ArrayLike,
) -> _RddDistributionIntermediateResult:
    """Compute the RD complier quantiles and LATE without bootstrap inference."""
    # Extract some data
    y, d = data.y, data.d
    below, above = data.below_thres_fit, data.above_thres_fit
    is_above_threshold = data.is_above_threshold

    # Compute the first stage (denomiator of the Wald estimator)
    first_stage = float(
        above.intercept_maker @ d[is_above_threshold]
        - below.intercept_maker @ d[~is_above_threshold]
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
        below.intercept_maker,
        above.intercept_maker,
        first_stage,
    )

    # Compute distributions
    grid = _make_outcome_grid(y, data.total_weights, outcome_grid)
    treated_cdf, control_cdf = _compute_complier_cdfs(
        y,
        d,
        grid,
        data.is_above_threshold,
        below.intercept_maker,
        above.intercept_maker,
        first_stage,
    )

    qtt = _get_quantile_differences(
        qs,
        Ecdf(grid, np.sort(treated_cdf)).evaluate_inverse(qs, side="right"),
        Ecdf(grid, np.sort(control_cdf)).evaluate_inverse(qs, side="right"),
    )

    return _RddDistributionIntermediateResult(
        qtt=qtt,
        att=_get_mean_differences(m_t, m_c),
        first_stage=first_stage,
        outcome_grid_size=grid.size,
        grid=grid,
        treated_cdf_monotone=np.maximum.accumulate(treated_cdf),  # [0.2,0.3,0.25]->[0.2,0.3,0.3]
        control_cdf_monotone=np.maximum.accumulate(control_cdf),
        n_obs=data.n_obs,
    )


def _prepare_data(
    ds: pl.DataFrame,
    outcome: ColumnName,
    treatment: ColumnName,
    running_variable: RunningVariableConfig,
    weights: ColumnName,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Extract the outcome, treatment, running variable and sampling weights."""
    return (
        ds[outcome].to_numpy(),
        ds[treatment].to_numpy(),
        ds[running_variable.name].to_numpy() - running_variable.cutoff,
        ds[weights].to_numpy(),
    )


def _make_rdd_data(
    ds: pl.DataFrame,
    outcome: ColumnName,
    treatment: ColumnName,
    running_variable: RunningVariableConfig,
    *,
    local_polynomial_config: LocalPolynomialConfig,
    weights: ColumnName,
) -> _RddData:
    """Build the local-polynomial design and weights for one RD sample."""
    y, d, r, sampling_weights = _prepare_data(ds, outcome, treatment, running_variable, weights)
    total_weights = sampling_weights * local_polynomial_config.kernel._compute_weights(
        r / local_polynomial_config.bandwidth
    )
    is_above_threshold = r >= 0.0
    degree = local_polynomial_config.degree

    design_below = _make_local_polynomial_design(
        r[~is_above_threshold], side=_Side.LEFT, degree=degree
    )
    operator_below = _make_local_polynomial_operator(
        design_below, total_weights[~is_above_threshold], side=_Side.LEFT
    )
    design_above = _make_local_polynomial_design(
        r[is_above_threshold], side=_Side.RIGHT, degree=degree
    )
    operator_above = _make_local_polynomial_operator(
        design_above, total_weights[is_above_threshold], side=_Side.RIGHT
    )

    return _RddData(
        y=y,
        d=d,
        n_obs=ds.height,
        is_above_threshold=is_above_threshold,
        total_weights=total_weights,
        below_thres_fit=_RddSideFit(design=design_below, coefficient_maker=operator_below),
        above_thres_fit=_RddSideFit(design=design_above, coefficient_maker=operator_above),
    )


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
    """Build the RD data and compute the point estimates in one step."""
    data = _make_rdd_data(
        ds,
        outcome,
        treatment,
        running_variable,
        local_polynomial_config=local_polynomial_config,
        weights=weights,
    )
    return _compute_point_estimate(data, qs, outcome_grid=outcome_grid)


def _perturb_intercept(
    residuals: NDArray[np.float64],
    mult_draw: NDArray[np.float64],
    fit: _RddSideFit,
) -> NDArray[np.float64]:
    """Perturb one side's local-polynomial intercept."""
    if residuals.ndim == 1:  # Residual can be one dim (mean) or multirdim (one for each quantile)
        return fit.intercept_maker @ (mult_draw * residuals[:, None])
    return mult_draw.T @ (fit.intercept_maker[:, None] * residuals)


def _perturb_wald_ratio(
    pert_numerator: NDArray[np.float64],
    numerator: float | NDArray[np.float64],
    pert_first_stage: NDArray[np.float64],
    first_stage: float,
) -> NDArray[np.float64]:
    """Perturb a local Wald ratio."""
    if pert_numerator.ndim == 1:
        return (pert_numerator - numerator * pert_first_stage) / first_stage
    return (
        pert_numerator - np.asarray(numerator)[None, :] * pert_first_stage[:, None]
    ) / first_stage


def _check_density_scale(
    treated_density: NDArray[np.float64],
    control_density: NDArray[np.float64],
) -> None:
    """Raise if an estimated density is negligible relative to the others."""
    density_scale = max(float(treated_density.max()), float(control_density.max()), _EPS)
    if np.any(treated_density <= _MIN_DENSITY_RATIO * density_scale) or np.any(
        control_density <= _MIN_DENSITY_RATIO * density_scale
    ):
        raise ValueError(
            "Estimated complier density is too small at one or more quantiles. "
            "Drop those quantiles or increase the bandwidth."
        )


def _make_multiplier_draws(
    rng: np.random.Generator,
    n_obs: int,
    n_draws: int,
    is_above_threshold: NDArray[np.bool_],
) -> Iterator[tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """Yield below/above multiplier matrices in memory-safe chunks."""
    chunk_size = max(1, min(n_draws, 4_000_000 // max(n_obs, 1)))
    for start in range(0, n_draws, chunk_size):
        stop = min(start + chunk_size, n_draws)
        mult_draw = rng.standard_normal((n_obs, stop - start))
        yield mult_draw[~is_above_threshold], mult_draw[is_above_threshold]


def _compute_se_via_multiplier_bootstrap(
    data: _RddData,
    point: _RddDistributionIntermediateResult,
    n_draws: int,
    seed: int | None,
) -> tuple[
    NDArray[np.float64],
    float,
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Multiplier-bootstrap pointwise SEs, constant-width and sup-t draws."""
    if n_draws < 2:
        raise ValueError("At least two multiplier draws are required.")

    y, d = data.y, data.d
    below_tresh_fit, above_thres_fit = data.below_thres_fit, data.above_thres_fit
    is_above_threshold = data.is_above_threshold
    total_weights = data.total_weights

    qs = point.qtt[QUANTILE_ID].to_numpy()
    q1s, q0s = (
        point.qtt[QUANTILE_TREATED_VAL_ID].to_numpy(),
        point.qtt[QUANTILE_CONTROL_VAL_ID].to_numpy(),
    )
    m_t, m_c = point.att[MEAN_TREATED_ID][0], point.att[MEAN_CONTROL_ID][0]
    first_stage = point.first_stage

    # Density estimate at the cutoff per quantile varies depending on the design (sharp vs fuzzy).
    # It is eventually used for the derivation of the quantile effect (see below).
    density_at_jump_treated, density_at_jump_control = (
        _estimate_densities_at_jump_for_sharp_design(
            y,
            d,
            q1s,
            q0s,
            is_above_threshold=is_above_threshold,
            below_thres_fit=below_tresh_fit,
            above_thres_fit=above_thres_fit,
            first_stage=first_stage,
            total_weights=total_weights,
        )
        if first_stage >= 1.0 - _EPS
        else _estimate_density_from_at_jump_for_fuzzy_design(y, point, q1s, q0s, total_weights)
    )
    _check_density_scale(density_at_jump_treated, density_at_jump_control)

    # Step: Get residuals for the perturbation in the multiplier boostrap.
    # Residuals for the first stage, the quantile indicators and the means.
    residual_d: _Residuals = _Residuals(
        below=below_tresh_fit.get_residuals(d[~is_above_threshold]),
        above=above_thres_fit.get_residuals(d[is_above_threshold]),
    )

    # Quantiles: matrices are n_obs, n_quantiels).
    treated_quantile_values = d[:, None] * (y[:, None] <= q1s[None, :])
    control_quantile_values = (1.0 - d)[:, None] * (y[:, None] <= q0s[None, :])
    treated_quantile_residuals: _Residuals = _Residuals(
        below=below_tresh_fit.get_residuals(treated_quantile_values[~is_above_threshold]),
        above=above_thres_fit.get_residuals(treated_quantile_values[is_above_threshold]),
    )
    control_quantile_residuals: _Residuals = _Residuals(
        below=below_tresh_fit.get_residuals(control_quantile_values[~is_above_threshold]),
        above=above_thres_fit.get_residuals(control_quantile_values[is_above_threshold]),
    )

    # Means
    treated_mean_residuals: _Residuals = _Residuals(
        below=below_tresh_fit.get_residuals((d * y)[~is_above_threshold]),
        above=above_thres_fit.get_residuals((d * y)[is_above_threshold]),
    )
    control_mean_residuals: _Residuals = _Residuals(
        below=below_tresh_fit.get_residuals(((1.0 - d) * y)[~is_above_threshold]),
        above=above_thres_fit.get_residuals(((1.0 - d) * y)[is_above_threshold]),
    )

    rng = np.random.default_rng(seed)
    perturbed_qte_parts: list[NDArray[np.float64]] = []
    perturbed_att_parts: list[NDArray[np.float64]] = []

    for mult_draw_below, mult_draw_above in _make_multiplier_draws(
        rng, y.size, n_draws, is_above_threshold
    ):
        # Perturbations of the components of the Wald ratio; the mean follows below.
        # This is the perturbed first stage.
        pert_first_stage = _perturb_intercept(
            residual_d.above, mult_draw_above, above_thres_fit
        ) - _perturb_intercept(residual_d.below, mult_draw_below, below_tresh_fit)

        # Quantiles, vectorized across all requested quantiles.
        pert_treated = _perturb_intercept(
            treated_quantile_residuals.above, mult_draw_above, above_thres_fit
        ) - _perturb_intercept(treated_quantile_residuals.below, mult_draw_below, below_tresh_fit)
        pert_control = _perturb_intercept(
            control_quantile_residuals.below, mult_draw_below, below_tresh_fit
        ) - _perturb_intercept(control_quantile_residuals.above, mult_draw_above, above_thres_fit)
        pert_cdf_treated = _perturb_wald_ratio(pert_treated, qs, pert_first_stage, first_stage)
        pert_cdf_control = _perturb_wald_ratio(pert_control, qs, pert_first_stage, first_stage)

        # Lastly perturb the quantile function (this is the quantile effect).
        pert_qte_chunk = (
            -pert_cdf_treated / density_at_jump_treated[None, :]
            + pert_cdf_control / density_at_jump_control[None, :]
        )

        # Perturbation of the mean
        pert_m_t = _perturb_wald_ratio(
            _perturb_intercept(
                treated_mean_residuals.above,
                mult_draw_above,
                above_thres_fit,
            )
            - _perturb_intercept(
                treated_mean_residuals.below,
                mult_draw_below,
                below_tresh_fit,
            ),
            m_t,
            pert_first_stage,
            first_stage,
        )
        pert_m_c = _perturb_wald_ratio(
            _perturb_intercept(
                control_mean_residuals.below,
                mult_draw_below,
                below_tresh_fit,
            )
            - _perturb_intercept(
                control_mean_residuals.above,
                mult_draw_above,
                above_thres_fit,
            ),
            m_c,
            pert_first_stage,
            first_stage,
        )

        perturbed_qte_parts.append(pert_qte_chunk)
        perturbed_att_parts.append(pert_m_t - pert_m_c)

    pert_qte = np.vstack(perturbed_qte_parts)
    pert_att = np.concatenate(perturbed_att_parts)

    pointwise_se = pert_qte.std(axis=0, ddof=1)
    # CHS/Stata use a constant half-width: the quantile of sup |G_b(theta)|.
    sup_draws = np.max(np.abs(pert_qte), axis=1)
    # Studentised sup-t draws: use the pointwise bootstrap SE in the denominator.
    safe_pointwise_se = np.maximum(pointwise_se, _EPS)
    sup_t_draws = np.max(np.abs(pert_qte) / safe_pointwise_se[None, :], axis=1)
    return pointwise_se, float(pert_att.std(ddof=1)), sup_draws, sup_t_draws


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
    bootstrap_config: BootstrapConfig | BootstrapMultiplierConfig | int = 100,
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
    bootstrap_config : `BootstrapConfig`, `BootstrapMultiplierConfig` or int, default=100
        Inference settings. A `BootstrapConfig` or integer requests the case
        bootstrap and pointwise standard errors. A
        `BootstrapMultiplierConfig` requests multiplier draws, pointwise
        standard errors and a constant-width uniform band.

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

    # Standard errors via multiplier Bootstrap
    if isinstance(bootstrap_config, BootstrapMultiplierConfig):
        data = _make_rdd_data(
            ds,
            outcome,
            treatment,
            running_variable,
            local_polynomial_config=local_polynomial_config,
            weights=weights,
        )
        point = _compute_point_estimate(data, qs, outcome_grid=outcome_grid)
        qtt_se, att_se, sup_draws, sup_t_draws = _compute_se_via_multiplier_bootstrap(
            data,
            point,
            n_draws=bootstrap_config.n_draws,
            seed=bootstrap_config.seed,
        )
        return RddDistributionResult(
            qtt=point.qtt.with_columns(pl.Series(SE_ID, qtt_se)),
            att=point.att.with_columns(pl.Series(SE_ID, [att_se])),
            outcome=outcome,
            group=None,
            treatment=treatment,
            running=running_variable.name,
            cutoff=running_variable.cutoff,
            local_polynomial_config=local_polynomial_config,
            outcome_grid_size=point.outcome_grid_size,
            first_stage=point.first_stage,
            n_obs=point.n_obs,
            inference=RddInference.MULTIPLIER,
            uniform_sup_draws=sup_draws,
            uniform_sup_t_draws=sup_t_draws,
        )

    # Standard errors via case bootstrap
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
    bootstrap_config = _make_bootstrap_config(bootstrap_config)
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
        inference=RddInference.BOOTSTRAP,
    )
