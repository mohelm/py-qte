"""Kitagawa (2015) instrument-validity test with the pooled bootstrap.

This module implements the Kitagawa (2015) test of instrument validity, with the
pooled bootstrap of her Algorithm 2.1. The instrument is taken as given: ``instrument == 1`` is the encouraged arm ("on") and
``instrument == 0`` the unencouraged arm ("off"), so no reordering happens. Write
``P`` for the distribution of ``(Y, D)`` given the instrument is on and ``Q`` for
the distribution given it is off, and let ``P_m`` and ``Q_n`` be their empirical
counterparts. Under IV validity (exclusion, random assignment, monotonicity with
``D(on) >= D(off)``),

    P(B, 1) - Q(B, 1) >= 0,      Q(B, 0) - P(B, 0) >= 0,

for every Borel set ``B``, because both differences equal the probability that a
complier's potential outcome lies in ``B``. The test measures the largest
violation of these inequalities over intervals of outcomes,

    T_N = sqrt(m n / N) * max{ sup_B [Q_n(B, 1) - P_m(B, 1)] / (xi v sigma(B, 1)),
                               sup_B [P_m(B, 0) - Q_n(B, 0)] / (xi v sigma(B, 0)) },

with ``sigma^2(B, d) = (1 - lambda) P_m(B, d) (1 - P_m(B, d))
+ lambda Q_n(B, d) (1 - Q_n(B, d))``, ``lambda = m / N``, and a trimming
constant ``xi`` (Kitagawa, 2015, equation 2.1). The unweighted variant replaces
the ``sigma`` denominator by one (equation 2.2).

Critical values come from Algorithm 2.1: the instrument is dropped and ``m``
observations for ``P*_m`` and ``n`` observations for ``Q*_n`` are drawn with
replacement from the pooled empirical distribution
``H_N = lambda P_m + (1 - lambda) Q_n``. Both bootstrap samples therefore follow
the *same* distribution, which imposes the null ``P = Q = H``; Kitagawa (2015,
footnote 11) notes that no recentering is needed for exactly this reason. The
statistic is recomputed on the bootstrap samples, including ``sigma``, and the
p value is the share of replications at least as large as the observed statistic.

The implementation is written in Polars. Resampling on the pooled frame goes
through `qte.bootstrap._make_bootstrap_runs`, which hands each replication a
with-replacement resample of the data, exactly as the other estimators in the
package do; the frame holds only outcome and treatment, because ``H_N`` is a law
of ``(Y, D)``. Empirical CDFs are running weight sums read off with
`polars.Series.search_sorted`, and the supremum over intervals is taken with
NumPy on the ``(endpoint, endpoint, treatment)`` array of interval masses, where
the variance weighting, the trimming and the maximum are plain array algebra.
The observed data and every bootstrap replication enter the same
`_compute_statistic`; they differ only in which two frames are handed to it.

With sampling weights the module works with the weighted empirical measures and a
*case* bootstrap: rows are drawn uniformly from the pooled frame and keep their
weights, so the effective size of a resampled arm is the sum of its drawn weights.
Kitagawa (2015) does not consider weights, so this is an extension rather than the
paper's algorithm: with unit weights it is literally Algorithm 2.1, and with
weights it is the same test for the weighted measures, bootstrapped the way the
rest of the package bootstraps (rows resampled, weights carried as attributes).
It is not the Rao--Wu/pseudo-population bootstrap, which would draw rows with
probability proportional to their weight and give the draws unit weight.

The pooled bootstrap is the procedure in the paper, at the price of rebuilding
the two empirical CDFs in every replication. A multiplier-bootstrap variant of
the same statistic, which perturbs the observed CDFs instead of resampling them,
is kept in ``attic/validity_mult.py`` for reference; the two agree closely in size and
power (both are conservative, because the bootstrap evaluates the inequalities as
if they were binding).
"""

from functools import partial
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from qte.bootstrap import BootstrapConfig, _make_bootstrap_config, _make_bootstrap_runs
from qte.custom_types import ColumnName
from qte.iv.custom_types import ValidityWeighting
from qte.iv.results import InstrumentValidityResult
from qte.iv.validity._common import _MIN_ARM_SIZE, _compute_outcome_grid


def _compute_joint_cdf(
    instr_arm_data: pl.DataFrame,
    outcome_grid: pl.Series,
    *,
    outcome: ColumnName,
    treatment: ColumnName,
    weights: ColumnName,
) -> NDArray[np.float64]:
    # The cdf is going to be the empirical cdf at the outcome grid plus the cdf value at minus
    # infinity (which is 0). This is necessary to be able to compute the empirical probability of
    # all intervals which includes [-inf,y] where y is some point on the grid. Eventually, we
    # return one cdf by treatment status.
    cdf = np.zeros((outcome_grid.len() + 1, 2), dtype=float)
    total = instr_arm_data[weights].sum()
    for treatment_value in (0, 1):
        data_by_trt_status = instr_arm_data.filter(pl.col(treatment) == treatment_value).sort(
            outcome
        )
        if data_by_trt_status.height == 0 or total <= 0:
            continue
        cumulative = np.concatenate(([0.0], data_by_trt_status[weights].cum_sum().to_numpy()))
        positions = data_by_trt_status[outcome].search_sorted(outcome_grid, side="right").to_numpy()
        cdf[1:, treatment_value] = cumulative[positions] / total
    return cdf


def _compute_interval_probabilities(cdf: NDArray[np.float64]) -> NDArray[np.float64]:
    """Interval masses for every pair of endpoints, ``p[i, j] = cdf[j] - cdf[i]``."""
    return cdf[None, :] - cdf[:, None]


def _compute_standardized_violations(
    cdf_given_instr_off: NDArray[np.float64],
    cdf_given_instr_on: NDArray[np.float64],
    effective_sample_size_instr_off: float,
    effective_sample_size_instr_on: float,
    *,
    weighting: ValidityWeighting,
    se_floor: float,
) -> NDArray[np.float64]:
    # Here we compute the (variance-weighted) violation for each interval on the outcome grid by
    # treatment value.
    total_effective_sample_size = effective_sample_size_instr_off + effective_sample_size_instr_on
    pair_scale = np.sqrt(
        effective_sample_size_instr_off
        * effective_sample_size_instr_on
        / total_effective_sample_size
    )
    share_instr_on = effective_sample_size_instr_on / total_effective_sample_size
    n_interval_endpoints = cdf_given_instr_off.shape[0]

    # The violation matrix holds the violations for each interval on the grid and each treatment
    # value. The cdf runs over the ordered outcome grid, so only the upper triangle is a genuine
    # interval: there the lower endpoint is at most the upper one (y <= y'), which is the mass in
    # (lower, upper]. The lower triangle just negates the mirrored interval, but it is computed
    # here anyway so that both halves come from one broadcast.
    violations = np.empty((n_interval_endpoints, n_interval_endpoints, 2), dtype=float)

    for treatment_value, direction in ((1, 1.0), (0, -1.0)):
        probs_instr_off = _compute_interval_probabilities(cdf_given_instr_off[:, treatment_value])
        probs_instr_on = _compute_interval_probabilities(cdf_given_instr_on[:, treatment_value])
        violation = pair_scale * direction * (probs_instr_off - probs_instr_on)
        if weighting is ValidityWeighting.VARIANCE:
            variance = (1.0 - share_instr_on) * probs_instr_on * (1.0 - probs_instr_on)
            variance += share_instr_on * probs_instr_off * (1.0 - probs_instr_off)
            violation = violation / np.maximum(np.sqrt(np.maximum(variance, 0.0)), se_floor)
        violations[:, :, treatment_value] = violation
    return violations


def _get_statistic_from_violations(
    violations: NDArray[np.float64], grid: pl.Series
) -> tuple[float, dict[str, Any] | None]:
    # We pass in the standardised violations for each interval and each treatment value and keep
    # only the genuine intervals, where the lower endpoint is at most the upper one (y <= y').
    # Those sit in the upper triangle; `positive_intervals` is the boolean mask for them.
    positive_intervals = np.triu(np.ones(violations.shape[:2], dtype=bool))
    statistic = 0.0
    place_of_violation = None  # Reported as the binding interval.
    for treatment_value in (1, 0):  # ties resolve towards the treated inequality
        # Get the maximum violation in the upper triangle.
        candidate_violations = np.where(
            positive_intervals, violations[:, :, treatment_value], -np.inf
        )
        largest_violation = float(candidate_violations.max())
        if largest_violation > statistic:
            lower, upper = (
                int(index)
                for index in np.unravel_index(
                    int(np.argmax(candidate_violations)), positive_intervals.shape
                )
            )
            statistic = largest_violation
            place_of_violation = {
                # Key names are shared with `InstrumentValidityResult.binding`.
                "z_low": 0,
                "z_high": 1,
                "direction": f"d = {treatment_value}",
                "y_lower": float("-inf") if lower == 0 else float(grid[lower - 1]),
                "y_upper": float(grid[upper - 1]),
            }
    return statistic, place_of_violation


def _compute_take_up(data: pl.DataFrame, treatment: ColumnName, weights: ColumnName) -> float:
    """Weighted treatment rate of one instrument arm."""
    return float((data[treatment] * data[weights]).sum()) / float(data[weights].sum())


def _compute_test_statistic(
    ds_instr_off: pl.DataFrame,
    ds_instr_on: pl.DataFrame,
    *,
    outcome: ColumnName,
    treatment: ColumnName,
    grid: pl.Series,
    n_off: float,
    n_on: float,
    weights: ColumnName,
    weighting: ValidityWeighting,
    se_floor: float,
) -> tuple[float, dict[str, Any] | None]:
    # To compute the test_statistic we compute the cdf for each instrument, treatment value combination.
    # Then we compute the standardized violations of the iv validity restriction from Kitagawa.
    # These violations are computed for each interval on the grid with the help of the cdfs.
    # Lastly, we get the largest violation which is our test statistic.
    cdf_arguments = {"outcome": outcome, "treatment": treatment, "weights": weights}
    violations = _compute_standardized_violations(
        _compute_joint_cdf(ds_instr_off, grid, **cdf_arguments),
        _compute_joint_cdf(ds_instr_on, grid, **cdf_arguments),
        n_off,
        n_on,
        weighting=weighting,
        se_floor=se_floor,
    )
    return _get_statistic_from_violations(violations, grid)


def _compute_sampled_statistic(
    sampled: pl.DataFrame,
    *,
    outcome: ColumnName,
    treatment: ColumnName,
    grid: pl.Series,
    n_on_rows: int,
    weights: ColumnName,
    weighting: ValidityWeighting,
    se_floor: float,
) -> float:
    """``T_N`` for one pooled bootstrap draw (Kitagawa, 2015, Algorithm 2.1).

    ``sampled`` is a with-replacement resample of the pooled ``(Y, D)`` data; its
    first ``n_on_rows`` rows play the role of ``P*_m`` and the remaining rows the
    role of ``Q*_n``. Disjoint positions of independent draws are themselves
    independent samples, so splitting one resample is the paper's two-sample draw
    from ``H_N``. Both CDFs, the effective sizes and the plug-in ``sigma`` are
    rebuilt from the drawn rows, so this is the same functional as the observed
    statistic rather than a rescaling of it.
    """
    n_off_rows = sampled.height - n_on_rows
    sampled_on = sampled.head(n_on_rows)
    sampled_off = sampled.tail(n_off_rows)
    statistic, _ = _compute_test_statistic(
        sampled_off,
        sampled_on,
        outcome=outcome,
        treatment=treatment,
        grid=grid,
        n_off=float(sampled_off[weights].sum()),
        n_on=float(sampled_on[weights].sum()),
        weights=weights,
        weighting=weighting,
        se_floor=se_floor,
    )
    return statistic


def perform_instrument_validity_test(
    ds: pl.DataFrame,
    outcome: ColumnName,
    treatment: ColumnName,
    instrument: ColumnName,
    *,
    weights: ColumnName | None = None,
    weighting: ValidityWeighting = ValidityWeighting.VARIANCE,
    se_floor: float = 0.15,
    outcome_grid_size: int = 50,
    alpha: float = 0.05,
    bootstrap_config: BootstrapConfig | int = 1000,
) -> InstrumentValidityResult:
    """Test instrument validity with Kitagawa (2015) and her pooled bootstrap.

    Only binary treatments and binary instruments are covered, and the test
    concerns validity, not relevance.
    Critical values come from Algorithm 2.1 of the paper, which resamples both
    instrument arms from the pooled empirical distribution of ``(Y, D)``. Since
    the pooled bootstrap is what the paper prescribes, this function does not
    take a multiplier distribution.

    Parameters
    ----------
    ds : polars.DataFrame
        Input data.
    outcome : ColumnName
        Name of the outcome column.
    treatment : ColumnName
        Name of the binary treatment column (``1`` treated, ``0`` not).
    instrument : ColumnName
        Name of the binary instrument column, coded ``1`` for the encouraged
        ("on") arm and ``0`` for the unencouraged ("off") arm. The levels are
        taken as given; a negative ``first_stage`` means the coding is the wrong
        way round for the monotonicity direction ``D(on) >= D(off)``.
    weights : ColumnName, optional
        Name of a column with sampling weights; when omitted every observation
        gets unit weight. The weights define the
        empirical measure of the observed CDFs; the bootstrap resamples rows,
        as everywhere else in the package, and the drawn rows keep their
        weights.
    weighting : `ValidityWeighting`, default=`ValidityWeighting.VARIANCE`
        Variance-weighted or unweighted Kolmogorov--Smirnov statistic.
    se_floor : float, default=0.15
        Lower bound for the plug-in standard error in the variance-weighted
        statistic (``xi`` in the paper, which recommends 0.05 to 0.1). The
        larger default is more stable in small or unbalanced instrument arms.
    outcome_grid_size : int, default=50
        Maximum number of outcome grid points used for the interval search.
    alpha : float, default=0.05
        Significance level used for the verdict.
    bootstrap_config : `BootstrapConfig` or int, default=1000
        Bootstrap settings, or the number of replications.

    Returns
    -------
    `InstrumentValidityResult`
        Test statistic, bootstrap p value and the binding interval.

    See Also
    --------
    `qte.iv.estimate_unconditional_local_effects`

    Notes
    -----
    The instrument is not reordered: the one-sided inequalities are stated for
    ``instrument == 1`` as the encouraged arm, so the caller is responsible for
    the coding. ``first_stage`` is reported as the on-minus-off take-up gap and
    is negative when the coding runs against monotonicity.

    Kitagawa (2015) defines the test for unweighted empirical measures; with unit
    weights this implementation is exactly her Algorithm 2.1. With sampling
    weights it applies the same statistic to the weighted measures and bootstraps
    it by resampling rows, which is not the paper's algorithm (it has none for
    this case) nor the pseudo-population bootstrap.

    The bootstrap re-draws both arms from the pooled sample, so the bootstrap
    world has ``P = Q`` by construction; Kitagawa (2015, footnote 11) notes that
    this is why no recentering is needed. The null distribution is therefore
    evaluated as if every inequality were binding, which makes the test
    conservative: a p value near one is uninformative about validity, and only
    a small p value is evidence against it. The paper suggests improving power
    by restricting the search to a preestimated contact set (footnote 10), which
    this implementation does not do.

    References
    ----------
    Kitagawa, T. (2015). A Test for Instrument Validity. *Econometrica*,
    83(5), 2043--2063.

    Examples
    --------
    Test the college-proximity instrument on the Card data:

    >>> from qte.datasets import load_card
    >>> from qte.iv import perform_instrument_validity_test
    >>> res = perform_instrument_validity_test(
    ...     load_card(treatment_cutoff=16),
    ...     "lwage",
    ...     "college",
    ...     "nearc4",
    ...     bootstrap_config=50,
    ... )
    >>> res.verdict
    'Reject IV validity'
    """
    if outcome_grid_size < 2:
        raise ValueError("'outcome_grid_size' must be at least 2.")
    if se_floor <= 0.0:
        raise ValueError("'se_floor' must be positive.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("'alpha' must lie between 0 and 1.")
    weighting = ValidityWeighting(weighting)

    if weights is None:
        weights = "_w"
        ds = ds.with_columns(pl.lit(1.0).alias(weights))
    data = ds.select(outcome, treatment, instrument, weights).drop_nulls()
    # Mean-one weights keep the effective sizes on the scale of row counts, so the
    # statistic does not depend on the units of the weights.
    data = data.with_columns((pl.col(weights) / pl.col(weights).mean()).alias(weights))
    ds_instr_on, ds_instr_off = (
        data.filter(pl.col(instrument) == 1),
        data.filter(pl.col(instrument) == 0),
    )
    first_stage = _compute_take_up(ds_instr_on, treatment, weights) - _compute_take_up(
        ds_instr_off, treatment, weights
    )
    effective_sample_size_instr_off = float(ds_instr_off[weights].sum())
    effective_sample_size_instr_on = float(ds_instr_on[weights].sum())

    outcome_grid = _compute_outcome_grid(data[outcome], outcome_grid_size)
    statistic, binding = _compute_test_statistic(
        ds_instr_off,
        ds_instr_on,
        outcome=outcome,
        treatment=treatment,
        grid=outcome_grid,
        n_off=effective_sample_size_instr_off,
        n_on=effective_sample_size_instr_on,
        weights=weights,
        weighting=weighting,
        se_floor=se_floor,
    )

    config = _make_bootstrap_config(bootstrap_config)
    if config.n_iter < 1:
        raise ValueError("The bootstrap needs at least one replication.")
    fcn = partial(
        _compute_sampled_statistic,
        outcome=outcome,
        treatment=treatment,
        grid=outcome_grid,
        n_on_rows=max(_MIN_ARM_SIZE, ds_instr_on.height),
        weights=weights,
        weighting=weighting,
        se_floor=se_floor,
    )
    pooled = data.select([outcome, treatment, weights])
    boot_stats = pl.Series(
        "boot_stats",
        _make_bootstrap_runs(
            pooled,
            fcn,
            n_iter=config.n_iter,
            seed=config.seed,
            n_workers=config.n_workers,
        ),
        dtype=pl.Float64,
    )

    return InstrumentValidityResult(
        statistic=float(statistic),
        p_value=float(np.mean(boot_stats.to_numpy() >= statistic)),
        alpha=alpha,
        n_boot=config.n_iter,
        boot_stats=boot_stats.to_numpy(),
        binding=binding,
        n=data.height,
        first_stage=first_stage,
        outcome=outcome,
        treatment=treatment,
        instrument=instrument,
        weighting=weighting,
        se_floor=se_floor,
        y_grid_size=outcome_grid_size,
    )
