from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial
from itertools import product

import numpy as np
import polars as pl
import polars.selectors as cs
from numpy.typing import ArrayLike, NDArray

from qte.bootstrap import BootstrapConfig, _make_block_bootstrap_run, _make_bootstrap_config
from qte.constants import MEDIAN
from qte.distributional_did.aggregate import (
    aggregate_group_time_effects,
    aggregate_group_time_effects_again_by_group,
    get_group_time_treatment_effects,
    get_weights_for_event_study_effects,
    get_weights_for_overall_effect,
    get_weights_for_treatment_group_effects,
)
from qte.distributional_did.counterfactuals import _SPECS, _CounterfactualSpec
from qte.distributional_did.custom_types import (
    BasePeriod,
    ColumnName,
    ControlGroup,
    Estimator,
    SamplingScheme,
    TrtGroupConfig,
    _DistributionalDidAggregation,
    _DistributionalDidAggregations,
)
from qte.distributional_did.results import (
    DistributionalDidResult,
    DistributionalDidResults,
    GroupTimeEffect,
)
from qte.names import EFFECT_ID, QUANTILE_ID, SE_ID
from qte.stats import Ecdf


@dataclass(frozen=True)
class Estimates:
    atts: pl.DataFrame
    qtes: pl.DataFrame


def _make_columns_to_group_by(groups: str | tuple[str, ...] | None, other: str | None) -> list[str]:
    columns = list(groups) if groups is not None else []
    if other is not None:
        columns.append(other)
    return columns


def get_statistics_from_bootstrap(
    boot_iter: Iterable[dict[str, _DistributionalDidAggregation]],
    aggregations: Iterable[tuple[str, str | tuple[str, ...] | None]],
) -> dict[str, Estimates]:
    runs = list(boot_iter)
    agg = pl.col(EFFECT_ID).std().alias(SE_ID)

    return {
        aggregation: Estimates(
            qtes=pl.concat(r[aggregation].qtt.with_columns(boot_id=i) for i, r in enumerate(runs))
            .group_by(*_make_columns_to_group_by(grouper, QUANTILE_ID))
            .agg(agg),
            atts=pl.concat(r[aggregation].att.with_columns(boot_id=i) for i, r in enumerate(runs))
            .group_by(*_make_columns_to_group_by(grouper, None))
            .agg(agg),
        )
        for aggregation, grouper in aggregations
    }


def _make_results(
    agg: _DistributionalDidAggregation,
    bs_res: Estimates,
    *,
    outcome: ColumnName,
    base_period: BasePeriod,
    control_group: ControlGroup,
    estimator: Estimator,
) -> DistributionalDidResult:
    qtt = agg.qtt.join(
        bs_res.qtes,
        on=[QUANTILE_ID, *agg.group] if agg.group is not None else [QUANTILE_ID],
    )
    att = agg.att.join(
        bs_res.atts,
        on=agg.group if agg.group is not None else None,
        how="inner" if agg.group is not None else "cross",
    )
    if agg.group is not None:
        qtt = qtt.with_columns(pl.col(g).cast(pl.Int64) for g in agg.group)
        att = att.with_columns(pl.col(g).cast(pl.Int64) for g in agg.group)
    return DistributionalDidResult(
        qtt,
        att,
        group=agg.group,
        outcome=outcome,
        base_period=base_period,
        control_group=control_group,
        sampling_scheme=SamplingScheme.PANEL,
        estimator=estimator,
    )


def _make_reference_period(
    treated_group: int,
    time_period: int,
    n_anticipation_periods: int,
    base_period: BasePeriod,
) -> int:
    # In the post-treatment period we always compare to the earliest pre-treatment period
    # (accounting for anticipation).
    reference_period = treated_group - n_anticipation_periods - 1

    # However, in the pre-treatment period we might want to compare to a time period that is prior
    # to the pre-treatment period in question ("varying" time period)
    if (time_period < treated_group) & (base_period == BasePeriod.VARYING):
        reference_period = time_period - n_anticipation_periods - 1
    return reference_period


def _get_data_for_two_by_two(
    ds: pl.DataFrame,
    treated_group: int,
    time_period: int,
    reference_period: int,
    n_anticipation_periods: int,
    treatment_group: ColumnName,
    time: ColumnName,
    control_group: ControlGroup,
) -> pl.DataFrame:

    is_treated_g = pl.col(treatment_group) == treated_group
    is_control_g = pl.col(treatment_group).is_infinite()
    if control_group == ControlGroup.NOT_YET_TREATED:
        is_control_g = is_control_g | (
            pl.col(treatment_group) > time_period + n_anticipation_periods
        )

    return ds.filter(
        (is_treated_g | is_control_g) & pl.col(time).is_in((time_period, reference_period))
    ).with_columns(
        _is_treated=pl.col(treatment_group) == treated_group,
        _is_post=pl.col(time) == time_period,
    )


def _get_group_data(
    ds: pl.DataFrame, filter_: pl.Expr, outcome: ColumnName, weights: ColumnName, unit: ColumnName
) -> NDArray:
    return (
        ds.filter(filter_)
        .select(
            outcome,
            weights,
            unit,
        )
        .sort(outcome)
        .rename({outcome: "o", weights: "w", unit: "u"})
        .to_numpy(structured=True)
    )


def _compute_group_time_effect(
    two_by_two_data: pl.DataFrame,
    g: int,
    tp: int,
    outcome: ColumnName,
    unit: ColumnName,
    weights: ColumnName,
    spec: _CounterfactualSpec,
) -> GroupTimeEffect:
    _group_time_extractor = partial(
        _get_group_data,
        two_by_two_data,
        outcome=outcome,
        weights=weights,
        unit=unit,
    )

    post_trt = _group_time_extractor(pl.col("_is_treated") & pl.col("_is_post"))
    pre_trt = _group_time_extractor(pl.col("_is_treated") & (~pl.col("_is_post")))
    pre_ctrl = _group_time_extractor((~pl.col("_is_treated")) & (~pl.col("_is_post")))
    post_ctrl = _group_time_extractor((~pl.col("_is_treated")) & pl.col("_is_post"))

    cf = spec.compute(pre_trt, pre_ctrl, post_ctrl)

    return GroupTimeEffect(
        group=g,
        tp=tp,
        ecdf_observed=Ecdf.make(post_trt["o"], post_trt["w"]),
        ecdf_counterfact=Ecdf.make(cf.y_counterfactual, cf.weights),
        mean_observed=np.average(post_trt["o"], weights=post_trt["w"]),
        mean_counterfact=np.average(cf.y_counterfactual, weights=cf.weights),
        group_size_observed=post_trt["w"].sum(),
        group_size_counterfactual=cf.group_size,
    )


def _compute_effects_for_panel(
    ds: pl.DataFrame,
    outcome: ColumnName,
    treatment_group: ColumnName,
    time: ColumnName,
    unit: ColumnName,
    qs: ArrayLike = MEDIAN,
    *,
    weights: ColumnName,
    spec: _CounterfactualSpec,
    base_period: BasePeriod = BasePeriod.UNIVERSAL,
    control_group: ControlGroup = ControlGroup.NEVER_TREATED,
    n_anticipation_periods: int = 0,
) -> _DistributionalDidAggregations:

    outcome_grid_size = 1000

    post_trt_ds = ds.filter(
        pl.col(treatment_group).is_finite() & (pl.col(time) >= pl.col(treatment_group))
    )
    outcome_values = post_trt_ds[outcome].unique().to_numpy()
    y_grid = np.linspace(outcome_values.min(), outcome_values.max(), outcome_grid_size)

    # Compute group time effects
    time_periods = ds[time].unique().sort()
    treated_groups = ds[treatment_group].unique().sort()[:-1]  # TODO: FIX

    group_time_effects = [
        _get_data_for_two_by_two(
            ds,
            g,
            tp,
            rp,
            n_anticipation_periods,
            treatment_group,
            time,
            control_group,
        ).pipe(
            _compute_group_time_effect,
            g,
            tp,
            outcome=outcome,
            unit=unit,
            weights=weights,
            spec=spec,
        )
        for tp, g in product(time_periods, treated_groups)
        if (
            (rp := _make_reference_period(g, tp, n_anticipation_periods, base_period))
            in time_periods
            and not (tp == rp and base_period == BasePeriod.UNIVERSAL)
        )
    ]
    group_sizes_per_time = ds.group_by(
        pl.col(treatment_group).alias("group"), pl.col(time).alias("time_period")
    ).agg(weight=pl.col(weights).sum())
    post_trt_group_time_effects = [gte for gte in group_time_effects if gte.tp >= gte.group]

    # AGGREGATE
    # Group Time Effects
    group_time_te = get_group_time_treatment_effects(qs, group_time_effects)

    # Group
    group_te = aggregate_group_time_effects_again_by_group(
        qs,
        post_trt_group_time_effects,
        get_weights_for_treatment_group_effects(group_sizes_per_time),
        y_grid,
        dim_id=lambda gte: gte.group,
        dim_name="treatment_group",
    )
    # Overall
    agg_te = aggregate_group_time_effects(
        qs,
        post_trt_group_time_effects,
        get_weights_for_overall_effect(group_sizes_per_time),
        y_grid,
    )

    # Event study
    event_study_te = aggregate_group_time_effects_again_by_group(
        qs,
        group_time_effects,
        get_weights_for_event_study_effects(group_sizes_per_time, base_period=base_period),
        y_grid,
        dim_id=lambda gte: gte.tp - gte.group,
        dim_name="event_study_period",
    )
    return _DistributionalDidAggregations(
        group=group_te, event_study=event_study_te, overall=agg_te, group_time=group_time_te
    )


def _estimate_effects(
    ds: pl.DataFrame,
    outcome: ColumnName,
    treatment_group: ColumnName | TrtGroupConfig,
    time: ColumnName,
    unit: ColumnName,
    qs: ArrayLike,
    *,
    spec: _CounterfactualSpec,
    weights: ColumnName | None = None,
    n_anticipation_periods: int = 0,
    base_period: BasePeriod = BasePeriod.UNIVERSAL,
    control_group: ControlGroup = ControlGroup.NEVER_TREATED,
    bootstrap_config: BootstrapConfig | int = 1000,
) -> DistributionalDidResults:
    bootstrap_config = _make_bootstrap_config(bootstrap_config)
    if isinstance(treatment_group, str):
        treatment_group = TrtGroupConfig(name=treatment_group)
    ds = ds.with_columns(
        cs.by_name(
            treatment_group.name,
            time,
        ).cast(pl.Float64)
    )
    if treatment_group.never_treated_identifier != float("inf"):
        ds = ds.with_columns(
            pl.when(pl.col(treatment_group.name) == treatment_group.never_treated_identifier)
            .then(float("inf"))
            .otherwise(pl.col(treatment_group.name))
            .alias(treatment_group.name)
        )

    if weights is None:
        weights = "_w"
        ds = ds.with_columns(pl.lit(1).alias(weights))
    all_groups = ds[treatment_group.name].unique()
    all_treated_groups = all_groups.filter(all_groups.is_finite())
    time_periods = ds[time].unique().sort().to_list()

    treated_groups: pl.Series = all_treated_groups.filter(
        all_treated_groups >= min(time_periods) + 1 + n_anticipation_periods
    )
    ds = ds.filter(
        pl.col(treatment_group.name).is_in(set(treated_groups.to_list()))
        | pl.col(treatment_group.name).is_infinite()
    )
    fcn = partial(
        _compute_effects_for_panel,
        outcome=outcome,
        treatment_group=treatment_group.name,
        time=time,
        unit=unit,
        qs=qs,
        weights=weights,
        spec=spec,
        base_period=base_period,
        control_group=control_group,
        n_anticipation_periods=n_anticipation_periods,
    )
    estimate: _DistributionalDidAggregations = fcn(ds)
    bs_iterations = _make_block_bootstrap_run(
        ds,
        fcn,
        unit,
        n_iter=bootstrap_config.n_iter,
        seed=bootstrap_config.seed,
        n_workers=bootstrap_config.n_workers,
    )
    # We must explicitly type cast iteration items to silence ty
    groupers: list[tuple[str, tuple[str, ...] | None]] = [
        (agg_name, agg.group) for agg_name, agg in estimate.items()
    ]
    bs_aggs = get_statistics_from_bootstrap(bs_iterations, groupers)

    results = {
        agg_name: _make_results(
            agg,
            bs_aggs[agg_name],
            outcome=outcome,
            base_period=base_period,
            control_group=control_group,
            estimator=Estimator(spec.key),
        )
        for agg_name, agg in estimate.items()
    }
    return DistributionalDidResults(**results)


def estimate_changes_in_changes_effects(
    ds: pl.DataFrame,
    outcome: ColumnName,
    treatment_group: ColumnName | TrtGroupConfig,
    time: ColumnName,
    unit: ColumnName,
    qs: ArrayLike = MEDIAN,
    *,
    weights: ColumnName | None = None,
    n_anticipation_periods: int = 0,
    base_period: BasePeriod = BasePeriod.UNIVERSAL,
    control_group: ControlGroup = ControlGroup.NEVER_TREATED,
    bootstrap_config: BootstrapConfig | int = 1000,
) -> DistributionalDidResults:
    """Estimate quantile treatment effects with changes-in-changes.

    Changes-in-changes compares individuals across groups by their outcomes and over time by
    their quantiles. It
    requires a time-invariant, strictly increasing production function mapping unobservables to
    outcomes, and thus rank invariance; the distribution of unobservables may differ between the
    treated and untreated groups. Returns the group, event-study, overall and group-time effects.

    Parameters
    ----------
    ds : polars.DataFrame
        Panel data in long format.
    outcome : ColumnName
        Name of the outcome column.
    treatment_group : ColumnName or `TrtGroupConfig`
        Column holding the first treatment period of a unit. A plain string
        treats the column as already coded with ``inf`` for never-treated units;
        a `TrtGroupConfig` maps a custom never-treated value to ``inf``.
    time : ColumnName
        Name of the time period column.
    unit : ColumnName
        Name of the unit identifier column.
    qs : array_like, default=0.5
        Quantiles in ``(0, 1)`` at which to estimate the effects.
    weights : ColumnName, optional
        Name of a column with sampling weights.
    n_anticipation_periods : int, default=0
        Number of periods before treatment in which units may anticipate it.
    base_period : `BasePeriod`, default=`BasePeriod.UNIVERSAL`
        Pre-treatment period used for the comparisons.
    control_group : `ControlGroup`, default=`ControlGroup.NEVER_TREATED`
        Comparison group used for the counterfactual.
    bootstrap_config : `BootstrapConfig` or int, default=1000
        Bootstrap settings, or the number of replications.

    Returns
    -------
    `DistributionalDidResults`
        The group, event-study, overall and group-time effects.

    See Also
    --------
    estimate_quantile_did_effects
        Compares individuals across both groups and time by their quantiles.

    References
    ----------
    Athey, Susan and Guido Imbens. "Identification and Inference in Nonlinear
    Difference-in-Differences Models." Econometrica 74(2), pp. 431-497, 2006.
    """
    return _estimate_effects(
        ds,
        outcome,
        treatment_group,
        time,
        unit,
        qs,
        spec=_SPECS["changes_in_changes"],
        weights=weights,
        n_anticipation_periods=n_anticipation_periods,
        base_period=base_period,
        control_group=control_group,
        bootstrap_config=bootstrap_config,
    )


def estimate_quantile_did_effects(
    ds: pl.DataFrame,
    outcome: ColumnName,
    treatment_group: ColumnName | TrtGroupConfig,
    time: ColumnName,
    unit: ColumnName,
    qs: ArrayLike = MEDIAN,
    *,
    weights: ColumnName | None = None,
    n_anticipation_periods: int = 0,
    base_period: BasePeriod = BasePeriod.UNIVERSAL,
    control_group: ControlGroup = ControlGroup.NEVER_TREATED,
    bootstrap_config: BootstrapConfig | int = 1000,
) -> DistributionalDidResults:
    """Estimate quantile treatment effects with quantile difference-in-differences.

    Quantile difference-in-differences applies difference-in-differences to every quantile instead
    of to the mean: each treated pre-treatment observation is shifted by the change over time of
    the control group at the same quantile. It is identified by an additively separable model whose unobservables must be
    identically distributed in all groups and periods, which rules out the treated and untreated
    groups differing in the distribution of unobservables. Under continuity, the implied average
    treatment effect equals that of the mean difference-in-differences estimator, even though the
    counterfactual distributions differ. Returns the group, event-study, overall and group-time
    effects.

    Parameters
    ----------
    ds : polars.DataFrame
        Panel data in long format.
    outcome : ColumnName
        Name of the outcome column.
    treatment_group : ColumnName or `TrtGroupConfig`
        Column holding the first treatment period of a unit. A plain string
        treats the column as already coded with ``inf`` for never-treated units;
        a `TrtGroupConfig` maps a custom never-treated value to ``inf``.
    time : ColumnName
        Name of the time period column.
    unit : ColumnName
        Name of the unit identifier column.
    qs : array_like, default=0.5
        Quantiles in ``(0, 1)`` at which to estimate the effects.
    weights : ColumnName, optional
        Name of a column with sampling weights.
    n_anticipation_periods : int, default=0
        Number of periods before treatment in which units may anticipate it.
    base_period : `BasePeriod`, default=`BasePeriod.UNIVERSAL`
        Pre-treatment period used for the comparisons.
    control_group : `ControlGroup`, default=`ControlGroup.NEVER_TREATED`
        Comparison group used for the counterfactual.
    bootstrap_config : `BootstrapConfig` or int, default=1000
        Bootstrap settings, or the number of replications.

    Returns
    -------
    `DistributionalDidResults`
        The group, event-study, overall and group-time effects.

    See Also
    --------
    estimate_changes_in_changes_effects
        Does not require the unobservables to be identically distributed across groups.

    References
    ----------
    Athey, Susan and Guido Imbens. "Identification and Inference in Nonlinear
    Difference-in-Differences Models." Econometrica 74(2), pp. 431-497, 2006, Section 3.3.
    """
    return _estimate_effects(
        ds,
        outcome,
        treatment_group,
        time,
        unit,
        qs,
        spec=_SPECS["quantile_did"],
        weights=weights,
        n_anticipation_periods=n_anticipation_periods,
        base_period=base_period,
        control_group=control_group,
        bootstrap_config=bootstrap_config,
    )


def estimate_mean_did_effects(
    ds: pl.DataFrame,
    outcome: ColumnName,
    treatment_group: ColumnName | TrtGroupConfig,
    time: ColumnName,
    unit: ColumnName,
    qs: ArrayLike = MEDIAN,
    *,
    weights: ColumnName | None = None,
    n_anticipation_periods: int = 0,
    base_period: BasePeriod = BasePeriod.UNIVERSAL,
    control_group: ControlGroup = ControlGroup.NEVER_TREATED,
    bootstrap_config: BootstrapConfig | int = 1000,
) -> DistributionalDidResults:
    """Estimate treatment effects with mean difference-in-differences.

    The counterfactual distribution of untreated potential outcomes for the treated group is the
    treated group's pre-treatment distribution shifted by the change in the control group's mean.
    This implies the usual parallel trends assumption, so the average treatment effect on the
    treated is identified, but it is stronger: it also requires that only the location of the
    distribution shifts, not its shape. The reported quantile treatment effects are therefore not
    identified by the assumption. They are the difference between the observed treated quantiles
    and the shifted pre-treatment quantiles, ``Q(Y | treated, post) - Q(Y | treated, pre) - delta``,
    which varies with the quantile only because the treated group's own distribution changes.
    Reading a slope into them as evidence of treatment effect heterogeneity is a mistake. Returns
    the group, event-study, overall and group-time effects.

    Parameters
    ----------
    ds : polars.DataFrame
        Panel data in long format.
    outcome : ColumnName
        Name of the outcome column.
    treatment_group : ColumnName or `TrtGroupConfig`
        Column holding the first treatment period of a unit. A plain string
        treats the column as already coded with ``inf`` for never-treated units;
        a `TrtGroupConfig` maps a custom never-treated value to ``inf``.
    time : ColumnName
        Name of the time period column.
    unit : ColumnName
        Name of the unit identifier column.
    qs : array_like, default=0.5
        Quantiles in ``(0, 1)`` at which to estimate the effects.
    weights : ColumnName, optional
        Name of a column with sampling weights.
    n_anticipation_periods : int, default=0
        Number of periods before treatment in which units may anticipate it.
    base_period : `BasePeriod`, default=`BasePeriod.UNIVERSAL`
        Pre-treatment period used for the comparisons.
    control_group : `ControlGroup`, default=`ControlGroup.NEVER_TREATED`
        Comparison group used for the counterfactual.
    bootstrap_config : `BootstrapConfig` or int, default=1000
        Bootstrap settings, or the number of replications.

    Returns
    -------
    `DistributionalDidResults`
        The group, event-study, overall and group-time effects.

    See Also
    --------
    estimate_distributional_did_effects
        Relaxes the location-shift restriction by restricting the dependence between the change and
        the initial level instead, which identifies distributional effects.

    References
    ----------
    Thuysbaert, Bram. "Distributional Comparisons in Difference in Differences Models." Working
    paper, 2007 (as in ``qte::mdid``).

    Heckman, James, Hidehiko Ichimura and Petra Todd. "Matching as an Econometric Evaluation
    Estimator: Evidence from Evaluating a Job Training Programme." Review of Economic Studies
    64(4), pp. 605-654, 1997.

    Abadie, Alberto. "Semiparametric Difference-in-Differences Estimators." Review of Economic
    Studies 72(1), pp. 1-19, 2005.
    """
    return _estimate_effects(
        ds,
        outcome,
        treatment_group,
        time,
        unit,
        qs,
        spec=_SPECS["mean_did"],
        weights=weights,
        n_anticipation_periods=n_anticipation_periods,
        base_period=base_period,
        control_group=control_group,
        bootstrap_config=bootstrap_config,
    )


def estimate_distributional_did_effects(
    ds: pl.DataFrame,
    outcome: ColumnName,
    treatment_group: ColumnName | TrtGroupConfig,
    time: ColumnName,
    unit: ColumnName,
    qs: ArrayLike = MEDIAN,
    *,
    weights: ColumnName | None = None,
    n_anticipation_periods: int = 0,
    base_period: BasePeriod = BasePeriod.UNIVERSAL,
    control_group: ControlGroup = ControlGroup.NEVER_TREATED,
    bootstrap_config: BootstrapConfig | int = 1000,
) -> DistributionalDidResults:
    """Estimate quantile treatment effects with distributional difference-in-differences.

    Combines a distributional parallel trends assumption -- the distribution of the change in
    untreated potential outcomes is the same for the treated and untreated groups -- with the
    assumption that the dependence, that is the copula, between the change in untreated potential
    outcomes and the initial level of untreated potential outcomes is the same for the two groups.
    The initial distribution of outcomes is allowed to differ between the groups, which is what
    distinguishes this estimator from quantile difference-in-differences, where the unobservables
    must be identically distributed everywhere. The counterfactual pairs each control unit's own
    change over time with the treated outcome at that unit's rank, so it needs the same units in
    both periods of a cell, and it is indexed on the control units, which means the distribution of
    individual treatment effects is not identified. Returns the group, event-study, overall and
    group-time effects.

    Parameters
    ----------
    ds : polars.DataFrame
        Panel data in long format.
    outcome : ColumnName
        Name of the outcome column.
    treatment_group : ColumnName or `TrtGroupConfig`
        Column holding the first treatment period of a unit. A plain string
        treats the column as already coded with ``inf`` for never-treated units;
        a `TrtGroupConfig` maps a custom never-treated value to ``inf``.
    time : ColumnName
        Name of the time period column.
    unit : ColumnName
        Name of the unit identifier column.
    qs : array_like, default=0.5
        Quantiles in ``(0, 1)`` at which to estimate the effects.
    weights : ColumnName, optional
        Name of a column with sampling weights.
    n_anticipation_periods : int, default=0
        Number of periods before treatment in which units may anticipate it.
    base_period : `BasePeriod`, default=`BasePeriod.UNIVERSAL`
        Pre-treatment period used for the comparisons.
    control_group : `ControlGroup`, default=`ControlGroup.NEVER_TREATED`
        Comparison group used for the counterfactual.
    bootstrap_config : `BootstrapConfig` or int, default=1000
        Bootstrap settings, or the number of replications.

    Returns
    -------
    `DistributionalDidResults`
        The group, event-study, overall and group-time effects.

    See Also
    --------
    estimate_mean_did_effects
        A location-shift restriction that identifies only the average effect.
    estimate_quantile_did_effects
        Requires the unobservables to be identically distributed across groups.

    References
    ----------
    Callaway, Brantly, Tong Li and Tatsushi Oka. "Quantile Treatment Effects in Difference in
    Differences Models under Dependence Restrictions and with Only Two Time Periods." Journal of
    Econometrics 206(2), pp. 395-413, 2018.
    """
    return _estimate_effects(
        ds,
        outcome,
        treatment_group,
        time,
        unit,
        qs,
        spec=_SPECS["distributional_did"],
        weights=weights,
        n_anticipation_periods=n_anticipation_periods,
        base_period=base_period,
        control_group=control_group,
        bootstrap_config=bootstrap_config,
    )
