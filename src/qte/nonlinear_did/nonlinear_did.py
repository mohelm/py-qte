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
from qte.names import EFFECT_ID, QUANTILE_ID, SE_ID
from qte.nonlinear_did.aggregate import (
    aggregate_group_time_effects,
    aggregate_group_time_effects_again_by_group,
    get_group_time_treatment_effects,
    get_weights_for_event_study_effects,
    get_weights_for_overall_effect,
    get_weights_for_treatment_group_effects,
)
from qte.nonlinear_did.custom_types import (
    BasePeriod,
    ColumnName,
    ControlGroup,
    CounterfactualModel,
    SamplingScheme,
    TrtGroupConfig,
    _NonlinearDidAggregation,
    _NonlinearDidAggregations,
)
from qte.nonlinear_did.results import (
    GroupTimeEffect,
    NonlinearDidResult,
    NonlinearDidResults,
)
from qte.stats import Ecdf, get_quantiles


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
    boot_iter: Iterable[dict[str, _NonlinearDidAggregation]],
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


def _make_nonlinear_did_results(
    agg: _NonlinearDidAggregation,
    bs_res: Estimates,
    *,
    outcome: ColumnName,
    base_period: BasePeriod,
    control_group: ControlGroup,
    counterfactual_model: CounterfactualModel,
) -> NonlinearDidResult:
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
    return NonlinearDidResult(
        qtt,
        att,
        group=agg.group,
        outcome=outcome,
        base_period=base_period,
        control_group=control_group,
        sampling_scheme=SamplingScheme.PANEL,
        counterfactual_model=counterfactual_model,
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


def _compute_kcic(pre_trt: NDArray, pre_ctrl: NDArray, post_ctrl: NDArray) -> NDArray:
    # Compute the ecdf of (pre/ctrl)
    w_pre_ctrl_norm = pre_ctrl["w"] / pre_ctrl["w"].sum()
    cdf_pre_ctrl = np.concatenate(([0.0], np.cumsum(w_pre_ctrl_norm)))

    # Now find where the (pre/trt) values are on this cdf. That is, essentially evaluate the
    # (pre/trt) values at the (pre/ctrl) cdf.
    idx_u = np.searchsorted(pre_ctrl["o"], pre_trt["o"], side="right")
    u = cdf_pre_ctrl[idx_u]

    # Get the (post/ctrl) cdf
    w_post_ctrl_norm = post_ctrl["w"] / post_ctrl["w"].sum()
    cdf_post_ctrl = np.cumsum(w_post_ctrl_norm)

    # Now check where each u (which is the (pre/trt) values of the (pre/ctrl) cdf) fall on the
    # (post/ctrl) cdf (at what index).
    idx_k = np.searchsorted(cdf_post_ctrl, u, side="left")
    idx_k = np.clip(idx_k, 0, len(post_ctrl["o"]) - 1)
    return post_ctrl["o"][idx_k]


def _compute_kqdid(pre_trt: NDArray, pre_ctrl: NDArray, post_ctrl: NDArray) -> NDArray:
    # QDiD shifts every treated pre-treatment observation by the control group's quantile change at
    # that observation's rank *within the treated pre-treatment distribution* (unlike CiC, which
    # ranks within the control's pre-treatment distribution).
    y = pre_trt["o"]
    _, inv = np.unique(y, return_inverse=True)
    w_unique = np.bincount(inv, weights=pre_trt["w"])
    u = np.cumsum(w_unique)[inv] / w_unique.sum()
    u = np.clip(u, 0.0, 1.0)

    q_pre_ctrl = get_quantiles(u, pre_ctrl["o"], pre_ctrl["w"])
    q_post_ctrl = get_quantiles(u, post_ctrl["o"], post_ctrl["w"])

    return y + q_post_ctrl - q_pre_ctrl


def _compute_group_time_effect(
    two_by_two_data: pl.DataFrame,
    g: int,
    tp: int,
    outcome: ColumnName,
    unit: ColumnName,
    weights: ColumnName,
    counterfactual_model: CounterfactualModel,
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

    kcf = (
        _compute_kcic(pre_trt, pre_ctrl, post_ctrl)
        if counterfactual_model == CounterfactualModel.CIC
        else _compute_kqdid(pre_trt, pre_ctrl, post_ctrl)
    )

    ecdf_post_treated_observed = Ecdf.make(post_trt["o"], post_trt["w"])
    ecdf_post_treated_counterfact = Ecdf.make(kcf, pre_trt["w"])

    return GroupTimeEffect(
        group=g,
        tp=tp,
        ecdf_observed=ecdf_post_treated_observed,
        ecdf_counterfact=ecdf_post_treated_counterfact,
        mean_observed=np.average(post_trt["o"], weights=post_trt["w"]),
        mean_counterfact=np.average(kcf, weights=pre_trt["w"]),
        group_size_observed=post_trt["w"].sum(),
        group_size_counterfactual=pre_trt["w"].sum(),
    )


def _compute_nonlinear_did_for_panel(
    ds: pl.DataFrame,
    outcome: ColumnName,
    treatment_group: ColumnName,
    time: ColumnName,
    unit: ColumnName,
    qs: ArrayLike = MEDIAN,
    *,
    weights: ColumnName,
    base_period: BasePeriod = BasePeriod.UNIVERSAL,
    control_group: ControlGroup = ControlGroup.NEVER_TREATED,
    counterfactual_model: CounterfactualModel = CounterfactualModel.CIC,
    n_anticipation_periods: int = 0,
) -> _NonlinearDidAggregations:

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
            counterfactual_model=counterfactual_model,
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
    return _NonlinearDidAggregations(
        group=group_te, event_study=event_study_te, overall=agg_te, group_time=group_time_te
    )


def estimate_nonlinear_did_for_panel(
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
    counterfactual_model: CounterfactualModel = CounterfactualModel.CIC,
    bootstrap_config: BootstrapConfig | int = 1000,
) -> NonlinearDidResults:
    """Estimate nonlinear difference-in-differences effects for panel data.

    Implements the changes-in-changes (CiC) and quantile difference-in-differences
    (QDiD) estimators with the aggregations of Callaway and Li. Returns the
    group, event-study, overall and group-time effects.

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
    counterfactual_model : `CounterfactualModel`, default=`CounterfactualModel.CIC`
        Model used to construct the counterfactual distribution.
    bootstrap_config : `BootstrapConfig` or int, default=1000
        Bootstrap settings, or the number of replications.

    Returns
    -------
    `NonlinearDidResults`
        The group, event-study, overall and group-time effects.
    """
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
        _compute_nonlinear_did_for_panel,
        outcome=outcome,
        treatment_group=treatment_group.name,
        time=time,
        unit=unit,
        qs=qs,
        weights=weights,
        base_period=base_period,
        control_group=control_group,
        counterfactual_model=counterfactual_model,
        n_anticipation_periods=n_anticipation_periods,
    )
    estimate: _NonlinearDidAggregations = fcn(ds)
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
        agg_name: _make_nonlinear_did_results(
            agg,
            bs_aggs[agg_name],
            outcome=outcome,
            base_period=base_period,
            control_group=control_group,
            counterfactual_model=counterfactual_model,
        )
        for agg_name, agg in estimate.items()
    }
    return NonlinearDidResults(**results)
