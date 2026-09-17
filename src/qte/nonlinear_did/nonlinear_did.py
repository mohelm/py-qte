from functools import partial
from itertools import product

import numpy as np
import polars as pl
import polars.selectors as cs
from numpy.typing import ArrayLike, NDArray

from qte.constants import MEDIAN
from qte.custom_types import ColumnName
from qte.names import QUANTILE_ID
from qte.nonlinear_did.aggregate import (
    aggregate_group_time_effects_again,
    aggregate_group_time_effects_again_by_group,
    get_weights_for_event_study_effects,
    get_weights_for_overall_effect,
    get_weights_for_treatment_group_effects,
)
from qte.nonlinear_did.bootstrap import (
    Estimates,
    get_statistics_from_bootstrap,
    perform_bootstrap,
)
from qte.nonlinear_did.custom_types import (
    BasePeriod,
    ControlGroup,
    CounterfactualModel,
    NonlinearDidAggregation,
    NonlinearDidAggregations,
    SamplingScheme,
    TrtGroupConfig,
)
from qte.nonlinear_did.results import (
    GroupTimeEffect,
    NonlinearDidResult,
    NonlinearDidResults,
)
from qte.stats import Ecdf, get_quantiles


def _make_nonlinear_did_results(
    agg: NonlinearDidAggregation,
    bs_res: Estimates,
    *,
    outcome: str,
    base_period: BasePeriod,
    control_group: ControlGroup,
    counterfactual_model: CounterfactualModel,
) -> NonlinearDidResult:
    qtt = agg.qtt.join(
        bs_res.qtes,
        on=[QUANTILE_ID, agg.group] if agg.group is not None else [QUANTILE_ID],
    )
    att = agg.att.join(
        bs_res.atts,
        on=[agg.group] if agg.group is not None else None,
        how="inner" if agg.group is not None else "cross",
    )
    if agg.group is not None:
        qtt = qtt.with_columns(pl.col(agg.group).cast(pl.Int64))
        att = att.with_columns(pl.col(agg.group).cast(pl.Int64))
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
    # In the post-treatment period we always compare to the earlist pre-treatment period
    # (accounting for anticipation).
    reference_period = treated_group - n_anticipation_periods - 1

    # However, in the pre-treatment period we might want to compare to a time period that is prior
    # to the per-treatment period in question ("varying" time period)
    if (time_period < treated_group) & (base_period == BasePeriod.VARYING):
        reference_period = time_period - n_anticipation_periods - 1
    return reference_period


def _get_data_for_two_by_two(
    ds: pl.DataFrame,
    treated_group: int,
    time_period: int,
    reference_period: int,
    n_anticipation_periods: int,
    treatment_group_c: str,
    time_c: str,
    control_group: ControlGroup,
) -> pl.DataFrame:

    is_treated_g = pl.col(treatment_group_c) == treated_group
    is_control_g = pl.col(treatment_group_c).is_infinite()
    if control_group == ControlGroup.NOT_YET_TREATED:
        is_control_g = is_control_g | (
            pl.col(treatment_group_c) > time_period + n_anticipation_periods
        )

    return ds.filter(
        (is_treated_g | is_control_g) & pl.col(time_c).is_in((time_period, reference_period))
    ).with_columns(
        _is_treated=pl.col(treatment_group_c) == treated_group,
        _is_post=pl.col(time_c) == time_period,
    )


def _get_group_data(
    ds: pl.DataFrame, filter_: pl.Expr, outcome_c: str, weights_c: str, unit_c: str
) -> NDArray:
    return (
        ds.filter(filter_)
        .select(
            outcome_c,
            weights_c,
            unit_c,
        )
        .sort(outcome_c)
        .rename({outcome_c: "o", weights_c: "w", unit_c: "u"})
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
    outcome_c: ColumnName,
    unit_c: ColumnName,
    weights_c: ColumnName,
    counterfactual_model: CounterfactualModel,
) -> GroupTimeEffect:
    _group_time_extractor = partial(
        _get_group_data,
        two_by_two_data,
        outcome_c=outcome_c,
        weights_c=weights_c,
        unit_c=unit_c,
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
        mean_countfact=np.average(kcf, weights=pre_trt["w"]),
        group_size_observed=post_trt["w"].sum(),
        group_size_counterfactual=pre_trt["w"].sum(),
    )


def _compute_nonlinear_did_for_panel(
    ds: pl.DataFrame,
    outcome_c: str,
    treatment_group_c: str,
    time_c: str,
    unit_c: str,
    qs: ArrayLike = MEDIAN,
    *,
    weights_c: str,
    base_period: BasePeriod = BasePeriod.UNIVERSAL,
    control_group: ControlGroup = ControlGroup.NEVER_TREATED,
    counterfactual_model: CounterfactualModel = CounterfactualModel.CIC,
    n_anticipation_periods: int = 0,
) -> NonlinearDidAggregations:

    outcome_grid_size = 1000

    post_trt_ds = ds.filter(
        pl.col(treatment_group_c).is_finite() & (pl.col(time_c) >= pl.col(treatment_group_c))
    )
    outcome = post_trt_ds[outcome_c].unique().to_numpy()
    y_grid = np.linspace(outcome.min(), outcome.max(), outcome_grid_size)

    # Compute group time effects
    time_periods = ds[time_c].unique().sort()
    treated_groups = ds[treatment_group_c].unique().sort()[:-1]  # TODO: FIX

    group_time_effects = [
        _get_data_for_two_by_two(
            ds,
            g,
            tp,
            rp,
            n_anticipation_periods,
            treatment_group_c,
            time_c,
            control_group,
        ).pipe(
            _compute_group_time_effect,
            g,
            tp,
            outcome_c=outcome_c,
            unit_c=unit_c,
            weights_c=weights_c,
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
        pl.col(treatment_group_c).alias("group"), pl.col(time_c).alias("time_period")
    ).agg(weight=pl.col(weights_c).sum())
    post_trt_group_time_effects = [gte for gte in group_time_effects if gte.tp >= gte.group]

    # AGGREGATE
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
    agg_te = aggregate_group_time_effects_again(
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
    return NonlinearDidAggregations(group=group_te, event_study=event_study_te, overall=agg_te)


def estimate_nonlinear_did_for_panel(
    ds: pl.DataFrame,
    outcome_c: ColumnName,
    treatment_group_c: ColumnName | TrtGroupConfig,
    time_c: ColumnName,
    unit_c: ColumnName,
    qs: ArrayLike = MEDIAN,
    *,
    weights_c: str | None = None,
    n_anticipation_periods: int = 0,
    base_period: BasePeriod = BasePeriod.UNIVERSAL,
    control_group: ControlGroup = ControlGroup.NEVER_TREATED,
    counterfactual_model: CounterfactualModel = CounterfactualModel.CIC,
    n_bootstrap_iter: int = 1000,
) -> NonlinearDidResults:
    if isinstance(treatment_group_c, str):
        treatment_group_c = TrtGroupConfig(name=treatment_group_c)
    ds = ds.with_columns(
        cs.by_name(
            treatment_group_c.name,
            time_c,
        ).cast(pl.Float64)
    )
    if treatment_group_c.never_treated_identifier != float("inf"):
        ds = ds.with_columns(
            pl.when(pl.col(treatment_group_c.name) == treatment_group_c.never_treated_identifier)
            .then(float("inf"))
            .otherwise(pl.col(treatment_group_c.name))
            .alias(treatment_group_c.name)
        )

    if weights_c is None:
        weights_c = "_w"
        ds = ds.with_columns(pl.lit(1).alias(weights_c))
    all_groups = ds[treatment_group_c.name].unique()
    all_treated_groups = all_groups.filter(all_groups.is_finite())
    time_periods = ds[time_c].unique().sort().to_list()

    treated_groups: pl.Series = all_treated_groups.filter(
        all_treated_groups >= min(time_periods) + 1 + n_anticipation_periods
    )
    ds = ds.filter(
        pl.col(treatment_group_c.name).is_in(set(treated_groups.to_list()))
        | pl.col(treatment_group_c.name).is_infinite()
    )
    fcn = partial(
        _compute_nonlinear_did_for_panel,
        outcome_c=outcome_c,
        treatment_group_c=treatment_group_c.name,
        time_c=time_c,
        unit_c=unit_c,
        qs=qs,
        weights_c=weights_c,
        base_period=base_period,
        control_group=control_group,
        counterfactual_model=counterfactual_model,
        n_anticipation_periods=n_anticipation_periods,
    )
    estimate: NonlinearDidAggregations = fcn(ds)
    bs_iterations = perform_bootstrap(ds, fcn, unit_c, n_iter=n_bootstrap_iter)
    # We must explicitly type cast iteration items to silence ty
    groupers: list[tuple[str, str | None]] = [
        (agg_name, agg.group) for agg_name, agg in estimate.items()
    ]
    bs_aggs = get_statistics_from_bootstrap(bs_iterations, groupers)

    results = {
        agg_name: _make_nonlinear_did_results(
            agg,
            bs_aggs[agg_name],
            outcome=outcome_c,
            base_period=base_period,
            control_group=control_group,
            counterfactual_model=counterfactual_model,
        )
        for agg_name, agg in estimate.items()
    }
    return NonlinearDidResults(**results)
