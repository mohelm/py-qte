from collections.abc import Callable, Iterable, Iterator
from functools import partial
from itertools import product
from typing import NamedTuple

import numpy as np
import polars as pl
from numpy.typing import ArrayLike, NDArray

from qte.constants import MEDIAN
from qte.names import EFFECT_ID, QUANTILE_ID
from qte.nonlinear_difference_in_differences.aggregate import (
    aggregate_group_time_effects_again,
    aggregate_group_time_effects_again_by_group,
)
from qte.nonlinear_difference_in_differences.custom_types import (
    BasePeriod,
    CicAggregations,
    ControlGroup,
)
from qte.nonlinear_difference_in_differences.results import (
    CicResult,
    CicResults,
    GroupTimeEffect,
)


class Ecdf(NamedTuple):
    values: NDArray
    probs: NDArray
    weights: NDArray | None = None

    def evaluate_inverse(self, qs: NDArray) -> NDArray:
        idx = np.searchsorted(self.probs, qs, side="left")
        idx = np.clip(idx, 0, len(self.values) - 1)
        return self.values[idx]


def _make_base_period(
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
        (is_treated_g | is_control_g)
        & pl.col(time_c).is_in((time_period, reference_period))
    ).with_columns(
        _is_treated=pl.col(treatment_group_c) == treated_group,
        _is_post=pl.col(time_c) == time_period,
    )


def _get_group_data(
    ds: pl.DataFrame, filter_: pl.Expr, outcome_c: str, weight_c: str, unit_c: str
) -> NDArray:
    return (
        ds.filter(filter_)
        .select(
            outcome_c,
            weight_c,
            unit_c,
        )
        .sort(outcome_c)
        .rename({outcome_c: "o", weight_c: "w", unit_c: "u"})
        .to_numpy(structured=True)
    )


def _agg_ecdf(ecdf, grid) -> NDArray:
    idx = np.searchsorted(ecdf["o"], grid, side="right")
    padded = np.concatenate(([0.0], ecdf["ecdf"]))
    return padded[idx]


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


def _get_ecdf(y: NDArray, w: NDArray) -> dict[str, NDArray]:
    sorter = np.argsort(y)
    y_sorted = y[sorter]
    w_sorted = w[sorter]

    # Handle exact ties: accumulate weights for identical outcome values
    y_unique, indices = np.unique(y_sorted, return_inverse=True)
    w_unique = np.bincount(indices, weights=w_sorted)

    w_norm = w_unique / w_unique.sum()
    return {"o": y_unique, "ecdf": np.cumsum(w_norm), "w": w_unique}


def _compute_group_time_effect(
    two_by_two_data: pl.DataFrame,
    g: int,
    tp: int,
    outcome_c,
    unit_c,
    weight_c: str,
) -> GroupTimeEffect:
    _group_time_extractor = partial(
        _get_group_data,
        two_by_two_data,
        outcome_c=outcome_c,
        weight_c=weight_c,
        unit_c=unit_c,
    )

    post_trt = _group_time_extractor(pl.col("_is_treated") & pl.col("_is_post"))
    pre_trt = _group_time_extractor(pl.col("_is_treated") & (~pl.col("_is_post")))
    pre_ctrl = _group_time_extractor((~pl.col("_is_treated")) & (~pl.col("_is_post")))
    post_ctrl = _group_time_extractor((~pl.col("_is_treated")) & pl.col("_is_post"))

    kcic = _compute_kcic(pre_trt, pre_ctrl, post_ctrl)
    ecdf_post_treated_observed = _get_ecdf(post_trt["o"], post_trt["w"])
    ecdf_post_treated_counterfact = _get_ecdf(kcic, pre_trt["w"])

    return GroupTimeEffect(
        group=g,
        tp=tp,
        ecdf_observed=ecdf_post_treated_observed,
        ecdf_counterfact=ecdf_post_treated_counterfact,
        mean_observed=np.average(post_trt["o"], weights=post_trt["w"]),
        mean_countfact=np.average(kcic, weights=pre_trt["w"]),
        group_size_observed=post_trt["w"].sum(),
        group_size_counterfactual=pre_trt["w"].sum(),
    )


def weights_to_dict(d: pl.DataFrame) -> dict[tuple[int, int], float]:
    return dict(zip(zip(d["group"], d["time_period"]), d["weight"]))


def _compute_changes_in_changes_for_panel(
    ds: pl.DataFrame,
    outcome_c: str,
    treatment_group_c: str,
    time_c: str,
    unit_c: str,
    qs: ArrayLike = MEDIAN,
    *,
    weight_c: str,
    base_period: BasePeriod = BasePeriod.UNIVERSAL,
    control_group: ControlGroup = ControlGroup.NEVER_TREATED,
    n_anticipation_periods=0,
) -> CicAggregations:

    outcome_grid_size = 1000

    post_trt_ds = ds.filter(
        pl.col(treatment_group_c).is_finite()
        & (pl.col(time_c) >= pl.col(treatment_group_c))
    )
    outcome = post_trt_ds[outcome_c].unique().to_numpy()
    y_grid = np.linspace(outcome.min(), outcome.max(), outcome_grid_size)

    # Compute group time effects
    time_periods = ds[time_c].unique().sort()
    if base_period == BasePeriod.VARYING:
        time_periods = time_periods.slice(1)
    treated_groups = ds[treatment_group_c].unique().sort()[:-1]  # TODO: FIX

    names = {"outcome_c": outcome_c, "weight_c": weight_c, "unit_c": unit_c}
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
        ).pipe(_compute_group_time_effect, g, tp, **names)
        for tp, g in product(time_periods, treated_groups)
        if (rp := _make_base_period(g, tp, n_anticipation_periods, base_period))
        in time_periods
        and not (tp == rp and base_period == BasePeriod.UNIVERSAL)
    ]

    group_level_effect_weights = (
        ds.group_by(
            pl.col(treatment_group_c).alias("group"),
            pl.col(time_c).alias("time_period"),
        )
        .agg(weight=pl.col(weight_c).sum())
        .filter(pl.col("time_period") >= pl.col("group"))
        .with_columns(weight=pl.col("weight") / pl.col("weight").sum().over("group"))
    )
    post_trt_group_time_effects = [
        gte for gte in group_time_effects if gte.tp >= gte.group
    ]
    group_te = aggregate_group_time_effects_again_by_group(
        qs,
        post_trt_group_time_effects,
        group_level_effect_weights.pipe(weights_to_dict),
        y_grid,
        dim_id=lambda gte: gte.group,
        dim_name="group",
    )

    overall_effect_weights = (
        ds.group_by(
            pl.col(treatment_group_c).alias("group"),
            pl.col(time_c).alias("time_period"),
        )
        .agg(weight=pl.col(weight_c).sum())
        .filter(pl.col("time_period") >= pl.col("group"))
        .with_columns(
            weight=(pl.col("weight") / pl.col("weight").sum().over("group"))
            * (
                pl.col("weight").first().over("group")
                / (
                    (
                        pl.col("weight").first().over("group") / pl.len().over("group")
                    ).sum()
                )
            ),
        )
    )
    agg_te = aggregate_group_time_effects_again(
        qs,
        post_trt_group_time_effects,
        overall_effect_weights.pipe(weights_to_dict),
        y_grid,
    )

    event_study_period = pl.col("time_period") - pl.col("group")
    event_study_effect_weights = (
        ds.group_by(
            pl.col(treatment_group_c).alias("group"),
            pl.col(time_c).alias("time_period"),
        )
        .agg(weight=pl.col(weight_c).sum())
        .with_columns(
            weight=pl.col("weight") / pl.col("weight").sum().over(event_study_period)
        )
    ).pipe(weights_to_dict)
    event_study_te = aggregate_group_time_effects_again_by_group(
        qs,
        group_time_effects,
        event_study_effect_weights,
        y_grid,
        dim_id=lambda gte: gte.tp - gte.group,
        dim_name="es_period",
    )

    return CicAggregations(group=group_te, event_study=event_study_te, overall=agg_te)


def perform_bootstrap(
    ds: pl.DataFrame,
    fcn: Callable[[pl.DataFrame], CicAggregations],
    unit_id: str,
    n_iter: int,
) -> Iterator[CicAggregations]:
    units = ds.select(unit_id).unique()
    n_units = len(units)
    for _ in range(n_iter):
        # 1. Resample unit IDs with replacement and assign new unique IDs
        sampled_units = units.sample(n=n_units, with_replacement=True).with_columns(
            pl.int_range(0, n_units).alias("new_id")
        )

        boot_ds = (
            sampled_units.join(ds, on=unit_id, how="inner")
            .with_columns(pl.col("new_id").alias(unit_id))
            .drop("new_id")
        )

        yield fcn(boot_ds)


class Estimates(NamedTuple):
    atts: pl.DataFrame
    qtes: pl.DataFrame


def stack_dict_bootstraps(
    boot_iter: Iterable[CicAggregations],
    aggregations: Iterable[tuple[str, str | None]],
) -> dict[str, Estimates]:
    runs = list(boot_iter)

    return {
        aggregation: Estimates(
            qtes=pl.concat(
                r[aggregation].qtt.with_columns(boot_id=i) for i, r in enumerate(runs)
            )
            .group_by(
                *([grouper, QUANTILE_ID] if grouper is not None else [QUANTILE_ID])
            )
            .agg(se=pl.col(EFFECT_ID).std()),
            atts=pl.concat(
                r[aggregation].att.with_columns(boot_id=i) for i, r in enumerate(runs)
            )
            .group_by(grouper if grouper is not None else [])
            .agg(se=pl.col(EFFECT_ID).std()),
        )
        for aggregation, grouper in aggregations
    }


def estimate_changes_in_changes_for_panel(
    ds: pl.DataFrame,
    outcome_c: str,
    treatment_group_c: str,
    time_c: str,
    unit_c: str,
    qs: ArrayLike = MEDIAN,
    *,
    weight_c: str | None = None,
    n_anticipation_periods=0,
    base_period: BasePeriod = BasePeriod.UNIVERSAL,
    control_group: ControlGroup = ControlGroup.NEVER_TREATED,
    n_bootstrap_iter: int = 1000,
) -> CicResults:
    if weight_c is None:
        weight_c = "_w"
        ds = ds.with_columns(pl.lit(1).alias(weight_c))
    ntg_id = float("inf")
    all_groups = ds[treatment_group_c].unique()
    all_treated_groups = all_groups.filter(all_groups.is_finite())
    time_periods = ds[time_c].unique().sort().to_list()

    treated_groups: pl.Series = all_treated_groups.filter(
        all_treated_groups >= min(time_periods) + 1 + n_anticipation_periods
    )
    ds = ds.filter(
        pl.col(treatment_group_c).is_in(set(treated_groups.to_list()).union([ntg_id]))
    )
    fcn = partial(
        _compute_changes_in_changes_for_panel,
        outcome_c=outcome_c,
        treatment_group_c=treatment_group_c,
        time_c=time_c,
        unit_c=unit_c,
        qs=qs,
        weight_c=weight_c,
        base_period=base_period,
        control_group=control_group,
        n_anticipation_periods=n_anticipation_periods,
    )
    estimate: CicAggregations = fcn(ds)
    bs_iterations = perform_bootstrap(ds, fcn, unit_c, n_iter=n_bootstrap_iter)
    # We must explicitly type cast iteration items to silence ty
    groupers: list[tuple[str, str | None]] = [
        (agg_name, agg.group) for agg_name, agg in estimate.items()
    ]
    bs_aggs = stack_dict_bootstraps(bs_iterations, groupers)
    combined = {
        agg_name: CicResult(
            agg.qtt.join(
                bs_aggs[agg_name].qtes,
                on=[QUANTILE_ID, agg.group] if agg.group is not None else [QUANTILE_ID],
            ),
            agg.att.join(
                bs_aggs[agg_name].atts,
                on=[agg.group] if agg.group is not None else [],
            ),
            group=agg.group,
            outcome=outcome_c,
            base_period=base_period,
            control_group=control_group,
        )
        for agg_name, agg in estimate.items()
    }

    return CicResults(**combined)
