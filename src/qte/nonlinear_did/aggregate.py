from collections.abc import Callable

import numpy as np
import polars as pl
from numpy.typing import ArrayLike, NDArray

from qte.names import (
    EFFECT_ID,
    MEAN_CONTROL_ID,
    MEAN_TREATED_ID,
    QUANTILE_CONTROL_VAL_ID,
    QUANTILE_ID,
    QUANTILE_TREATED_VAL_ID,
)
from qte.nonlinear_did.custom_types import BasePeriod, NonlinearDidAggregation, WeightsLookup
from qte.nonlinear_did.results import GroupTimeEffect
from qte.stats import Ecdf


def _weights_to_dict(d: pl.DataFrame) -> WeightsLookup:
    return dict(zip(zip(d["group"], d["time_period"]), d["weight"]))


def get_weights_for_overall_effect(ds: pl.DataFrame) -> WeightsLookup:
    return (
        ds.filter(pl.col("time_period") >= pl.col("group"))
        .with_columns(
            weight=(pl.col("weight") / pl.col("weight").sum().over("group"))
            * (
                pl.col("weight").first().over("group")
                / ((pl.col("weight").first().over("group") / pl.len().over("group")).sum())
            ),
        )
        .pipe(_weights_to_dict)
    )


def get_weights_for_treatment_group_effects(ds: pl.DataFrame) -> WeightsLookup:
    return (
        ds.filter(pl.col("time_period") >= pl.col("group"))
        .with_columns(weight=pl.col("weight") / pl.col("weight").sum().over("group"))
        .pipe(_weights_to_dict)
    )


def get_weights_for_event_study_effects(ds: pl.DataFrame, base_period: BasePeriod) -> WeightsLookup:
    event_study_period = pl.col("time_period") - pl.col("group")
    if base_period == BasePeriod.VARYING:
        ds = ds.filter(pl.col("time_period") > pl.col("time_period").min())
    return ds.with_columns(
        weight=pl.col("weight") / pl.col("weight").sum().over(event_study_period)
    ).pipe(_weights_to_dict)


def _merge_group_time_effects_on_grid(
    gtes: list[GroupTimeEffect], weights: dict[tuple[int, int], float], y_grid: NDArray
) -> tuple[NDArray, NDArray, NDArray, NDArray]:

    gtes = [g for g in gtes if (g.group, g.tp) in weights]
    weights_ = np.array([weights[(gte.group, gte.tp)] for gte in gtes])[:, None]

    ecdfs_on_grid_o = np.array([g.ecdf_observed.evaluate(y_grid) for g in gtes]) * weights_
    ecdfs_on_grid_cf = np.array([g.ecdf_counterfact.evaluate(y_grid) for g in gtes]) * weights_
    means_o = np.array([gte.mean_observed for gte in gtes]) * weights_.squeeze()
    means_cf = np.array([gte.mean_countfact for gte in gtes]) * weights_.squeeze()
    return (ecdfs_on_grid_o, ecdfs_on_grid_cf, means_o, means_cf)


def aggregate_group_time_effects_again_by_group(
    qs: ArrayLike,
    gtes: list[GroupTimeEffect],
    weights: dict[tuple[int, int], float],
    y_grid: NDArray,
    *,
    dim_id: Callable,
    dim_name: str,
) -> NonlinearDidAggregation:

    qs = np.array(qs)
    ecdf_on_grid_o, ecdf_on_grid_cf, means_o, means_cf = _merge_group_time_effects_on_grid(
        gtes, weights, y_grid
    )

    dims = np.array([dim_id(gte) for gte in gtes])
    unique_dims, group_ids = np.unique(dims, return_inverse=True)
    n_groups = len(unique_dims)
    G = (group_ids[:, None] == np.arange(n_groups)).T.astype(float)
    ecdf_o = G @ ecdf_on_grid_o
    ecdf_cf = G @ ecdf_on_grid_cf
    qtts = pl.concat(
        pl.DataFrame(
            {
                dim_name: d,
                f"{QUANTILE_ID}": qs,
                f"{QUANTILE_TREATED_VAL_ID}": Ecdf(y_grid, ecdf_o[i_d]).evaluate_inverse(qs),
                f"{QUANTILE_CONTROL_VAL_ID}": Ecdf(y_grid, ecdf_cf[i_d]).evaluate_inverse(qs),
            }
        )
        for i_d, d in enumerate(unique_dims)
    ).with_columns(
        (pl.col(QUANTILE_TREATED_VAL_ID) - pl.col(QUANTILE_CONTROL_VAL_ID)).alias(EFFECT_ID)
    )
    atts = pl.DataFrame(
        {
            dim_name: unique_dims,
            f"{MEAN_TREATED_ID}": G @ means_o,
            f"{MEAN_CONTROL_ID}": G @ means_cf,
        }
    ).with_columns((pl.col(MEAN_TREATED_ID) - pl.col(MEAN_CONTROL_ID)).alias(EFFECT_ID))
    return NonlinearDidAggregation(qtts, atts, dim_name)


def aggregate_group_time_effects_again(
    qs: ArrayLike,
    gtes: list[GroupTimeEffect],
    weights: dict[tuple[int, int], float],
    y_grid: NDArray,
) -> NonlinearDidAggregation:
    ecdf_on_grid_o, ecdf_on_grid_cf, means_o, means_cf = _merge_group_time_effects_on_grid(
        gtes, weights, y_grid
    )
    qs = np.array(qs)

    qtes = pl.DataFrame(
        {
            f"{QUANTILE_ID}": qs,
            f"{QUANTILE_TREATED_VAL_ID}": Ecdf(y_grid, ecdf_on_grid_o.sum(axis=0)).evaluate_inverse(
                qs
            ),
            f"{QUANTILE_CONTROL_VAL_ID}": Ecdf(
                y_grid, ecdf_on_grid_cf.sum(axis=0)
            ).evaluate_inverse(qs),
        }
    ).with_columns(
        (pl.col(QUANTILE_TREATED_VAL_ID) - pl.col(QUANTILE_CONTROL_VAL_ID)).alias(EFFECT_ID)
    )

    atts = pl.DataFrame(
        {
            f"{MEAN_TREATED_ID}": [means_o.sum()],
            f"{MEAN_CONTROL_ID}": [means_cf.sum()],
        }
    ).with_columns((pl.col(MEAN_TREATED_ID) - pl.col(MEAN_CONTROL_ID)).alias(EFFECT_ID))
    return NonlinearDidAggregation(qtes, atts, None)
