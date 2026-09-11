from collections.abc import Callable, Iterable, Iterator
from typing import NamedTuple

import polars as pl

from qte.names import EFFECT_ID, QUANTILE_ID, SE_ID
from qte.nonlinear_difference_in_differences.custom_types import (
    CicAggregations,
)


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


def get_statistics_from_bootstrap(
    boot_iter: Iterable[CicAggregations],
    aggregations: Iterable[tuple[str, str | None]],
) -> dict[str, Estimates]:
    runs = list(boot_iter)
    agg = pl.col(EFFECT_ID).std().alias(SE_ID)

    return {
        aggregation: Estimates(
            qtes=pl.concat(
                r[aggregation].qtt.with_columns(boot_id=i) for i, r in enumerate(runs)
            )
            .group_by(
                *([grouper, QUANTILE_ID] if grouper is not None else [QUANTILE_ID])
            )
            .agg(agg),
            atts=pl.concat(
                r[aggregation].att.with_columns(boot_id=i) for i, r in enumerate(runs)
            )
            .group_by(grouper if grouper is not None else [])
            .agg(agg),
        )
        for aggregation, grouper in aggregations
    }
