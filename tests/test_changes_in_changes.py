import polars as pl

from qte.non_linear_did.changes_in_changes import (
    estimate_changes_in_changes_for_panel,
)


def test_estimate_changes_in_changes_for_panel(mpdata):
    ds = mpdata.with_columns(
        pl.col("first.treat").replace(0, float("inf")).alias("first.treat"),
        pl.col("year").cast(pl.Float64).alias("year"),
    )
    _ = estimate_changes_in_changes_for_panel(
        ds, "lemp", "first.treat", "year", "countyreal", qs=[0.25, 0.5, 0.75]
    )
