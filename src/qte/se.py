from collections.abc import Iterator

import polars as pl

from qte.results import QteResult


def get_se(bs_results: Iterator[QteResult]) -> pl.DataFrame:
    return (
        pl
        .concat(bs_r.to_polars() for bs_r in bs_results)
        .group_by("q")
        .agg(se=pl.col("effect").std())
    )
