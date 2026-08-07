from collections.abc import Iterator

import polars as pl

from qte.cross_sectional import _QteIntermediateResult


def get_se(bs_results: Iterator[_QteIntermediateResult]) -> pl.DataFrame:
    return (
        pl
        .concat(bs_r.to_polars() for bs_r in bs_results)
        .group_by("q")
        .agg(se=pl.col("effect").std())
    )
