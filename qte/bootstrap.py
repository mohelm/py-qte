from collections.abc import Callable

import polars as pl

from qte.results import QteResult


def perform_bootstrap(
    ds: pl.DataFrame, fcn: Callable[[pl.DataFrame], QteResult], n_iter: int = 100
) -> list[QteResult]:
    return [fcn(ds.sample(fraction=1, with_replacement=True)) for _ in range(n_iter)]
