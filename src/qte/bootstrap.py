from collections.abc import Callable, Iterator

import polars as pl

from qte.results import QteResult


def perform_bootstrap(
    ds: pl.DataFrame, fcn: Callable[[pl.DataFrame], QteResult], n_iter: int = 100
) -> Iterator[QteResult]:
    return (fcn(ds.sample(fraction=1.0, with_replacement=True)) for _ in range(n_iter))
