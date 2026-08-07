from collections.abc import Callable, Iterator

import polars as pl

from qte.cross_sectional.results import _QteIntermediateResult


def perform_bootstrap(
    ds: pl.DataFrame, fcn: Callable[[pl.DataFrame], _QteIntermediateResult], n_iter: int = 100
) -> Iterator[_QteIntermediateResult]:
    return (fcn(ds.sample(fraction=1.0, with_replacement=True)) for _ in range(n_iter))
