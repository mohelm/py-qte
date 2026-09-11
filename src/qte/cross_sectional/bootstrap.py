from collections.abc import Callable, Iterable, Iterator

import polars as pl

from qte.cross_sectional.results import _QteIntermediateResult
from qte.names import EFFECT_ID, QUANTILE_ID, SE_ID


def perform_bootstrap(
    ds: pl.DataFrame,
    fcn: Callable[[pl.DataFrame], _QteIntermediateResult],
    n_iter: int = 100,
) -> Iterator[_QteIntermediateResult]:
    return (fcn(ds.sample(fraction=1.0, with_replacement=True)) for _ in range(n_iter))


def get_statistics_from_bootstrap(
    boot_iter: Iterable[_QteIntermediateResult],
) -> _QteIntermediateResult:
    runs = list(boot_iter)
    agg = pl.col(EFFECT_ID).std().alias(SE_ID)
    qte = pl.concat([r.qtt for r in runs]).group_by(QUANTILE_ID).agg(agg)
    att = pl.concat([r.att for r in runs]).group_by([]).agg(agg)
    return _QteIntermediateResult(qte, att)
