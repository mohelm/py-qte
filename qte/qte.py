from functools import partial

import numpy as np
import polars as pl
from numpy.typing import NDArray

from qte.bootstrap import perform_bootstrap
from qte.results import QteResult
from qte.se import get_se
from qte.stats import get_quantiles


def _compute_simple_qte(
    ds: pl.DataFrame,
    outcome_c: str,
    treatment_c: str,
    qs: NDArray[np.float64],
    weight_c: str | None = None,
):
    treated, control = (
        ds.filter(pl.col(treatment_c) == 1.0),
        ds.filter(pl.col(treatment_c) == 0.0),
    )
    q_t = get_quantiles(
        qs,
        treated[outcome_c].to_numpy(),
        weight_c if weight_c is None else treated[weight_c].to_numpy(),
    )
    q_c = get_quantiles(
        qs,
        control[outcome_c].to_numpy(),
        weight_c if weight_c is None else control[weight_c].to_numpy(),
    )
    return QteResult(qs, q_t, q_c)


def estimate_qte(
    ds: pl.DataFrame,
    outcome_c: str,
    treatment_c: str,
    qs: NDArray[np.float64] = (0.5,),
    weight_c: str | None = None,
    n_bootstrap_iter: int = 100,
) -> QteResult:
    fcn = partial(
        _compute_simple_qte,
        outcome_c=outcome_c,
        treatment_c=treatment_c,
        qs=qs,
        weight_c=weight_c,
    )
    estimate = fcn(ds)
    bs_res = perform_bootstrap(ds, fcn=fcn, n_iter=n_bootstrap_iter)
    return estimate.to_polars().join(get_se(bs_res), on="qs")
