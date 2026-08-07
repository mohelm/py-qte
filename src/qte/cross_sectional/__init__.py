from functools import partial

import numpy as np
import polars as pl
from numpy.typing import ArrayLike

from qte.constants import MEDIAN
from qte.cross_sectional.aipw import compute_aipw_qte
from qte.cross_sectional.bootstrap import perform_bootstrap
from qte.cross_sectional.ipw import compute_ipw_qte as compute_ipw_qte
from qte.cross_sectional.or_ import compute_or_qte
from qte.cross_sectional.results import QteResult
from qte.cross_sectional.results import _QteIntermediateResult as _QteIntermediateResult
from qte.cross_sectional.se import get_se
from qte.cross_sectional.simple import compute_simple_qte
from qte.custom_types import CausalTarget, ColumnName, Estimator, FormularRhs
from qte.names import QUANTILE_ID


def estimate_simple_qte(
    ds: pl.DataFrame,
    outcome_c: str,
    treatment_c: str,
    qs: ArrayLike = MEDIAN,
    *,
    weight_c: str | None = None,
    n_bootstrap_iter: int = 100,
) -> QteResult:
    qs = np.array(qs)
    fcn = partial(
        compute_simple_qte,
        outcome_c=outcome_c,
        treatment_c=treatment_c,
        qs=qs,
        weight_c=weight_c,
    )
    estimate = fcn(ds)
    bs_res = perform_bootstrap(ds, fcn=fcn, n_iter=n_bootstrap_iter)
    res = estimate.to_polars().join(get_se(bs_res), on=QUANTILE_ID)
    return QteResult(
        _res=res, causal_target=CausalTarget.QTE, estimator=Estimator.SIMPLE
    )


def estimate_ipw_qte(
    ds: pl.DataFrame,
    outcome_c: ColumnName,
    treatment_c: ColumnName,
    qs: ArrayLike = MEDIAN,
    *,
    target: CausalTarget = CausalTarget.QTE,
    ps_x_formular: FormularRhs,
    weight_c: str | None = None,
    n_bootstrap_iter: int = 100,
) -> QteResult:
    qs = np.array(qs)
    fcn = partial(
        compute_ipw_qte,
        outcome_c=outcome_c,
        treatment_c=treatment_c,
        ps_x_formular=ps_x_formular,
        qs=qs,
        weight_c=weight_c,
        target=target,
    )
    estimate = fcn(ds)
    bs_res = perform_bootstrap(ds, fcn=fcn, n_iter=n_bootstrap_iter)
    res = estimate.to_polars().join(get_se(bs_res), on=QUANTILE_ID)
    return QteResult(_res=res, causal_target=target, estimator=Estimator.IPW)


def estimate_or_qte(
    ds: pl.DataFrame,
    outcome_c: str,
    treatment_c: str,
    qs: ArrayLike = MEDIAN,
    *,
    target: CausalTarget = CausalTarget.QTE,
    or_x_formular: str | None = None,
    weights_c: str | None = None,
    n_bootstrap_iter: int = 100,
) -> QteResult:
    qs = np.array(qs)
    fcn = partial(
        compute_or_qte,
        outcome_c=outcome_c,
        treatment_c=treatment_c,
        or_x_formular=or_x_formular,
        qs=qs,
        weights_c=weights_c,
        target=target,
    )
    estimate = fcn(ds)
    bs_res = perform_bootstrap(ds, fcn=fcn, n_iter=n_bootstrap_iter)
    res = estimate.to_polars().join(get_se(bs_res), on=QUANTILE_ID)
    return QteResult(_res=res, causal_target=target, estimator=Estimator.OR)


def estimate_aipw_qte(
    ds: pl.DataFrame,
    outcome_c: str,
    treatment_c: str,
    qs: ArrayLike = (0.5,),
    *,
    target: CausalTarget = CausalTarget.QTE,
    or_x_formular: str | None = None,
    ps_x_formular: FormularRhs | None = None,
    weights_c: str | None = None,
    n_bootstrap_iter: int = 100,
) -> QteResult:
    qs = np.array(qs)
    fcn = partial(
        compute_aipw_qte,
        outcome_c=outcome_c,
        treatment_c=treatment_c,
        or_x_formular=or_x_formular,
        ps_x_formular=ps_x_formular,
        qs=qs,
        weights_c=weights_c,
        target=target,
    )
    estimate = fcn(ds)
    bs_res = perform_bootstrap(ds, fcn=fcn, n_iter=n_bootstrap_iter)
    res = estimate.to_polars().join(get_se(bs_res), on=QUANTILE_ID)
    return QteResult(_res=res, causal_target=target, estimator=Estimator.AIPW)
