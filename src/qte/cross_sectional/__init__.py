from functools import partial

import numpy as np
import polars as pl
from numpy.typing import ArrayLike

from qte.constants import MEDIAN
from qte.cross_sectional.aipw import compute_aipw_qte
from qte.cross_sectional.bootstrap import (
    get_statistics_from_bootstrap,
    perform_bootstrap,
)
from qte.cross_sectional.ipw import compute_ipw_qte as compute_ipw_qte
from qte.cross_sectional.or_ import compute_or_qte
from qte.cross_sectional.results import QteResult
from qte.cross_sectional.results import _QteIntermediateResult as _QteIntermediateResult
from qte.cross_sectional.simple import compute_simple_qte
from qte.custom_types import CausalTarget, ColumnName, Estimator, FormularRhs
from qte.names import QUANTILE_ID, SE_ID


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
    bs_it = perform_bootstrap(ds, fcn=fcn, n_iter=n_bootstrap_iter)
    bs_stats = get_statistics_from_bootstrap(bs_it)
    return QteResult(
        qtt=estimate.qtt.join(bs_stats.qtt, on=QUANTILE_ID),
        att=estimate.att.with_columns(bs_stats.att[SE_ID]),
        causal_target=CausalTarget.QTE,
        estimator=Estimator.SIMPLE,
        outcome=outcome_c,
        group=None,
    )


def estimate_ipw_qte(
    ds: pl.DataFrame,
    outcome_c: ColumnName,
    treatment_c: ColumnName,
    qs: ArrayLike = MEDIAN,
    *,
    ps_x_formular: FormularRhs,
    target: CausalTarget = CausalTarget.QTE,
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
    bs_it = perform_bootstrap(ds, fcn=fcn, n_iter=n_bootstrap_iter)
    bs_stats = get_statistics_from_bootstrap(bs_it)
    return QteResult(
        qtt=estimate.qtt.join(bs_stats.qtt, on=QUANTILE_ID),
        att=estimate.att.with_columns(bs_stats.att[SE_ID]),
        causal_target=target,
        estimator=Estimator.IPW,
        outcome=outcome_c,
        group=None,
        ps_x_formular=ps_x_formular,
    )


def estimate_or_qte(
    ds: pl.DataFrame,
    outcome_c: str,
    treatment_c: str,
    qs: ArrayLike = MEDIAN,
    *,
    or_x_formular: str,
    target: CausalTarget = CausalTarget.QTE,
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
    bs_it = perform_bootstrap(ds, fcn=fcn, n_iter=n_bootstrap_iter)
    bs_stats = get_statistics_from_bootstrap(bs_it)
    return QteResult(
        qtt=estimate.qtt.join(bs_stats.qtt, on=QUANTILE_ID),
        att=estimate.att.with_columns(bs_stats.att[SE_ID]),
        causal_target=target,
        estimator=Estimator.OR,
        outcome=outcome_c,
        group=None,
        or_x_formular=or_x_formular,
    )


def estimate_aipw_qte(
    ds: pl.DataFrame,
    outcome_c: str,
    treatment_c: str,
    qs: ArrayLike = (0.5,),
    *,
    ps_x_formular: FormularRhs,
    or_x_formular: FormularRhs,
    target: CausalTarget = CausalTarget.QTE,
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
    bs_it = perform_bootstrap(ds, fcn=fcn, n_iter=n_bootstrap_iter)
    bs_stats = get_statistics_from_bootstrap(bs_it)
    return QteResult(
        qtt=estimate.qtt.join(bs_stats.qtt, on=QUANTILE_ID),
        att=estimate.att.with_columns(bs_stats.att[SE_ID]),
        causal_target=target,
        estimator=Estimator.AIPW,
        outcome=outcome_c,
        group=None,
        ps_x_formular=ps_x_formular,
        or_x_formular=or_x_formular,
    )
