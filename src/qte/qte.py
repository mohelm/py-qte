from enum import StrEnum
from functools import partial

import numpy as np
import polars as pl
import statsmodels.formula.api as smf
from numpy.typing import ArrayLike, NDArray

from qte.bootstrap import perform_bootstrap
from qte.custom_types import ColumnName, DataFrame, FormularRhs
from qte.quantile_regression import QuantileRegression, QuantileRegressionResult
from qte.results import QteResult
from qte.se import get_se
from qte.stats import get_quantiles


class CausalTarget(StrEnum):
    QTT = "qtt"
    QTE = "qte"


def _compute_simple_qte(
    ds: pl.DataFrame,
    outcome_c: ColumnName,
    treatment_c: ColumnName,
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


def _estimate_propensity_score(ds: pl.DataFrame, treatment_c: str, x_formular: str):
    ps_full_formular = f"{treatment_c} ~ {x_formular}"

    ds_as_dict = {col.name: col.to_numpy() for col in ds.iter_columns()}
    return smf.logit(formula=ps_full_formular, data=ds_as_dict).fit(disp=0)


def _prepare_dataset_for_stats_models(ds: pl.DataFrame) -> dict[str, NDArray]:
    return {col.name: col.to_numpy() for col in ds.iter_columns()}


def _compute_weighted_qte(
    ds: pl.DataFrame,
    outcome_c: str,
    treatment_c: str,
    ps_x_formular: str,
    qs: NDArray[np.float64],
    weight_c: str | None = None,
    target: CausalTarget = CausalTarget.QTE,
) -> QteResult:
    ps = _estimate_propensity_score(ds, treatment_c, ps_x_formular).predict()

    if target == CausalTarget.QTE:
        weights = 1 if weight_c is None else ds[weight_c].to_numpy()
        bw_treated = weights * ds[treatment_c].to_numpy() / ps
        bw_control = weights * (1 - ds[treatment_c]).to_numpy() / (1 - ps)
        q_t = get_quantiles(qs, ds[outcome_c].to_numpy(), bw_treated)
        q_c = get_quantiles(qs, ds[outcome_c].to_numpy(), bw_control)

    if target == CausalTarget.QTT:
        treated, control = (
            ds.filter(pl.col(treatment_c) == 1),
            ds.filter(pl.col(treatment_c) == 0),
        )
        q_t = get_quantiles(
            qs,
            treated[outcome_c].to_numpy(),
            treated[weight_c].to_numpy() if weight_c is not None else None,
        )
        control_obs_selector = ds[treatment_c].to_numpy() == 0
        weights = (
            1 if weight_c is None else ds[weight_c].to_numpy()[:, control_obs_selector]
        )
        ps_c = ps[control_obs_selector]
        bw_control = weights * ps_c / (1 - ps_c)
        q_c = get_quantiles(qs, control[outcome_c].to_numpy(), bw_control)

    return QteResult(qs, q_t, q_c)


def estimate_simple_qte(
    ds: pl.DataFrame,
    outcome_c: str,
    treatment_c: str,
    qs: ArrayLike = (0.5,),
    *,
    weight_c: str | None = None,
    n_bootstrap_iter: int = 100,
) -> QteResult:
    qs = np.array(qs)
    fcn = partial(
        _compute_simple_qte,
        outcome_c=outcome_c,
        treatment_c=treatment_c,
        qs=qs,
        weight_c=weight_c,
    )
    estimate = fcn(ds)
    bs_res = perform_bootstrap(ds, fcn=fcn, n_iter=n_bootstrap_iter)
    return estimate.to_polars().join(get_se(bs_res), on="q").sort("q")


def estimate_ipw_qte(
    ds: pl.DataFrame,
    outcome_c: ColumnName,
    treatment_c: ColumnName,
    qs: ArrayLike = (0.5,),
    *,
    target: CausalTarget = CausalTarget.QTE,
    ps_x_formular: FormularRhs,
    weight_c: str | None = None,
    n_bootstrap_iter: int = 100,
):
    qs = np.array(qs)
    fcn = partial(
        _compute_weighted_qte,
        outcome_c=outcome_c,
        treatment_c=treatment_c,
        ps_x_formular=ps_x_formular,
        qs=qs,
        weight_c=weight_c,
        target=target,
    )
    estimate = fcn(ds)
    bs_res = perform_bootstrap(ds, fcn=fcn, n_iter=n_bootstrap_iter)
    return estimate.to_polars().join(get_se(bs_res), on="q").sort("q")


def _estimate_outcome_model(
    ds: pl.DataFrame, outcome_c: ColumnName, x_formular: str, qs: NDArray[np.float64]
) -> QuantileRegressionResult:
    return QuantileRegression(f"{outcome_c}~{x_formular}", data=ds).fit(qs)


def _predict_outcome_model(
    or_: QuantileRegressionResult, ds: pl.DataFrame | None = None
) -> NDArray[np.float64]:
    return np.sort(or_.predict(ds), axis=1).flatten()


def _make_weights(
    weights_c: str | None, ds: pl.DataFrame, rep: int | None = None
) -> NDArray[np.float64] | None:
    if weights_c is None:
        return None
    weights = ds[weights_c].to_numpy()
    return weights if rep is None else np.tile(weights, rep)


def _compute_or_qte(
    ds: DataFrame,
    outcome_c: ColumnName,
    treatment_c: ColumnName,
    qs: NDArray[np.float64] = (0.5,),  # type: ignore
    *,
    weights_c: ColumnName | None = None,
    or_x_formular: str | None = None,
    target: CausalTarget = CausalTarget.QTE,
    or_quantiles: NDArray = np.arange(0.01, 0.99, 0.01),
) -> QteResult:
    treated, control = (
        ds.filter(pl.col(treatment_c) == 1),
        ds.filter(pl.col(treatment_c) == 0),
    )
    ors_control = _estimate_outcome_model(
        control, outcome_c, or_x_formular, or_quantiles
    )

    if target == CausalTarget.QTE:
        preds_control = _predict_outcome_model(ors_control, ds)

        weights = _make_weights(weights_c, ds, or_quantiles.shape[0])
        q_c = get_quantiles(qs, preds_control, weights)

        ors_treated = _estimate_outcome_model(
            treated, outcome_c, or_x_formular, or_quantiles
        )
        preds_treated = _predict_outcome_model(ors_treated, ds)
        q_t = get_quantiles(qs, preds_treated, weights)

    if target == CausalTarget.QTT:
        preds_control = _predict_outcome_model(ors_control, treated)
        weights = _make_weights(weights_c, treated, or_quantiles.shape[0])
        q_c = get_quantiles(qs, preds_control, weights)
        q_t = get_quantiles(
            qs, treated[outcome_c].to_numpy(), w=_make_weights(weights_c, treated)
        )

    return QteResult(qs, q_t, q_c)


def estimate_or_qte(
    ds: pl.DataFrame,
    outcome_c: str,
    treatment_c: str,
    qs: ArrayLike = (0.5,),
    *,
    target: CausalTarget = CausalTarget.QTE,
    or_x_formular: str | None = None,
    weights_c: str | None = None,
    n_bootstrap_iter: int = 100,
) -> pl.DataFrame:
    fcn = partial(
        _compute_or_qte,
        outcome_c=outcome_c,
        treatment_c=treatment_c,
        or_x_formular=or_x_formular,
        qs=qs,
        weights_c=weights_c,
        target=target,
    )
    estimate = fcn(ds)
    bs_res = perform_bootstrap(ds, fcn=fcn, n_iter=n_bootstrap_iter)
    return estimate.to_polars().join(get_se(bs_res), on="q").sort("q")


def estimate_aipw_qte(
    ds: pl.DataFrame,
    outcome_c: str,
    treatment_c: str,
    qs: NDArray[np.float64] = (0.5,),
    *,
    pw_x_formular: str | None = None,
    or_x_formular: str | None = None,
    weight_c: str | None = None,
    n_bootstrap_iter: int = 100,
    target: CausalTarget = CausalTarget.QTE,
):
    if target == CausalTarget.QTE:
        ...
    if target == CausalTarget.QTT:
        ...
