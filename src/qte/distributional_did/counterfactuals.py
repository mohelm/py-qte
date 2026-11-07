"""Counterfactual distributions for the distributional difference-in-differences models.

Each model is a `_CounterfactualSpec`: a 2x2 workhorse that turns the four observed cells of a
group-time comparison into a counterfactual outcome for every unit of one of them. The specs are
interchangeable -- they are registered in `_SPECS` and dispatched on by the estimator engine -- so
adding a model means adding a function here, not touching the estimation machinery.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from qte.stats import Ecdf, get_quantiles


@dataclass(frozen=True)
class _Counterfactual:
    """The counterfactual outcome distribution of one group-time cell.

    Parameters
    ----------
    y_counterfactual : NDArray
        Counterfactual untreated potential outcomes, one entry per unit of the cell that the
        model is indexed on (the treated pre-period cell for all currently supported models).
    weights : NDArray
        Sampling weights that belong to ``y_counterfactual``. Models that index the
        counterfactual on the control group must pass the control weights here.
    group_size : int
        Total weight of the units the counterfactual is indexed on.
    """

    y_counterfactual: NDArray
    weights: NDArray
    group_size: int


@dataclass(frozen=True)
class _CounterfactualSpec:
    """A model that constructs the counterfactual distribution of a 2x2 cell.

    Parameters
    ----------
    key : str
        Short name of the model, also stored on the results as `Estimator`.
    compute : Callable
        Maps the three cells it needs to a `_Counterfactual`.
    references : tuple of str
        Papers to cite for the identifying assumption.
    """

    key: str
    compute: Callable[[NDArray, NDArray, NDArray], _Counterfactual]
    references: tuple[str, ...] = ()


def _compute_changes_in_changes(
    pre_trt: NDArray, pre_ctrl: NDArray, post_ctrl: NDArray
) -> _Counterfactual:
    # Changes-in-changes maps the treated pre-treatment outcomes through the control group's
    # pre-to-post quantile transformation.

    # Compute the ecdf of (pre/ctrl)
    w_pre_ctrl_norm = pre_ctrl["w"] / pre_ctrl["w"].sum()
    cdf_pre_ctrl = np.concatenate(([0.0], np.cumsum(w_pre_ctrl_norm)))

    # Now find where the (pre/trt) values are on this cdf. That is, essentially evaluate the
    # (pre/trt) values at the (pre/ctrl) cdf.
    idx_u = np.searchsorted(pre_ctrl["o"], pre_trt["o"], side="right")
    u = cdf_pre_ctrl[idx_u]

    # Get the (post/ctrl) cdf
    w_post_ctrl_norm = post_ctrl["w"] / post_ctrl["w"].sum()
    cdf_post_ctrl = np.cumsum(w_post_ctrl_norm)

    # Now check where each u (which is the (pre/trt) values of the (pre/ctrl) cdf) fall on the
    # (post/ctrl) cdf (at what index).
    idx_k = np.searchsorted(cdf_post_ctrl, u, side="left")
    idx_k = np.clip(idx_k, 0, len(post_ctrl["o"]) - 1)
    return _Counterfactual(
        y_counterfactual=post_ctrl["o"][idx_k],
        weights=pre_trt["w"],
        group_size=int(pre_trt["w"].sum()),
    )


def _compute_quantile_did(
    pre_trt: NDArray, pre_ctrl: NDArray, post_ctrl: NDArray
) -> _Counterfactual:
    # Quantile difference-in-differences shifts every treated pre-treatment observation by the
    # control group's quantile change at that observation's rank *within the treated pre-treatment
    # distribution* (unlike changes-in-changes, which ranks within the control's pre-treatment
    # distribution).
    y = pre_trt["o"]
    _, inv = np.unique(y, return_inverse=True)
    w_unique = np.bincount(inv, weights=pre_trt["w"])
    u = np.cumsum(w_unique)[inv] / w_unique.sum()
    u = np.clip(u, 0.0, 1.0)

    q_pre_ctrl = get_quantiles(u, pre_ctrl["o"], pre_ctrl["w"])
    q_post_ctrl = get_quantiles(u, post_ctrl["o"], post_ctrl["w"])

    return _Counterfactual(
        y_counterfactual=y + q_post_ctrl - q_pre_ctrl,
        weights=pre_trt["w"],
        group_size=int(pre_trt["w"].sum()),
    )


def _compute_mean_did(pre_trt: NDArray, pre_ctrl: NDArray, post_ctrl: NDArray) -> _Counterfactual:
    # Mean difference-in-differences shifts the whole treated pre-treatment distribution by the
    # change in the control group's mean, that is the counterfactual is a pure location shift. The
    # reported quantile treatment effects are then not identified: they vary with the quantile only
    # because the treated group's own distribution changes.
    delta = np.average(post_ctrl["o"], weights=post_ctrl["w"]) - np.average(
        pre_ctrl["o"], weights=pre_ctrl["w"]
    )
    return _Counterfactual(
        y_counterfactual=pre_trt["o"] + delta,
        weights=pre_trt["w"],
        group_size=int(pre_trt["w"].sum()),
    )


def _compute_distributional_did(
    pre_trt: NDArray, pre_ctrl: NDArray, post_ctrl: NDArray
) -> _Counterfactual:
    # Distributional difference-in-differences pairs each control unit's own change over time with
    # the treated pre-treatment outcome at that unit's rank in the control pre-treatment
    # distribution. The pairing therefore carries the control group's dependence between the change
    # and the initial level over to the treated group; the counterfactual is indexed on the control
    # units, which is why no unit-level treatment effects are identified.
    pre_ctrl_by_unit = pre_ctrl[np.argsort(pre_ctrl["u"])]
    post_ctrl_by_unit = post_ctrl[np.argsort(post_ctrl["u"])]
    if not np.array_equal(pre_ctrl_by_unit["u"], post_ctrl_by_unit["u"]):
        raise ValueError(
            "distributional difference-in-differences needs the same control units in both "
            "periods of a group-time cell to form each unit's change; got "
            f"{len(pre_ctrl_by_unit)} units in the earlier period and "
            f"{len(post_ctrl_by_unit)} in the later one. Drop units that are not observed in "
            "both periods, or use another estimator."
        )
    delta_ctrl = post_ctrl_by_unit["o"] - pre_ctrl_by_unit["o"]

    u = Ecdf.make(pre_ctrl["o"], pre_ctrl["w"]).evaluate(pre_ctrl_by_unit["o"])
    y_level = get_quantiles(np.clip(u, 0.0, 1.0), pre_trt["o"], pre_trt["w"])

    return _Counterfactual(
        y_counterfactual=delta_ctrl + y_level,
        weights=pre_ctrl_by_unit["w"],
        group_size=int(pre_ctrl["w"].sum()),
    )


_SPECS: dict[str, _CounterfactualSpec] = {
    "changes_in_changes": _CounterfactualSpec(
        key="changes_in_changes",
        compute=_compute_changes_in_changes,
        references=("Athey and Imbens (2006), Econometrica 74(2), 431-497.",),
    ),
    "quantile_did": _CounterfactualSpec(
        key="quantile_did",
        compute=_compute_quantile_did,
        references=("Athey and Imbens (2006), Econometrica 74(2), 431-497, Section 3.3.",),
    ),
    "mean_did": _CounterfactualSpec(
        key="mean_did",
        compute=_compute_mean_did,
        references=("Thuysbaert (2007), working paper, as in qte::mdid.",),
    ),
    "distributional_did": _CounterfactualSpec(
        key="distributional_did",
        compute=_compute_distributional_did,
        references=("Callaway, Li and Oka (2018), Journal of Econometrics 206(2), 395-413.",),
    ),
}
