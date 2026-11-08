"""Counterfactual distributions for the distributional difference-in-differences models.

Each model is a 2x2 workhorse that turns the observed cells of a group-time comparison into a
counterfactual outcome for every unit of one of them. The workhorses are interchangeable: the
estimator engine takes one as an argument, so adding a model means adding a function here.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from qte._validation import check_units_are_observed_in_both_periods
from qte.stats import Ecdf


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


def _compute_changes_in_changes(
    pre_trt: NDArray, pre_ctrl: NDArray, post_ctrl: NDArray
) -> _Counterfactual:
    # Changes-in-changes maps the treated pre-treatment outcomes through the control group's
    # pre-to-post quantile transformation. Here we essentially compte the k^cic(y) expression from
    # the paper which takes the pre-treatment outcomes of the treated, evaluates the pre-treatment
    # cdf of the controls at those values and then applies the inverse of the post-treatment
    # control cdf to the results. This gives the outcome y' that someone with outcome y in the
    # pre-treatment period had experiment in the post-treatment period under no treatment.

    # First: evalulate F_(Y,00) at the t=pre and g=trt outcomes.
    rank_in_pre_ctrl_cdf = Ecdf.make(pre_ctrl["o"], pre_ctrl["w"]).evaluate(pre_trt["o"])

    # Second: evaluate F^(-1)_(Y,01) at the ranks from above.
    return _Counterfactual(
        y_counterfactual=Ecdf.make(post_ctrl["o"], post_ctrl["w"]).evaluate_inverse(
            rank_in_pre_ctrl_cdf
        ),
        weights=pre_trt["w"],
        group_size=int(pre_trt["w"].sum()),
    )


def _compute_quantile_did(
    pre_trt: NDArray, pre_ctrl: NDArray, post_ctrl: NDArray
) -> _Counterfactual:
    # Here, we want to obtain the counterfactual y' (in t=post, g=trt) from the pre-treatment y plus a change that
    # corresponds to the differences in the quantile function (inverse cdf) of the untreated
    # evaluated at the quantile that corresponds to the value y in the F_(Y,10)(.).

    # Step 1: Get the quantiles of all y-values of g=trt and t=pre.
    y = pre_trt["o"]
    rank_in_pre_trt_cdf = Ecdf.make(y, pre_trt["w"]).evaluate(y)

    # Step 2: Evaluate the inverses of F_(Y,00) and F_(Y,01) at these values.
    out_in_pre_ctrl = Ecdf.make(pre_ctrl["o"], pre_ctrl["w"]).evaluate_inverse(rank_in_pre_trt_cdf)
    out_in_post_ctrl = Ecdf.make(post_ctrl["o"], post_ctrl["w"]).evaluate_inverse(
        rank_in_pre_trt_cdf
    )

    return _Counterfactual(
        y_counterfactual=y + out_in_post_ctrl - out_in_pre_ctrl,
        weights=pre_trt["w"],
        group_size=int(pre_trt["w"].sum()),
    )


def _compute_mean_did(pre_trt: NDArray, pre_ctrl: NDArray, post_ctrl: NDArray) -> _Counterfactual:
    # Mean difference-in-differences shifts the whole treated pre-treatment distribution by the
    # change in the control group's mean.
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
    # In distributional difference-in-differences, the counterfactual (empricial) distribution is found as the
    # (empirical) distribution of the changes in the outcome for the untreated group between the pre- and post-treatment
    # period PLUS an outcome level that is found by taking the control groups pre-treatment period
    # outcome, finding its quantile in the corresponding cdf, and obtaining an outcome level with
    # this quantile from the pre-treatment period cdf of the treated group. In other words, we need
    # Delta Y + F_[Y_pre, D=1]^[-1] compose F[Y_pre,D=0]](Y)
    check_units_are_observed_in_both_periods(pre_ctrl["u"], post_ctrl["u"])

    # First: Get the changes. A change needs a unit's outcome in both periods.
    pre_ctrl_by_unit = pre_ctrl[np.argsort(pre_ctrl["u"])]
    post_ctrl_by_unit = post_ctrl[np.argsort(post_ctrl["u"])]
    delta_ctrl = post_ctrl_by_unit["o"] - pre_ctrl_by_unit["o"]

    # Second: Get the levels.
    # Obtain the quantiles.
    rank = Ecdf.make(pre_ctrl["o"], pre_ctrl["w"]).evaluate(pre_ctrl_by_unit["o"])
    # Obtain the corresponding levels with the treatment group's pre-treatment period cdf.
    y_level = Ecdf.make(pre_trt["o"], pre_trt["w"]).evaluate_inverse(rank)

    return _Counterfactual(
        y_counterfactual=delta_ctrl + y_level,
        weights=pre_ctrl_by_unit["w"],
        group_size=int(pre_ctrl["w"].sum()),
    )
