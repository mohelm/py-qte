import numpy as np
import polars as pl
from numpy.typing import NDArray

from qte.cross_sectional.custom_types import (
    CausalTarget,
    OutcomeRegressionConfig,
    resolve_grid,
)
from qte.cross_sectional.or_helpers import make_weights, predict_outcome_model
from qte.cross_sectional.results import _QteIntermediateResult
from qte.custom_types import ColumnName, DataFrame
from qte.helpers import _get_mean_differences, _get_quantile_differences
from qte.stats import estimate_outcome_model, get_quantiles


def compute_outcome_regression_effects(
    ds: DataFrame,
    outcome: ColumnName,
    treatment: ColumnName,
    qs: NDArray[np.float64] = (0.5,),  # type: ignore
    *,
    outcome_regression_config: OutcomeRegressionConfig,
    weights: ColumnName | None = None,
    target: CausalTarget = CausalTarget.QTE,
) -> _QteIntermediateResult:
    treated, control = (
        ds.filter(pl.col(treatment) == 1),
        ds.filter(pl.col(treatment) == 0),
    )
    grid = resolve_grid(outcome_regression_config.grid)

    # Estimate the outcome model on the controls
    ors_control = estimate_outcome_model(
        control,
        outcome,
        outcome_regression_config.formula,
        grid,
        weights,
        algorithm=outcome_regression_config.algorithm,
    )

    if target == CausalTarget.QTE:
        # For global treatment effects, we obtain for each observation the 'distribution' as
        # represented by the predicted quantiles in a stacked manner for the scenario under no
        # treatment from the outcome model estimated on the controls.
        preds_control = predict_outcome_model(ors_control, ds, flatten=True)
        # Align the sampling weights with this 'distribution'.
        sample_weights = make_weights(weights, ds, grid.shape[0])  # TODO: really tile??
        # Obtain the statistics of interest from that distribution.
        q_c = get_quantiles(qs, preds_control, sample_weights)
        mean_c = np.average(preds_control, weights=sample_weights)

        # Do the same for the scenario under trewatment. That is, estimate an outcome model on the
        # treated. Then obtain the predictions from that model for all observations, and lastly
        # compute the statistics of interest.
        ors_treated = estimate_outcome_model(
            treated,
            outcome,
            outcome_regression_config.formula,
            grid,
            weights,
            algorithm=outcome_regression_config.algorithm,
        )
        preds_treated = predict_outcome_model(ors_treated, ds, flatten=True)
        q_t = get_quantiles(qs, preds_treated, sample_weights)
        mean_t = np.average(preds_treated, weights=sample_weights)

    if target == CausalTarget.QTT:
        # For treatment effects on the treated, get (counterfactual) predictions for the treated's
        # data.
        preds_control = predict_outcome_model(ors_control, treated, flatten=True)
        # Get the corresponding weights.
        w_c = make_weights(weights, treated, grid.shape[0])
        # Compute the statistics of interest from the counterfactual outcomes and the corresponding
        # weights.
        q_c = get_quantiles(qs, preds_control, w_c)
        mean_c = np.average(preds_control, weights=w_c)

        # For the treatment effects on the treated, we can obtain the statistics of the treated
        # 'directly'.
        w_t = make_weights(weights, treated)
        y_t = treated[outcome].to_numpy()
        q_t = get_quantiles(qs, y_t, w=w_t)
        mean_t = np.average(y_t, weights=w_t)

    return _QteIntermediateResult(
        qtt=_get_quantile_differences(qs, q_t, q_c),
        att=_get_mean_differences(mean_t, mean_c),
    )
