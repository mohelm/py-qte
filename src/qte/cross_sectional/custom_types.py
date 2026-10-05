"""Custom types for the cross-sectional estimators."""

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import ArrayLike, NDArray

from qte.constants import _make_q
from qte.custom_types import FormulaRhs
from qte.quantile_regression import QuantileRegressionAlgorithms


class CausalTarget(StrEnum):
    """Causal estimand a cross-sectional estimator targets.

    Use a member as the ``target`` argument of the cross-sectional estimators,
    for example ``CausalTarget.QTT``.

    Attributes
    ----------
    QTE
        Quantile treatment effect for the whole population.
    QTT
        Quantile treatment effect on the treated.

    Examples
    --------
    Target the effect on the treated instead of the whole population:

    >>> from qte.cross_sectional import CausalTarget, estimate_ipw_effects
    >>> from qte.datasets import load_lalonde
    >>> res = estimate_ipw_effects(
    ...     load_lalonde(),
    ...     "re78",
    ...     "treat",
    ...     propensity_score_formula="age + education",
    ...     target=CausalTarget.QTT,
    ... )
    """

    QTE = "qte"
    QTT = "qtt"


class Estimator(StrEnum):
    """Cross-sectional estimator that produced a result.

    This is an output field: read it from ``result.estimator`` to see how an
    estimate was produced, and compare it against a member.

    Attributes
    ----------
    UNADJUSTED
        Difference in group quantiles without covariate adjustment.
    IPW
        Inverse probability weighting.
    OR
        Outcome regression.
    AIPW
        Augmented inverse probability weighting.

    Examples
    --------
    Inspect which estimator produced a result:

    >>> from qte.cross_sectional import Estimator, estimate_unadjusted_effects
    >>> from qte.datasets import load_lalonde
    >>> res = estimate_unadjusted_effects(load_lalonde(), "re78", "treat")
    >>> res.estimator is Estimator.UNADJUSTED
    True
    """

    UNADJUSTED = "unadjusted"
    IPW = "ipw"
    OR = "or"
    AIPW = "aipw"


@dataclass(frozen=True)
class OutcomeRegressionConfig:
    """Configuration of the outcome (quantile) regression step.

    Parameters
    ----------
    formula : FormulaRhs
        Right-hand side of the outcome regression formula.
    grid : int or array_like, default=100
        Quantile grid used to fit the outcome regression. An integer ``k`` is
        expanded to the ``k - 1`` interior quantiles; an explicit sequence is
        used as given.
    algorithm : `QuantileRegressionAlgorithms`, default=`QuantileRegressionAlgorithms.PREPROCESSING`
        Quantile regression algorithm used to fit the grid.
    """

    formula: FormulaRhs
    grid: int | ArrayLike = 100
    algorithm: QuantileRegressionAlgorithms = QuantileRegressionAlgorithms.PREPROCESSING


def make_outcome_regression_config(
    value: FormulaRhs | OutcomeRegressionConfig,
) -> OutcomeRegressionConfig:
    """Coerce a formula shorthand into an `OutcomeRegressionConfig`."""
    return value if isinstance(value, OutcomeRegressionConfig) else OutcomeRegressionConfig(value)


def resolve_grid(grid: int | ArrayLike) -> NDArray[np.float64]:
    """Expand an integer grid size into quantiles, or pass a sequence through."""
    if isinstance(grid, (int, np.integer)):
        return _make_q(int(grid))
    return np.asarray(grid, dtype=np.float64)
