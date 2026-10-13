"""Instrumental-variables estimators for quantile treatment effects.

The family provides the local quantile treatment effect for compliers, both
unconditional (`estimate_unconditional_local_effects`) and conditional on
covariates (`estimate_conditional_local_effects`), under the LATE assumptions.
It lives apart from the cross-sectional estimators because the estimand is the
effect on compliers, not on the whole population.
"""

from qte.iv.conditional import estimate_conditional_local_effects
from qte.iv.custom_types import Estimator, WeightAdjustment
from qte.iv.results import ConditionalLocalQteResult, UnconditionalLocalQteResult
from qte.iv.unconditional import estimate_unconditional_local_effects

__all__ = [
    "ConditionalLocalQteResult",
    "Estimator",
    "UnconditionalLocalQteResult",
    "WeightAdjustment",
    "estimate_conditional_local_effects",
    "estimate_unconditional_local_effects",
]
