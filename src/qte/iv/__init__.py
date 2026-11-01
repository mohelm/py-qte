"""Instrumental-variables estimators for quantile treatment effects.

The family provides the local quantile treatment effect for compliers, both
unconditional (`estimate_unconditional_local_effects`) and conditional on
covariates (`estimate_conditional_local_effects`), under the LATE assumptions.
It also provides the structural instrumental-variable quantile regression of
Chernozhukov and Hansen (`estimate_ivqr_effects`). The family lives apart from
the cross-sectional estimators because its estimands are not effects on the
whole population.
"""

from qte.iv.conditional import estimate_conditional_local_effects
from qte.iv.custom_types import Estimator, IvqrTreatment, WeightAdjustment
from qte.iv.ivqr import estimate_ivqr_effects
from qte.iv.results import (
    ConditionalLocalQteResult,
    IvqrResult,
    UnconditionalLocalQteResult,
)
from qte.iv.unconditional import estimate_unconditional_local_effects
from qte.results import InstrumentProbabilityResult

__all__ = [
    "ConditionalLocalQteResult",
    "Estimator",
    "InstrumentProbabilityResult",
    "IvqrResult",
    "IvqrTreatment",
    "UnconditionalLocalQteResult",
    "WeightAdjustment",
    "estimate_conditional_local_effects",
    "estimate_ivqr_effects",
    "estimate_unconditional_local_effects",
]
