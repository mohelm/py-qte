"""Instrumental-variables estimators for quantile treatment effects.

The family provides the local quantile treatment effect for compliers, both
unconditional (`estimate_unconditional_local_effects`) and conditional on
covariates (`estimate_conditional_local_effects`), under the LATE assumptions.
It also provides the structural instrumental-variable quantile regression of
Chernozhukov and Hansen (`estimate_ivqr_effects`) and
`perform_instrument_validity_test`, which checks the testable implication of
those assumptions for a binary treatment and a binary instrument. The family
lives apart from the unconfounded estimators because its estimands are not
effects on the whole population.
"""

from qte.iv.conditional import estimate_conditional_local_effects
from qte.iv.custom_types import Estimator, IvqrTreatment, ValidityWeighting, WeightAdjustment
from qte.iv.ivqr import estimate_ivqr_effects
from qte.iv.results import (
    ConditionalLocalQteResult,
    InstrumentValidityResult,
    IvqrResult,
    UnconditionalLocalQteResult,
)
from qte.iv.unconditional import estimate_unconditional_local_effects
from qte.iv.validity import perform_instrument_validity_test
from qte.results import InstrumentProbabilityResult

__all__ = [
    "ConditionalLocalQteResult",
    "Estimator",
    "InstrumentProbabilityResult",
    "InstrumentValidityResult",
    "IvqrResult",
    "IvqrTreatment",
    "UnconditionalLocalQteResult",
    "ValidityWeighting",
    "WeightAdjustment",
    "estimate_conditional_local_effects",
    "estimate_ivqr_effects",
    "estimate_unconditional_local_effects",
    "perform_instrument_validity_test",
]
