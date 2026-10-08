"""Instrumental-variables estimators for quantile treatment effects.

The family currently provides the local quantile treatment effect for compliers
(`estimate_local_effects`), Abadie's kappa-weighting under the LATE
assumptions. It lives apart from the cross-sectional estimators because the
estimand is the effect on compliers, not on the whole population.
"""

from qte.iv.local import estimate_local_effects
from qte.iv.results import LocalQteResult

__all__ = [
    "LocalQteResult",
    "estimate_local_effects",
]
