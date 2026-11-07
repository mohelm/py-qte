"""Backwards-compatible alias of `qte.distributional_did`.

The module was renamed to `qte.distributional_did` because it covers both the nonlinear
difference-in-differences models of Athey and Imbens and the distributional ones of Callaway and
Li. Import from `qte.distributional_did` instead; this shim is kept so that old code keeps working.
"""

from qte.distributional_did import (
    BasePeriod,
    ControlGroup,
    DistributionalDidResult,
    DistributionalDidResults,
    Estimator,
    SamplingScheme,
    TrtGroupConfig,
    estimate_changes_in_changes_effects,
    estimate_quantile_did_effects,
)

__all__ = [
    "BasePeriod",
    "ControlGroup",
    "DistributionalDidResult",
    "DistributionalDidResults",
    "Estimator",
    "SamplingScheme",
    "TrtGroupConfig",
    "estimate_changes_in_changes_effects",
    "estimate_distributional_did_effects",
    "estimate_mean_did_effects",
    "estimate_quantile_did_effects",
]
