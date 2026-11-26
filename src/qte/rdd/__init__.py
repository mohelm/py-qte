"""Quantile treatment effects in regression-discontinuity designs.

The family implements the local distribution-regression estimator of Frandsen,
Frölich and Melly (2012) for sharp and fuzzy RD designs. It targets the quantile
treatment effect for local compliers at the cutoff. Unlike the cross-sectional
and instrumental-variables families, the identifying variation comes from a
discontinuity in the treatment probability along a running variable.
"""

from qte.rdd.custom_types import Kernel, LocalPolynomialConfig, RddInference, RunningVariableConfig
from qte.rdd.distribution import estimate_rdd_distribution_effects
from qte.rdd.presentation import plot_rdd_design, plot_rdd_treatment
from qte.rdd.results import RddDistributionResult
from qte.rdd.simulate import (
    get_true_first_stage,
    get_true_local_effects,
    simulate_rdd_data,
)

__all__ = [
    "Kernel",
    "LocalPolynomialConfig",
    "RddDistributionResult",
    "RddInference",
    "RunningVariableConfig",
    "estimate_rdd_distribution_effects",
    "get_true_first_stage",
    "get_true_local_effects",
    "plot_rdd_design",
    "plot_rdd_treatment",
    "simulate_rdd_data",
]
