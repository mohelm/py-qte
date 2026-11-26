"""Custom types for the regression-discontinuity estimators."""

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from qte.custom_types import ColumnName


class _Side(StrEnum):
    """Side of the cutoff."""

    LEFT = "left"
    RIGHT = "right"


class Kernel(StrEnum):
    """Kernel used to weight observations around the cutoff.

    Attributes
    ----------
    EPANECHNIKOV
        Epanechnikov kernel, ``0.75 * (1 - u**2)`` on ``[-1, 1]``. The default
        and the default of Stata's ``rddqte``.
    UNIFORM
        Uniform kernel on ``[-1, 1]``.
    TRIANGULAR
        Triangular kernel, ``1 - |u|`` on ``[-1, 1]``.
    GAUSSIAN
        Standard normal kernel.
    """

    EPANECHNIKOV = "epanechnikov"
    UNIFORM = "uniform"
    TRIANGULAR = "triangular"
    GAUSSIAN = "gaussian"

    def _compute_weights(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate the kernel at standardised distances ``u``."""
        if self is Kernel.EPANECHNIKOV:
            return np.where(np.abs(u) <= 1.0, 0.75 * (1.0 - u * u), 0.0)
        if self is Kernel.UNIFORM:
            return np.where(np.abs(u) <= 1.0, 0.5, 0.0)
        if self is Kernel.TRIANGULAR:
            return np.where(np.abs(u) <= 1.0, 1.0 - np.abs(u), 0.0)
        return np.exp(-0.5 * u * u) / np.sqrt(2.0 * np.pi)


class RddInference(StrEnum):
    """Inference method for the RDD distribution estimator.

    Attributes
    ----------
    BOOTSTRAP
        Case bootstrap, pointwise standard errors only.
    MULTIPLIER
        Multiplier bootstrap, pointwise standard errors plus a constant-width
        uniform band and process draws.
    """

    BOOTSTRAP = "bootstrap"
    MULTIPLIER = "multiplier"


@dataclass(frozen=True)
class RunningVariableConfig:
    """Running variable and the cutoff at which treatment status jumps.

    Use an instance as the ``running`` argument of
    `estimate_rdd_distribution_effects`, `plot_rdd_design` and
    `plot_rdd_treatment`.

    Parameters
    ----------
    name : ColumnName
        Name of the running (forcing) variable column.
    cutoff : float
        Value of the running variable at which the treatment probability jumps.
    """

    name: ColumnName
    cutoff: float

    def __post_init__(self) -> None:
        """Validate the cutoff."""
        if not np.isfinite(self.cutoff):
            raise ValueError("`cutoff` must be finite.")


@dataclass(frozen=True)
class LocalPolynomialConfig:
    """Local-polynomial smoothing settings.

    Use an instance as the ``local_polynomial_config`` argument of
    `estimate_rdd_distribution_effects`.

    Parameters
    ----------
    bandwidth : float
        Bandwidth for the local-linear distribution regressions. Must be
        positive and finite.
    kernel : `Kernel`, default=`Kernel.EPANECHNIKOV`
        Kernel used to weight observations around the cutoff.
    degree : int, default=1
        Degree of the local polynomial. ``1`` is local linear, ``2`` is local
        quadratic (the CHS bias-corrected fit).
    """

    bandwidth: float
    kernel: Kernel = Kernel.EPANECHNIKOV
    degree: int = 1

    def __post_init__(self) -> None:
        """Validate and normalise the smoothing settings."""
        if not np.isfinite(self.bandwidth) or self.bandwidth <= 0.0:
            raise ValueError("`bandwidth` must be a positive finite number.")
        if self.degree not in (1, 2):
            raise ValueError("`degree` must be 1 or 2.")
        object.__setattr__(self, "kernel", Kernel(self.kernel))
