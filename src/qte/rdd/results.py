"""Result container for the distribution-regression RD estimator."""

from dataclasses import dataclass
from typing import Any

import polars as pl

from qte.custom_types import ColumnName
from qte.rdd.custom_types import LocalPolynomialConfig, RunningVariableConfig
from qte.rdd.presentation import plot_rdd_design, plot_rdd_treatment
from qte.results import AltairChart, _BasicQteResult


@dataclass
class RddDistributionResult(_BasicQteResult):
    """Result of a local distribution-regression RD estimation.

    Use `plot`, `tabulate`, `summarize` or `get_as_dataframe` to present the
    estimates. Instances are returned by `estimate_rdd_distribution_effects`;
    the constructor is not part of the public API.

    The quantile effects are the differences between the complier quantiles of
    the treated and untreated potential outcomes at the cutoff. The average
    effect is the local average treatment effect for the same compliers.

    Attributes
    ----------
    qtt : polars.DataFrame
        Quantile-specific effects, one row per quantile.
    att : polars.DataFrame
        Local average treatment effect for compliers.
    outcome : ColumnName
        Outcome variable the effect was estimated for.
    group : tuple[str, ...] or None
        Columns the estimates are grouped by. Always ``None`` for this
        estimator.
    treatment : ColumnName
        Treatment variable the effect was estimated for.
    running : ColumnName
        Running (forcing) variable.
    cutoff : float
        Value of the running variable at which treatment status jumps.
    local_polynomial_config : `LocalPolynomialConfig`
        Bandwidth and kernel used for the local-linear distribution
        regressions.
    outcome_grid_size : int
        Number of outcome-grid points used for the distribution regressions.
    first_stage : float
        Estimated jump in the treatment probability at the cutoff. In a fuzzy
        design this is the local complier share.
    n_obs : int
        Number of observations used to estimate the effects.
    """

    treatment: ColumnName
    running: ColumnName
    cutoff: float
    local_polynomial_config: LocalPolynomialConfig
    outcome_grid_size: int
    first_stage: float
    n_obs: int

    def _make_table_header_content(self, alpha: float) -> dict[str, Any]:
        """Add the RD design settings to the shared table and chart header."""
        return {
            "Outcome": self.outcome,
            "Treatment": self.treatment,
            "Running": self.running,
            "Cutoff": self.cutoff,
            "Bandwidth": self.local_polynomial_config.bandwidth,
            "Kernel": self.local_polynomial_config.kernel.value,
            "First Stage": f"{self.first_stage:.3f}",
            "Observations": f"{self.n_obs:,}",
            "Confidence Lvl": alpha,
        }

    def plot_design(self, ds: pl.DataFrame, *, n_bins: int = 30) -> AltairChart:
        """Plot the outcome against the running variable around the cutoff.

        Parameters
        ----------
        ds : polars.DataFrame
            Data the model was estimated on.
        n_bins : int, default=30
            Number of bins on each side of the cutoff.

        Returns
        -------
        altair.Chart
            Binned outcomes and local-linear fits around the cutoff.
        """
        return plot_rdd_design(
            ds,
            self.outcome,
            RunningVariableConfig(self.running, self.cutoff),
            n_bins=n_bins,
            local_polynomial_config=self.local_polynomial_config,
        )

    def plot_treatment(self, ds: pl.DataFrame, *, n_bins: int = 30) -> AltairChart:
        """Plot the treatment probability against the running variable.

        Parameters
        ----------
        ds : polars.DataFrame
            Data the model was estimated on.
        n_bins : int, default=30
            Number of bins on each side of the cutoff.

        Returns
        -------
        altair.Chart
            Binned treatment rates and local-linear fits around the cutoff.
        """
        return plot_rdd_treatment(
            ds,
            self.treatment,
            RunningVariableConfig(self.running, self.cutoff),
            n_bins=n_bins,
            local_polynomial_config=self.local_polynomial_config,
        )
