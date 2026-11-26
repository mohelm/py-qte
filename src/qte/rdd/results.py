from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import ArrayLike, NDArray

from qte.custom_types import ColumnName
from qte.names import CI_LB_ID, CI_UB_ID, EFFECT_ID, SE_ID
from qte.presentation.altair_charts import make_plot
from qte.rdd.custom_types import LocalPolynomialConfig, RddInference, RunningVariableConfig
from qte.rdd.presentation import plot_rdd_design, plot_rdd_treatment
from qte.results import AltairChart, _BasicQteResult
from qte.stats import get_ci


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
    inference : `RddInference`
        Inference engine used for the standard errors.
    uniform_sup_draws : numpy.ndarray or None
        Multiplier draws of the supremum of the QTE process, i.e. the
        CHS/Stata-style constant-width uniform half-width. ``None`` for
        case-bootstrap inference.
    uniform_sup_t_draws : numpy.ndarray or None
        Studentised multiplier draws of the supremum of the QTE process.
        ``None`` for case-bootstrap inference.
    """

    treatment: ColumnName
    running: ColumnName
    cutoff: float
    local_polynomial_config: LocalPolynomialConfig
    outcome_grid_size: int
    first_stage: float
    n_obs: int
    inference: RddInference = RddInference.BOOTSTRAP
    uniform_sup_draws: NDArray[np.float64] | None = field(default=None, repr=False)
    uniform_sup_t_draws: NDArray[np.float64] | None = field(default=None, repr=False)

    def _make_table_header_content(self, alpha: float) -> dict[str, Any]:
        """Add the RD design settings to the shared table and chart header."""
        return {
            "Outcome": self.outcome,
            "Treatment": self.treatment,
            "Running": self.running,
            "Cutoff": self.cutoff,
            "Bandwidth": self.local_polynomial_config.bandwidth,
            "Kernel": self.local_polynomial_config.kernel.value,
            "Degree": self.local_polynomial_config.degree,
            "Inference": self.inference.value,
            "First Stage": f"{self.first_stage:.3f}",
            "Observations": f"{self.n_obs:,}",
            "Confidence Lvl": alpha,
        }

    def uniform_half_width(self, alpha: float = 0.95) -> float:
        """Return the CHS/Stata-style constant half-width of the uniform band."""
        if self.uniform_sup_draws is None:
            raise ValueError("Uniform bands require `inference=RddInference.MULTIPLIER`.")
        return float(np.quantile(self.uniform_sup_draws, alpha))

    def uniform_critical_value(self, alpha: float = 0.95) -> float:
        """Alias for `uniform_half_width`."""
        return self.uniform_half_width(alpha)

    def uniform_sup_t_critical_value(self, alpha: float = 0.95) -> float:
        """Return the studentised sup-t critical value."""
        if self.uniform_sup_t_draws is None:
            raise ValueError("Uniform bands require `inference=RddInference.MULTIPLIER`.")
        return float(np.quantile(self.uniform_sup_t_draws, alpha))

    def uniform_sup_t_half_width(self, alpha: float = 0.95) -> NDArray[np.float64]:
        """Return the quantile-specific studentised sup-t half-widths."""
        return self.uniform_sup_t_critical_value(alpha) * self.qtt[SE_ID].to_numpy()

    def get_as_dataframe(self, alpha: float = 0.95, *, band: str = "pointwise") -> pl.DataFrame:
        """Return the quantile effects with confidence-bound columns."""
        if band == "pointwise":
            return super().get_as_dataframe(alpha)
        if band == "uniform":
            half_width = self.uniform_half_width(alpha)
            return self.qtt.with_columns(
                (pl.col(EFFECT_ID) - half_width).alias(CI_LB_ID),
                (pl.col(EFFECT_ID) + half_width).alias(CI_UB_ID),
            )
        if band == "sup-t":
            critical_value = self.uniform_sup_t_critical_value(alpha)
            return self.qtt.with_columns(
                (pl.col(EFFECT_ID) - critical_value * pl.col(SE_ID)).alias(CI_LB_ID),
                (pl.col(EFFECT_ID) + critical_value * pl.col(SE_ID)).alias(CI_UB_ID),
            )
        raise ValueError("`band` must be 'pointwise', 'uniform' or 'sup-t'.")

    def plot(
        self,
        alpha: float = 0.95,
        quantiles: ArrayLike | None = None,
        highlight_quantiles: Sequence[float] | None = None,
        *,
        band: str = "pointwise",
    ) -> AltairChart:
        """Plot the quantile and average effects with confidence bounds."""
        if band == "pointwise":
            return super().plot(alpha, quantiles, highlight_quantiles)
        qtt = self._subset_quantiles(self.qtt, quantiles)
        if band == "uniform":
            half_width = self.uniform_half_width(alpha)
            qtt = qtt.with_columns(
                (pl.col(EFFECT_ID) - half_width).alias(CI_LB_ID),
                (pl.col(EFFECT_ID) + half_width).alias(CI_UB_ID),
            )
        elif band == "sup-t":
            critical_value = self.uniform_sup_t_critical_value(alpha)
            qtt = qtt.with_columns(
                (pl.col(EFFECT_ID) - critical_value * pl.col(SE_ID)).alias(CI_LB_ID),
                (pl.col(EFFECT_ID) + critical_value * pl.col(SE_ID)).alias(CI_UB_ID),
            )
        else:
            raise ValueError("`band` must be 'pointwise', 'uniform' or 'sup-t'.")
        att = self.att.with_columns(get_ci(alpha))
        return make_plot(
            qtt,
            att,
            group=self.group,
            subtitle=self._make_chart_subtitle(alpha),
            highlight_quantiles=highlight_quantiles,
        )

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
