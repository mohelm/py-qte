"""Result container for the local quantile treatment effect estimator."""

from dataclasses import dataclass, field
from typing import Any

import polars as pl
from scipy.stats import norm

from qte.custom_types import ColumnName, FormulaRhs
from qte.iv.custom_types import Estimator, WeightAdjustment
from qte.results import InstrumentProbabilityResult, _BasicQteResult


@dataclass
class UnconditionalLocalQteResult(_BasicQteResult):
    """Result of a local quantile treatment effect estimation.

    Use `plot`, `tabulate`, `summarize` or `get_as_dataframe` to present the
    estimates. Instances are returned by `estimate_unconditional_local_effects`; the
    constructor is not part of the public API.

    Attributes
    ----------
    qtt : polars.DataFrame
        Quantile-specific effects, one row per quantile.
    att : polars.DataFrame
        Average effects for compliers (the local average treatment effect).
    outcome : ColumnName
        Outcome variable the effect was estimated for.
    group : tuple[str, ...] or None
        Columns the estimates are grouped by.
    complier_share : float
        Estimated share of compliers, ``E[kappa]``.
    first_stage : float
        Difference in treatment take-up between instrument arms,
        ``E[D | Z = 1] - E[D | Z = 0]``.
    estimator : `Estimator`
        Weighting scheme used to estimate the effects.
    instrument_probability : `InstrumentProbabilityResult`
        Fitted logit model for the instrument probability. Without an
        ``instrument_probability_formula`` this is the intercept-only model.
    """

    complier_share: float
    first_stage: float
    estimator: Estimator
    instrument_probability: InstrumentProbabilityResult

    def _make_table_header_content(self, alpha: float) -> dict[str, Any]:
        """Add the weighting scheme to the shared table and chart header."""
        return {
            "Outcome": self.outcome,
            "Estimator": self.estimator.value,
            "Confidence Lvl": alpha,
        }


@dataclass
class ConditionalLocalQteResult(_BasicQteResult):
    """Result of a conditional local quantile treatment effect estimation.

    Instances are returned by `estimate_conditional_local_effects`; the
    constructor is not part of the public API.

    Attributes
    ----------
    qtt : polars.DataFrame
        Quantile-specific effects, one row per quantile. ``q_t``/``q_c`` are the
        counterfactual quantiles at the covariate reference (every covariate at
        its baseline), i.e. the ``Constant`` row of Abadie, Angrist & Imbens
        (2002), Table III, plus/minus the treatment effect.
    att : polars.DataFrame
        Average effects for compliers (the local average treatment effect), with
        ``m_t``/``m_c`` the complier treated/control mean outcomes.
    outcome : ColumnName
        Outcome variable the effect was estimated for.
    group : tuple[str, ...] or None
        Columns the estimates are grouped by.
    complier_share : float
        Estimated share of compliers.
    first_stage : float
        Difference in treatment take-up between instrument arms,
        ``E[D | Z = 1] - E[D | Z = 0]``.
    weight_adjustment : `WeightAdjustment`
        How the nonnegative complier weights were formed.
    instrument_probability : `InstrumentProbabilityResult`
        Fitted logit model for the instrument probability. Without an
        ``instrument_probability_formula`` this is the intercept-only model.
    """

    complier_share: float
    first_stage: float
    weight_adjustment: WeightAdjustment
    instrument_probability: InstrumentProbabilityResult

    def _make_table_header_content(self, alpha: float) -> dict[str, Any]:
        """Add the weight adjustment to the shared table and chart header."""
        return {
            "Outcome": self.outcome,
            "Weights": self.weight_adjustment.value,
            "Confidence Lvl": alpha,
        }


@dataclass
class IvqrResult(_BasicQteResult):
    """Result of an instrumental-variable quantile regression.

    Use `plot`, `tabulate`, `summarize` or `get_as_dataframe` to present the
    estimates. Instances are returned by `estimate_ivqr_effects`; the
    constructor is not part of the public API.

    Attributes
    ----------
    qtt : polars.DataFrame
        Quantile-specific structural treatment effects, one row per quantile.
    att : polars.DataFrame
        Average of the IVQR quantile effects over the requested quantiles, with
        its analytical standard error.
    outcome : ColumnName
        Outcome variable the model was estimated for.
    group : tuple[str, ...] or None
        Columns the estimates are grouped by. Always ``None`` for IVQR.
    covariate_coefficients : polars.DataFrame
        Intercept and covariate coefficients, one row per term and quantile.
    treatment : ColumnName
        Endogenous treatment variable the model was estimated for.
    instruments : tuple of str
        Instrument names.
    covariates : FormulaRhs
        Right-hand side of the covariate formula.
    n_obs : int
        Number of observations used to fit the model.
    tsls : tuple of float
        Separate 2SLS average IV effect and its standard error, reported as
        ``(estimate, standard_error)``.
    """

    covariate_coefficients: pl.DataFrame = field(repr=False)
    treatment: ColumnName
    instruments: tuple[str, ...]
    covariates: FormulaRhs
    n_obs: int
    tsls: tuple[float, float]

    def _make_table_header_content(self, alpha: float) -> dict[str, Any]:
        """Build the header shared by the table and chart."""
        alpha_half = (1.0 - alpha) / 2.0
        tsls_lower = self.tsls[0] + norm.ppf(alpha_half) * self.tsls[1]
        tsls_upper = self.tsls[0] + norm.ppf(1.0 - alpha_half) * self.tsls[1]
        return {
            "Outcome": self.outcome,
            "Treatment": self.treatment,
            "Instruments": ", ".join(self.instruments),
            "Observations": self.n_obs,
            "2SLS": f"{self.tsls[0]:.3f} [{tsls_lower:.3f}, {tsls_upper:.3f}]",
            "Confidence Lvl": alpha,
        }
