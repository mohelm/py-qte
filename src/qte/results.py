from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

import altair as alt
import numpy as np
import polars as pl
from great_tables import GT
from numpy.typing import ArrayLike
from rich.console import Group
from scipy.stats import chi2, norm
from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper

from qte.custom_types import ColumnName, FormulaRhs
from qte.names import (
    CI_LB_ID,
    CI_UB_ID,
    ESTIMATE_ID,
    MEAN_CONTROL_ID,
    MEAN_TREATED_ID,
    P_VALUE_ID,
    QUANTILE_CONTROL_VAL_ID,
    QUANTILE_ID,
    QUANTILE_TREATED_VAL_ID,
    SE_ID,
    TERM_ID,
    Z_ID,
)
from qte.presentation.altair_charts import make_plot
from qte.presentation.tables.gt_tables import (
    make_great_table,
    make_great_table_for_coefficients,
)
from qte.presentation.tables.rich_tables import (
    make_rich_table,
    make_rich_table_for_coefficients,
)
from qte.stats import get_ci

type AltairChart = alt.Chart | alt.LayerChart | alt.FacetChart


@dataclass
class InstrumentProbabilityResult:
    """Coefficient estimates from the logit model for the instrument probability.

    Use `tabulate`, `summarize` or `get_as_dataframe` to present the estimates.
    Instances are attached to the IV results as ``instrument_probability``.

    Attributes
    ----------
    coefficients : polars.DataFrame
        One row per model term with the estimate, standard error, z value and
        p value.
    instrument : ColumnName or None
        Instrument the model was estimated for.
    formula : FormulaRhs or None
        Right-hand side of the logit model. ``None`` for the intercept-only
        constant-probability model.
    n_obs : int or None
        Number of observations used to fit the model.
    df_model : int or None
        Model degrees of freedom, excluding the intercept.
    df_resid : int or None
        Residual degrees of freedom.
    converged : bool or None
        Whether the fitting algorithm converged.
    cov_type : str or None
        Covariance estimator used for the standard errors.
    method : str or None
        Fitting method reported by statsmodels.
    log_likelihood : float or None
        Maximized log-likelihood.
    log_likelihood_null : float or None
        Log-likelihood of the intercept-only model.
    pseudo_r_squared : float or None
        McFadden pseudo R-squared.
    llr : float or None
        Likelihood-ratio statistic against the intercept-only model.
    llr_p_value : float or None
        p value of the likelihood-ratio test.
    aic : float or None
        Akaike information criterion.
    bic : float or None
        Bayesian information criterion.
    """

    coefficients: pl.DataFrame = field(repr=False)
    instrument: ColumnName | None = None
    formula: FormulaRhs | None = None
    n_obs: int | None = None
    df_model: int | None = None
    df_resid: int | None = None
    converged: bool | None = None
    cov_type: str | None = None
    method: str | None = None
    log_likelihood: float | None = None
    log_likelihood_null: float | None = None
    pseudo_r_squared: float | None = None
    llr: float | None = None
    llr_p_value: float | None = None
    aic: float | None = None
    bic: float | None = None

    @classmethod
    def from_statsmodels(
        cls,
        result: GLMResultsWrapper,
        *,
        instrument: ColumnName | None = None,
        formula: FormulaRhs | None = None,
    ) -> "InstrumentProbabilityResult":
        """Collect coefficients and fit statistics from a fitted statsmodels model."""
        llf = result.llf
        llnull = result.llnull
        df_model = result.df_model
        llr = getattr(result, "llr", 2.0 * (llf - llnull))
        pseudo_r_squared = getattr(result, "prsquared", 1.0 - llf / llnull)
        llr_p_value = getattr(result, "llr_pvalue", None)
        if llr_p_value is None:
            llr_p_value = chi2.sf(llr, df_model) if df_model else None

        coefficients = pl.DataFrame(
            {
                TERM_ID: list(result.params.index),
                ESTIMATE_ID: result.params.to_numpy(),
                SE_ID: result.bse.to_numpy(),
                Z_ID: result.tvalues.to_numpy(),
                P_VALUE_ID: result.pvalues.to_numpy(),
            }
        )
        return cls(
            coefficients=coefficients,
            instrument=instrument,
            formula=formula,
            n_obs=result.nobs,
            df_model=df_model,
            df_resid=result.df_resid,
            converged=result.converged,
            cov_type=result.cov_type,
            method=result.method,
            log_likelihood=llf,
            log_likelihood_null=llnull,
            pseudo_r_squared=pseudo_r_squared,
            llr=llr,
            llr_p_value=llr_p_value,
            aic=result.aic,
            bic=result.bic,
        )

    @property
    def nobs(self) -> int | None:
        """Alias for ``n_obs`` matching statsmodels."""
        return self.n_obs

    @property
    def llf(self) -> float | None:
        """Alias for ``log_likelihood`` matching statsmodels."""
        return self.log_likelihood

    @property
    def llnull(self) -> float | None:
        """Alias for ``log_likelihood_null`` matching statsmodels."""
        return self.log_likelihood_null

    @property
    def prsquared(self) -> float | None:
        """Alias for ``pseudo_r_squared`` matching statsmodels."""
        return self.pseudo_r_squared

    @property
    def llr_pvalue(self) -> float | None:
        """Alias for ``llr_p_value`` matching statsmodels."""
        return self.llr_p_value

    def _make_table_header_content(self, alpha: float) -> dict[str, Any]:
        content: dict[str, Any] = {"Model": "Logit"}
        if self.instrument is not None:
            content["Instrument"] = self.instrument
        if self.formula is not None:
            content["Formula"] = self.formula
        if self.n_obs is not None:
            content["Observations"] = self.n_obs
        if self.df_model is not None:
            content["Df Model"] = f"{self.df_model:.0f}"
        if self.df_resid is not None:
            content["Df Residuals"] = f"{self.df_resid:.0f}"
        if self.method is not None:
            content["Method"] = self.method
        if self.converged is not None:
            content["Converged"] = self.converged
        if self.cov_type is not None:
            content["Covariance"] = self.cov_type
        if self.pseudo_r_squared is not None:
            content["Pseudo R²"] = f"{self.pseudo_r_squared:.3f}"
        if self.log_likelihood is not None:
            content["Log-Likelihood"] = f"{self.log_likelihood:.3f}"
        if self.log_likelihood_null is not None:
            content["LL-Null"] = f"{self.log_likelihood_null:.3f}"
        if self.llr_p_value is not None:
            content["LLR p-value"] = f"{self.llr_p_value:.3g}"
        if self.aic is not None:
            content["AIC"] = f"{self.aic:.3f}"
        if self.bic is not None:
            content["BIC"] = f"{self.bic:.3f}"
        content["Confidence Lvl"] = alpha
        return content

    def get_as_dataframe(self, alpha: float = 0.95) -> pl.DataFrame:
        """Return the coefficient estimates with confidence interval columns.

        Parameters
        ----------
        alpha : float, default=0.95
            Confidence level of the intervals.

        Returns
        -------
        polars.DataFrame
            Coefficient estimates with ``ci_lb`` and ``ci_ub`` columns.
        """
        alpha_half = (1 - alpha) / 2
        return self.coefficients.with_columns(
            (pl.col(ESTIMATE_ID) + norm.ppf(alpha_half) * pl.col(SE_ID)).alias(CI_LB_ID),
            (pl.col(ESTIMATE_ID) + norm.ppf(1 - alpha_half) * pl.col(SE_ID)).alias(CI_UB_ID),
        )

    def tabulate(self, alpha: float = 0.95) -> GT:
        """Build a Great Tables table of the coefficient estimates.

        Parameters
        ----------
        alpha : float, default=0.95
            Confidence level of the intervals.

        Returns
        -------
        great_tables.GT
            Formatted table of the instrument probability model.
        """
        return make_great_table_for_coefficients(
            self.get_as_dataframe(alpha),
            self._make_table_header_content(alpha),
            float_precision=3,
        )

    def summarize(self, alpha: float = 0.95) -> Group:
        """Render a rich text summary of the coefficient estimates.

        Parameters
        ----------
        alpha : float, default=0.95
            Confidence level of the intervals.

        Returns
        -------
        rich.console.Group
            Rich renderable for terminals and notebooks.
        """
        return make_rich_table_for_coefficients(
            self.get_as_dataframe(alpha),
            self._make_table_header_content(alpha),
            float_precision=3,
        )


@dataclass()
class _BasicQteResult:
    qtt: pl.DataFrame = field(repr=False)
    att: pl.DataFrame = field(repr=False)
    outcome: ColumnName
    group: tuple[str, ...] | None

    _exclude_from_qtt_tables: ClassVar[tuple[str, ...]] = (
        QUANTILE_TREATED_VAL_ID,
        QUANTILE_CONTROL_VAL_ID,
    )
    _exclude_from_att_tables: ClassVar[tuple[str, ...]] = (
        MEAN_CONTROL_ID,
        MEAN_TREATED_ID,
    )

    def __post_init__(self) -> None:
        self.qtt = self.qtt.sort(
            [
                *(
                    [self.group]
                    if isinstance(self.group, str)
                    else self.group
                    if self.group is not None
                    else []
                ),
                QUANTILE_ID,
            ]
        )
        self.att = self.att.sort(
            [self.group]
            if isinstance(self.group, str)
            else self.group
            if self.group is not None
            else []
        )

    def _make_table_header_content(self, alpha: float) -> dict[str, Any]:
        header_content = {"Outcome": self.outcome}
        header_content["Confidence Lvl"] = alpha
        return header_content

    def _make_chart_subtitle(self, alpha: float) -> list[str]:
        """Build the header lines shown under the chart title, like the table subtitle."""
        content = self._make_table_header_content(alpha)
        width = max(len(key) for key in content)
        return [f"{key}{' ' * (width - len(key) + 2)}: {value}" for key, value in content.items()]

    def _subset_quantiles(self, frame: pl.DataFrame, quantiles: ArrayLike | None) -> pl.DataFrame:
        """Keep only the requested quantiles (all of them when ``None``)."""
        if quantiles is None:
            return frame
        return frame.filter(pl.col(QUANTILE_ID).is_in(np.array(quantiles)))

    def plot(
        self,
        alpha: float = 0.95,
        quantiles: ArrayLike | None = None,
        highlight_quantiles: Sequence[float] | None = None,
    ) -> AltairChart:
        """Plot the quantile and average effects with confidence intervals.

        Parameters
        ----------
        alpha : float, default=0.95
            Confidence level of the intervals.
        quantiles : array_like, optional
            Quantiles to display. ``None`` shows all of them.
        highlight_quantiles : array_like, optional
            Quantiles at which to draw point markers on the full quantile line.
            ``None`` draws markers at every displayed quantile.

        Returns
        -------
        altair.Chart
            Altair chart of the estimates.
        """
        return make_plot(
            self._subset_quantiles(self.qtt, quantiles).with_columns(get_ci(alpha)),
            self.att.with_columns(get_ci(alpha)),
            group=self.group,
            subtitle=self._make_chart_subtitle(alpha),
            highlight_quantiles=highlight_quantiles,
        )

    def tabulate(self, alpha: float = 0.95, quantiles: ArrayLike | None = None) -> GT:
        """Build a Great Tables table of the estimates.

        Parameters
        ----------
        alpha : float, default=0.95
            Confidence level of the intervals.
        quantiles : array_like, optional
            Quantiles to display. ``None`` shows all of them.

        Returns
        -------
        great_tables.GT
            Formatted table of the estimates.
        """
        return make_great_table(
            self._subset_quantiles(self.qtt, quantiles)
            .drop(self._exclude_from_qtt_tables, strict=False)
            .with_columns(get_ci(alpha)),
            self.att.drop(self._exclude_from_att_tables, strict=False).with_columns(get_ci(alpha)),
            self.group,
            self._make_table_header_content(alpha),
        )

    def get_as_dataframe(self, alpha: float = 0.95) -> pl.DataFrame:
        """Return the quantile effects with confidence interval columns.

        Parameters
        ----------
        alpha : float, default=0.95
            Confidence level of the intervals.

        Returns
        -------
        polars.DataFrame
            Quantile effects with ``ci_lb`` and ``ci_ub`` columns.
        """
        return self.qtt.with_columns(get_ci(alpha))

    def summarize(self, alpha: float = 0.95, quantiles: ArrayLike | None = None) -> Group:
        """Render a rich text summary of the estimates.

        Parameters
        ----------
        alpha : float, default=0.95
            Confidence level of the intervals.
        quantiles : array_like, optional
            Quantiles to display. ``None`` shows all of them.

        Returns
        -------
        rich.console.Group
            Rich renderable for terminals and notebooks.
        """
        return make_rich_table(
            self._subset_quantiles(self.qtt, quantiles)
            .drop(self._exclude_from_qtt_tables, strict=False)
            .with_columns(get_ci(alpha)),
            self.att.drop(self._exclude_from_att_tables, strict=False).with_columns(get_ci(alpha)),
            self._make_table_header_content(alpha),
            float_precision=2,
            group=self.group,
        )
