from dataclasses import dataclass
from functools import partial

import numpy as np
import polars as pl
import statsmodels.api as sm
from formulaic import Formula
from numpy.typing import ArrayLike, NDArray

from qte.bootstrap import (
    BootstrapConfig,
    _attach_standard_errors,
    _make_bootstrap_config,
    _make_bootstrap_runs,
)
from qte.constants import MEDIAN
from qte.custom_types import ColumnName, FormulaRhs
from qte.helpers import _as_iterable_array, _get_mean_differences, _get_quantile_differences
from qte.iv.common import (
    _DEFAULT_TRIM,
    _PROPENSITY_ID,
    _conditional_takeup_diff,
    _estimate_instrument_propensity,
)
from qte.iv.custom_types import WeightAdjustment
from qte.iv.results import ConditionalLocalQteResult
from qte.quantile_regression import QuantileRegression
from qte.results import InstrumentProbabilityResult

_KAPPA_ID = "_kappa"


@dataclass(frozen=True)
class _ConditionalLocalQteIntermediateResult:
    """Point estimates from one (bootstrap) draw."""

    qtt: pl.DataFrame
    att: pl.DataFrame
    complier_share: float
    first_stage: float
    instrument_probability: InstrumentProbabilityResult | None = None


def _compute_kappa(
    treatment: ColumnName, instrument: ColumnName, propensity: ColumnName
) -> pl.Expr:
    """Polars expression for Abadie-Angrist-Imbens ``1 - D(1-Z)/(1-p) - (1-D)Z/p``."""
    d, z, p = pl.col(treatment), pl.col(instrument), pl.col(propensity)
    return 1.0 - d * (1.0 - z) / (1.0 - p) - (1.0 - d) * z / p


def _adjust_kappa_via_projection(
    estimation_data: pl.DataFrame,
    weight_formula: FormulaRhs,
    treatment: ColumnName,
) -> NDArray[np.float64]:
    """Linear projection of kappa on the weight design, separately per arm."""
    design = Formula.from_spec(f"~ {weight_formula}").get_model_matrix(
        estimation_data, output="numpy"
    )
    kappa = estimation_data[_KAPPA_ID].to_numpy()
    treatment_values = estimation_data[treatment].to_numpy()
    projected = np.zeros(kappa.shape, dtype=np.float64)
    for value in np.unique(treatment_values):
        mask = treatment_values == value
        projected[mask] = sm.OLS(kappa[mask], design[mask]).fit().fittedvalues
    return projected


def _adjust_kappa_via_plugin(
    estimation_data: pl.DataFrame,
    treatment: ColumnName,
    instrument: ColumnName,
    weight_formula: FormulaRhs,
) -> NDArray[np.float64]:
    """Plug-in estimate of ``E[kappa | U]`` via ``nu_0(U) = E[Z | Y, D, X]``.

    ``nu`` is fitted by OLS on the weight design separately for the treated and
    the untreated (as in Abadie, Angrist & Imbens, 2002); it is then plugged
    into the exact kappa identity, ``1 - D(1-nu)/(1-p) - (1-D)nu/p``.
    """
    d = estimation_data[treatment].to_numpy()
    z = estimation_data[instrument].to_numpy()
    propensity = estimation_data[_PROPENSITY_ID].to_numpy()
    design = Formula.from_spec(f"~ {weight_formula}").get_model_matrix(
        estimation_data, output="numpy"
    )
    nu = np.empty(z.shape, dtype=np.float64)
    for value in np.unique(d):
        mask = d == value
        nu[mask] = sm.OLS(z[mask], design[mask]).fit().fittedvalues
    return 1.0 - d * (1.0 - nu) / (1.0 - propensity) - (1.0 - d) * nu / propensity


def _compute_conditional_local_effects(
    ds: pl.DataFrame,
    outcome: ColumnName,
    treatment: ColumnName,
    instrument: ColumnName,
    qs: NDArray[np.float64],
    *,
    covariates: FormulaRhs,
    weight_formula: FormulaRhs,
    instrument_probability_formula: FormulaRhs | None = None,
    weight_adjustment: WeightAdjustment = WeightAdjustment.PROJECTION,
    weights: ColumnName | None = None,
    trim: float = _DEFAULT_TRIM,
    with_instrument_probability: bool = False,
) -> _ConditionalLocalQteIntermediateResult:
    """Compute conditional complier quantiles and means without bootstrap inference."""
    qs = _as_iterable_array(qs)
    if weights is None:
        weights = "_w"
        ds = ds.with_columns(pl.lit(1.0).alias(weights))

    propensity_fit = _estimate_instrument_propensity(
        ds,
        instrument,
        instrument_probability_formula,
        weights,
        with_model=with_instrument_probability,
    )
    propensity = propensity_fit.values
    estimation_data = ds.with_columns(pl.Series(_PROPENSITY_ID, propensity))
    if trim > 0.0:
        estimation_data = estimation_data.filter(
            (pl.col(_PROPENSITY_ID) >= trim) & (pl.col(_PROPENSITY_ID) <= 1.0 - trim)
        )
    estimation_data = estimation_data.with_columns(
        _compute_kappa(treatment, instrument, _PROPENSITY_ID).alias(_KAPPA_ID)
    )

    if weight_adjustment is WeightAdjustment.PROJECTION:
        kappa_nu = _adjust_kappa_via_projection(estimation_data, weight_formula, treatment)
    else:
        kappa_nu = _adjust_kappa_via_plugin(estimation_data, treatment, instrument, weight_formula)

    estimation_data_with_pos_kappa = estimation_data.with_columns(
        pl.Series(_KAPPA_ID, kappa_nu)
    ).filter(pl.col(_KAPPA_ID) > 0)
    formula = f"{outcome} ~ {treatment} + {covariates}"
    fit = QuantileRegression(formula, estimation_data_with_pos_kappa).fit(qs, weights=_KAPPA_ID)
    treatment_index = fit.regressor_names.index(treatment)
    intercept_index = fit.regressor_names.index("Intercept")

    # The intercept is the control/complier quantile at the covariate reference
    # (every covariate at its baseline), matching the ``Constant`` row of
    # Abadie, Angrist & Imbens (2002), Table III; adding the treatment
    # coefficient gives the treated quantile at the same reference.
    q_c = fit.coefficients[intercept_index]
    qtt = _get_quantile_differences(qs, q_c + fit.coefficients[treatment_index], q_c)

    att = _get_mean_differences(
        *estimation_data_with_pos_kappa.group_by(treatment)
        .agg(((pl.col(outcome) * pl.col(_KAPPA_ID)).sum() / pl.col(_KAPPA_ID).sum()).alias("_mean"))
        .sort(treatment, descending=True)["_mean"]
        .to_numpy()
    )

    return _ConditionalLocalQteIntermediateResult(
        qtt=qtt,
        att=att,
        complier_share=float(np.mean(kappa_nu)),
        first_stage=_conditional_takeup_diff(estimation_data, treatment, instrument, weights),
        instrument_probability=propensity_fit.model_result,
    )


def estimate_conditional_local_effects(
    ds: pl.DataFrame,
    outcome: ColumnName,
    treatment: ColumnName,
    instrument: ColumnName,
    qs: ArrayLike = MEDIAN,
    *,
    covariates: FormulaRhs,
    weight_formula: FormulaRhs | None = None,
    instrument_probability_formula: FormulaRhs | None = None,
    weight_adjustment: WeightAdjustment = WeightAdjustment.PROJECTION,
    weights: ColumnName | None = None,
    trim: float = _DEFAULT_TRIM,
    bootstrap_config: BootstrapConfig | int = 100,
) -> ConditionalLocalQteResult:
    r"""Estimate conditional quantile treatment effects for compliers as in Abadie et al. (2002).

    The model postulates that
    $Q_{\theta}(Y \mid X, D, D_{1} > D_{0}) = \alpha_{\theta} D + X' \beta_{\theta}$.
    $\theta$ indexes the quantile of $Y$. $D$ is binary, one-dimensional and endogenous, which is
    why it needs to be instrumented.

    $\alpha$ can, in principle, be obtained from the following quantile regression problem:
    $$
    (\alpha_{\theta}, \beta_{\theta})
    = \arg\min_{\alpha, \beta} E\big[\kappa \, \rho_{\theta}(Y - \alpha D - X' \beta)\big],
    $$

    where $\kappa$ is the _signed_ Abadie's kappa weight
    ($\kappa = 1 - \frac{D\,(1 - Z)}{1 - \pi_0(X)} - \frac{(1 - D)\,Z}{\pi_0(X)}$ with
    $\pi_0(X) = P(Z = 1 \mid X)$). This problem is computationally challenging since the weights can
    be negative. Thus, the weights are replaced with another set of weights, where
    $U = (Y, D, X)$ and $\nu_0(U) = E[Z \mid U]$:

    $$
    \tilde{\kappa}(U) = E[\kappa \mid U]
    = 1 - \frac{D\,(1 - \nu_0(U))}{1 - \pi_0(X)}
      - \frac{(1 - D)\,\nu_0(U)}{\pi_0(X)},
    $$

    which can be shown to be nonnegative across the board. The `weight_formula` gives the design of
    the first step and `weight_adjustment` the way it is used: `PLUGIN` approximates $\nu_0(U)$ and
    plugs it into the identity above, `PROJECTION` projects $\kappa$ on the design directly.

    Standard errors are found via a case bootstrap.

    The estimand is the coefficient on the treatment in a quantile regression of
    the outcome on the treatment and the covariates, for compliers. Notice that the conditional (on
    $X$) effect (on the compliers) generally does not coincide with the unconditional effect (on the
    compliers). This is because here we identify
    $Q_{\theta}(Y \mid D=1, X=x) - Q_{\theta}(Y \mid D=0, X=x) = \alpha_{\theta}$
    (for the compliers), but if we integrate out $X$ on the left hand side, the resulting
    functional _is not_ comprised of the marginal distributions of $Y$ given the treatment status
    for the compliers, since quantiles are not linear operators.

    Parameters
    ----------
    ds : polars.DataFrame
        Input data.
    outcome : ColumnName
        Name of the outcome column.
    treatment : ColumnName
        Name of the binary treatment column (``1`` treated, ``0`` not).
    instrument : ColumnName
        Name of the binary instrument column (``1`` encouraged, ``0`` not).
    qs : array_like, default=0.5
        Quantiles in ``(0, 1)`` at which to estimate the effect.
    covariates : FormulaRhs
        Right-hand side of the covariates the effect is conditional on; these
        are the controls of the quantile regression.
    weight_formula : FormulaRhs, optional
        Full right-hand side of the first-stage regression that forms the
        complier weights. Use ``poly``/``I`` to build an outcome series, e.g.
        ``"poly(earnings, 5) + covariates"``. When omitted, ``outcome +
        covariates`` is used.
    instrument_probability_formula : FormulaRhs, optional
        Right-hand side of the logit model for ``P(Z = 1 | X)``. When omitted
        the instrument probability is taken to be constant.
    weight_adjustment : `WeightAdjustment`, default=`WeightAdjustment.PROJECTION`
        How the nonnegative complier weights are formed.
    weights : ColumnName, optional
        Name of a column with sampling weights.
    trim : float, default=0.001
        Drop observations whose estimated instrument propensity lies outside
        ``[trim, 1 - trim]``. ``0`` keeps every observation.
    bootstrap_config : `BootstrapConfig` or int, default=100
        Bootstrap settings, or the number of replications.

    Returns
    -------
    `ConditionalLocalQteResult`
        Conditional complier quantile effects and the LATE with bootstrap
        standard errors.

    See Also
    --------
    `qte.iv.estimate_unconditional_local_effects`
    `WeightAdjustment`

    References
    ----------
    Abadie, A., Angrist, J., & Imbens, G. (2002). Instrumental Variables Estimates of the Effect of
    Subsidized Training on the Quantiles of Trainee Earnings. *Econometrica*, 70(1), 91--117.
    """
    bootstrap_config = _make_bootstrap_config(bootstrap_config)
    qs = _as_iterable_array(qs)
    if weight_formula is None:
        weight_formula = f"{outcome} + {covariates}"
    fcn = partial(
        _compute_conditional_local_effects,
        outcome=outcome,
        treatment=treatment,
        instrument=instrument,
        qs=qs,
        covariates=covariates,
        weight_formula=weight_formula,
        instrument_probability_formula=instrument_probability_formula,
        weight_adjustment=weight_adjustment,
        weights=weights,
        trim=trim,
    )
    point = fcn(ds, with_instrument_probability=True)
    runs = _make_bootstrap_runs(
        ds,
        fcn,
        n_iter=bootstrap_config.n_iter,
        seed=bootstrap_config.seed,
        n_workers=bootstrap_config.n_workers,
    )
    qtt, att = _attach_standard_errors(point, runs)
    assert point.instrument_probability is not None
    return ConditionalLocalQteResult(
        qtt=qtt,
        att=att,
        outcome=outcome,
        group=None,
        complier_share=point.complier_share,
        first_stage=point.first_stage,
        weight_adjustment=weight_adjustment,
        instrument_probability=point.instrument_probability,
    )
