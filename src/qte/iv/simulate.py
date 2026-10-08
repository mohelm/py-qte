"""Simulate data for the local quantile treatment effect with a binary instrument.

The design is a self-contained version of the JTPA-based simulation in Kaplan &
Sun (2017). A binary instrument ``Z`` is offered with probability
``offer_probability`` and induces one-sided non-compliance: units with ``Z = 0``
are untreated, while units with ``Z = 1`` take up the treatment with probability
``min(1, u / takeup_scale)``, where ``u`` is a uniform unobservable. The same
``u`` shifts the outcome and the size of the treatment effect, so the treatment
is endogenous and the effect is heterogeneous. Because the treatment effect
increases in ``u`` and compliers are selected towards larger ``u``, the effect
increases with the quantile and the observed-treatment difference overstates it.

The potential outcomes are

    Y(0) = covariate_effect * x + Phi^{-1}(u)
    Y(1) = Y(0) + effect_scale * u

with a standard normal covariate ``x``. By default the instrument is drawn
independently of the covariate and of the unobservable, so the exclusion and
monotonicity restrictions hold by construction and the constant-propensity
estimator is correctly specified. Two optional channels make the covariate
matter:

- ``propensity_scale`` makes the instrument probability depend on ``x``,
  ``P(Z = 1 | x) = logistic(logit(offer_probability) + propensity_scale * x)``,
  so the constant-propensity estimator is inconsistent and the estimator has to
  condition on ``x``.
- ``compliance_scale`` makes take-up depend on ``x``,
  ``P(D(1) = 1 | u, x) = min(1, u / (takeup_scale * exp(compliance_scale * x)))``,
  so the complier population varies with ``x``.

In every case the instrument remains valid conditional on ``x`` and the truth is
recovered from a large simulated population with `get_true_local_effects`.
"""

import numpy as np
import polars as pl
from numpy.typing import ArrayLike
from scipy.special import expit
from scipy.stats import norm

from qte.constants import MEDIAN
from qte.helpers import _as_iterable_array, _get_quantile_differences
from qte.names import (
    COMPLIER_ID,
    COVARIATE_ID,
    INSTRUMENT_ID,
    OBSERVED_OUTCOME_ID,
    POTENTIAL_OUTCOME_CTRL_ID,
    POTENTIAL_OUTCOME_TREAT_ID,
    TREATMENT_ID,
)


def simulate_data(
    n: int = 10_000,
    *,
    effect_scale: float = 2.0,
    covariate_effect: float = 1.0,
    offer_probability: float = 0.67,
    propensity_scale: float = 0.0,
    takeup_scale: float = 0.75,
    compliance_scale: float = 0.0,
    seed: int | None = None,
) -> pl.DataFrame:
    """Simulate a binary-instrument, binary-treatment design with non-compliance.

    Parameters
    ----------
    n : int
        Number of units.
    effect_scale : float
        Scale of the heterogeneous treatment effect ``effect_scale * u``. The
        effect increases with the quantile.
    covariate_effect : float
        Coefficient of the covariate in the outcome.
    offer_probability : float
        Instrument probability when ``propensity_scale`` is zero; also sets the
        intercept of the instrument propensity model otherwise.
    propensity_scale : float
        Coefficient of ``x`` in the logit model for ``P(Z = 1 | x)``. ``0``
        makes the instrument independent of the covariate.
    takeup_scale : float
        Take-up threshold; ``P(D(1) = 1 | u) = min(1, u / takeup_scale)``.
    compliance_scale : float
        Coefficient of ``x`` in the take-up threshold. ``0`` makes compliance
        independent of the covariate.
    seed : int, optional
        Seed for `numpy.random.default_rng`.

    Returns
    -------
    polars.DataFrame
        Columns ``x``, ``instrument``, ``treatment``, ``y``, ``y_0``, ``y_1``
        and ``complier``.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    unobserved = rng.uniform(size=n)

    # This is p(z|x)
    instrument_probability = expit(
        np.log(offer_probability / (1.0 - offer_probability)) + propensity_scale * x
    )
    instrument = rng.binomial(1, instrument_probability).astype(int)

    # Generally, we have a design here where there are no always-takers, i.e. only if the treatment is offered one can take it.
    # Here we comute P(D(1)=1 | u,x)=min(1, u/tau(x)) which is the probability of takeup given
    # (x,u). From that probability we derive at the individual level whether the user would take up
    # the treatment (via sampling from a binomial).
    takeup_threshold = takeup_scale * np.exp(compliance_scale * x)
    takeup_probability = np.minimum(1.0, unobserved / takeup_threshold)
    is_complier = rng.binomial(1, takeup_probability).astype(int)
    treatment = instrument * is_complier

    return (
        pl.DataFrame(
            {
                COVARIATE_ID: x,
                INSTRUMENT_ID: instrument,
                TREATMENT_ID: treatment,
                COMPLIER_ID: is_complier,
                POTENTIAL_OUTCOME_CTRL_ID: covariate_effect * x
                + norm.ppf(np.clip(unobserved, 1e-12, 1.0 - 1e-12)),
            }
        )
        .with_columns(
            (pl.col(POTENTIAL_OUTCOME_CTRL_ID) + effect_scale * unobserved).alias(
                POTENTIAL_OUTCOME_TREAT_ID
            )
        )
        .with_columns(
            pl.when(pl.col(TREATMENT_ID) == 1)
            .then(pl.col(POTENTIAL_OUTCOME_TREAT_ID))
            .otherwise(pl.col(POTENTIAL_OUTCOME_CTRL_ID))
            .alias(OBSERVED_OUTCOME_ID)
        )
        .select(
            COVARIATE_ID,
            INSTRUMENT_ID,
            TREATMENT_ID,
            OBSERVED_OUTCOME_ID,
            POTENTIAL_OUTCOME_CTRL_ID,
            POTENTIAL_OUTCOME_TREAT_ID,
            COMPLIER_ID,
        )
    )


def get_true_local_effects(
    qs: ArrayLike = MEDIAN,
    *,
    n_oracle: int = 1_000_000,
    effect_scale: float = 2.0,
    covariate_effect: float = 1.0,
    offer_probability: float = 0.67,
    propensity_scale: float = 0.0,
    takeup_scale: float = 0.75,
    compliance_scale: float = 0.0,
    seed: int = 0,
) -> tuple[pl.DataFrame, float]:
    """Simulation-based true complier quantile and average effects.

    Both the per-quantile effects and the local average treatment effect are
    recovered from a single simulated population of ``n_oracle`` units, so the
    oracle is evaluated only once. There is no closed form because the complier
    subpopulation is selected by the unobservable.

    Parameters
    ----------
    qs : array_like, default=0.5
        Quantiles in ``(0, 1)``.
    n_oracle : int
        Size of the simulated population used to recover the truth.
    effect_scale, covariate_effect, offer_probability, propensity_scale, takeup_scale, compliance_scale : float
        Passed to `simulate_data`.
    seed : int
        Seed for the oracle simulation.

    Returns
    -------
    tuple of polars.DataFrame and float
        The per-quantile local effects and the local average effect.
    """
    qs = _as_iterable_array(qs)
    ds = simulate_data(
        n_oracle,
        effect_scale=effect_scale,
        covariate_effect=covariate_effect,
        offer_probability=offer_probability,
        propensity_scale=propensity_scale,
        takeup_scale=takeup_scale,
        compliance_scale=compliance_scale,
        seed=seed,
    )
    compliers = ds.filter(pl.col(COMPLIER_ID) == 1)
    y_1 = compliers[POTENTIAL_OUTCOME_TREAT_ID].to_numpy().astype(float)
    y_0 = compliers[POTENTIAL_OUTCOME_CTRL_ID].to_numpy().astype(float)
    return (
        _get_quantile_differences(qs, np.quantile(y_1, qs), np.quantile(y_0, qs)),
        float(y_1.mean() - y_0.mean()),
    )
