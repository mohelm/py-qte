"""Custom types for the instrumental-variables estimators."""

from enum import StrEnum


class Estimator(StrEnum):
    """Weighting scheme for the complier quantile treatment effect.

    Use a member as the ``estimator`` argument of `qte.iv.estimate_unconditional_local_effects`.

    Attributes
    ----------
    SIGNED
        Abadie's signed instrument weights, with the complier empirical CDF
        inverted in outcome order.
    POSITIVE
        The instrument weights are projected onto the outcome separately for the
        treated and untreated, negative values are dropped, and the effect is a
        difference of weighted quantiles (Stata ``ivqte, positive``).

    Examples
    --------
    Estimate the effect with the positive-weight scheme:

    >>> from qte.datasets import load_card
    >>> from qte.iv import Estimator, estimate_unconditional_local_effects
    >>> res = estimate_unconditional_local_effects(
    ...     load_card(treatment_cutoff=13),
    ...     "lwage",
    ...     "college",
    ...     "nearc4",
    ...     estimator=Estimator.POSITIVE,
    ... )
    """

    SIGNED = "signed"
    POSITIVE = "positive"


class WeightAdjustment(StrEnum):
    """How the nonnegative complier weights are formed.

    Use a member as the ``weight_adjustment`` argument of
    `qte.iv.estimate_conditional_local_effects`.

    Attributes
    ----------
    PROJECTION
        Project the Abadie-Angrist-Imbens kappa onto the outcome and covariates
        separately for the treated and untreated (Stata ``ivqte, aai``).
    PLUGIN
        Estimate ``nu(U) = E[Z | Y, D, X]`` and plug it into the exact kappa
        identity, ``1 - D(1-nu)/(1-p) - (1-D)nu/p`` (Abadie, Angrist & Imbens,
        2002, eq. 4).
    """

    PROJECTION = "projection"
    PLUGIN = "plugin"
