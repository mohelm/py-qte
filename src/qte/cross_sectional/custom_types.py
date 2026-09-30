"""Custom types for the cross-sectional estimators."""

from enum import StrEnum


class CausalTarget(StrEnum):
    """Causal estimand a cross-sectional estimator targets.

    Use a member as the ``target`` argument of the cross-sectional estimators,
    for example ``CausalTarget.QTT``.

    Attributes
    ----------
    QTE
        Quantile treatment effect for the whole population.
    QTT
        Quantile treatment effect on the treated.

    Examples
    --------
    Target the effect on the treated instead of the whole population:

    >>> from qte.cross_sectional import CausalTarget, estimate_ipw_qte
    >>> from qte.datasets import load_lalonde
    >>> res = estimate_ipw_qte(
    ...     load_lalonde(),
    ...     "re78",
    ...     "treat",
    ...     ps_x_formular="age + education",
    ...     target=CausalTarget.QTT,
    ... )
    """

    QTE = "qte"
    QTT = "qtt"


class Estimator(StrEnum):
    """Cross-sectional estimator that produced a result.

    This is an output field: read it from ``result.estimator`` to see how an
    estimate was produced, and compare it against a member.

    Attributes
    ----------
    SIMPLE
        Difference in group quantiles without covariate adjustment.
    IPW
        Inverse probability weighting.
    OR
        Outcome regression.
    AIPW
        Augmented inverse probability weighting.

    Examples
    --------
    Inspect which estimator produced a result:

    >>> from qte.cross_sectional import Estimator, estimate_simple_qte
    >>> from qte.datasets import load_lalonde
    >>> res = estimate_simple_qte(load_lalonde(), "re78", "treat")
    >>> res.estimator is Estimator.SIMPLE
    True
    """

    SIMPLE = "simple"
    IPW = "ipw"
    OR = "or"
    AIPW = "aipw"
