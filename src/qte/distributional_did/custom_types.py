"""Custom types for the distributional difference-in-differences estimators."""

from dataclasses import dataclass
from enum import StrEnum

import polars as pl

from qte.custom_types import ColumnName


class ControlGroup(StrEnum):
    """Comparison group used for the counterfactual.

    Use a member as the ``control_group`` argument of
    `estimate_changes_in_changes_effects`.

    Attributes
    ----------
    NOT_YET_TREATED
        Units not yet treated at the time period of interest.
    NEVER_TREATED
        Units that are never treated.

    Examples
    --------
    Use not-yet-treated units as the comparison group:

    >>> from qte.datasets import load_mpdta
    >>> from qte.distributional_did import ControlGroup, TrtGroupConfig, estimate_changes_in_changes_effects
    >>> res = estimate_changes_in_changes_effects(
    ...     load_mpdta(),
    ...     "lemp",
    ...     TrtGroupConfig("first.treat", 0),
    ...     "year",
    ...     "countyreal",
    ...     control_group=ControlGroup.NOT_YET_TREATED,
    ...     bootstrap_config=2,
    ... )
    """

    NOT_YET_TREATED = "not_yet_treated"
    NEVER_TREATED = "never_treated"


class BasePeriod(StrEnum):
    """Pre-treatment period the post-treatment effect is compared to.

    Use a member as the ``base_period`` argument of
    `estimate_changes_in_changes_effects`.

    Attributes
    ----------
    UNIVERSAL
        Always compare to the same, earliest pre-treatment period.
    VARYING
        Compare to the immediately preceding pre-treatment period.

    Examples
    --------
    Compare each period to the immediately preceding one:

    >>> from qte.datasets import load_mpdta
    >>> from qte.distributional_did import BasePeriod, TrtGroupConfig, estimate_changes_in_changes_effects
    >>> res = estimate_changes_in_changes_effects(
    ...     load_mpdta(),
    ...     "lemp",
    ...     TrtGroupConfig("first.treat", 0),
    ...     "year",
    ...     "countyreal",
    ...     base_period=BasePeriod.VARYING,
    ...     bootstrap_config=2,
    ... )
    """

    UNIVERSAL = "universal"
    VARYING = "varying"


class SamplingScheme(StrEnum):
    """Sampling scheme of the input data.

    Attributes
    ----------
    PANEL
        The same units are observed in every time period.
    REPEATED_CROSS_SECTIONS
        Independent samples in every time period.
    """

    PANEL = "panel"
    REPEATED_CROSS_SECTIONS = "repeated_cross_sections"


class Estimator(StrEnum):
    """Estimator that produced a result.

    This is an output field: read it from ``result.estimator`` to see how an estimate was
    produced, and compare it against a member. The estimator is selected by calling the
    corresponding `estimate_changes_in_changes_effects` or `estimate_quantile_did_effects` function.

    Attributes
    ----------
    CHANGES_IN_CHANGES
        Changes-in-changes.
    QUANTILE_DID
        Quantile difference-in-differences.

    Examples
    --------
    Inspect which estimator produced a result:

    >>> from qte.datasets import load_mpdta
    >>> from qte.distributional_did import Estimator, TrtGroupConfig, estimate_quantile_did_effects
    >>> res = estimate_quantile_did_effects(
    ...     load_mpdta(),
    ...     "lemp",
    ...     TrtGroupConfig("first.treat", 0),
    ...     "year",
    ...     "countyreal",
    ...     bootstrap_config=2,
    ... )
    >>> res.overall.estimator is Estimator.QUANTILE_DID
    True
    """

    CHANGES_IN_CHANGES = "changes_in_changes"
    QUANTILE_DID = "quantile_did"


@dataclass(frozen=True)
class _DistributionalDidAggregation:
    qtt: pl.DataFrame
    att: pl.DataFrame
    group: tuple[str, ...] | None


_DistributionalDidAggregations = dict[str, _DistributionalDidAggregation]

TreatmentGroup = int
TimePeriod = int
WeightsLookup = dict[tuple[TreatmentGroup, TimePeriod], float]


@dataclass(frozen=True)
class TrtGroupConfig:
    """Lazy configuration for the treatment group column.

    Parameters
    ----------
    name : str
        Name of the column holding the first treatment period of a unit.
    never_treated_identifier : float, default=inf
        Value in ``name`` marking never-treated units. It is mapped to
        ``inf`` internally.
    """

    name: ColumnName
    never_treated_identifier: float = float("inf")
