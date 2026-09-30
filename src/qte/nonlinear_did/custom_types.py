"""Custom types for the nonlinear difference-in-differences estimators."""

from dataclasses import dataclass
from enum import StrEnum

import polars as pl

from qte.custom_types import ColumnName


class ControlGroup(StrEnum):
    """Comparison group used for the counterfactual.

    Use a member as the ``control_group`` argument of
    `estimate_nonlinear_did_for_panel`.

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
    >>> from qte.nonlinear_did import ControlGroup, estimate_nonlinear_did_for_panel
    >>> res = estimate_nonlinear_did_for_panel(
    ...     load_mpdta(),
    ...     "lemp",
    ...     "first.treat",
    ...     "year",
    ...     "countyreal",
    ...     control_group=ControlGroup.NOT_YET_TREATED,
    ... )
    """

    NOT_YET_TREATED = "not_yet_treated"
    NEVER_TREATED = "never_treated"


class BasePeriod(StrEnum):
    """Pre-treatment period the post-treatment effect is compared to.

    Use a member as the ``base_period`` argument of
    `estimate_nonlinear_did_for_panel`.

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
    >>> from qte.nonlinear_did import BasePeriod, estimate_nonlinear_did_for_panel
    >>> res = estimate_nonlinear_did_for_panel(
    ...     load_mpdta(),
    ...     "lemp",
    ...     "first.treat",
    ...     "year",
    ...     "countyreal",
    ...     base_period=BasePeriod.VARYING,
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


class CounterfactualModel(StrEnum):
    """Model used to construct the counterfactual distribution.

    Use a member as the ``counterfactual_model`` argument of
    `estimate_nonlinear_did_for_panel`.

    Attributes
    ----------
    CIC
        Changes-in-changes.
    QDID
        Quantile difference-in-differences.

    Examples
    --------
    Use quantile difference-in-differences instead of changes-in-changes:

    >>> from qte.datasets import load_mpdta
    >>> from qte.nonlinear_did import CounterfactualModel, estimate_nonlinear_did_for_panel
    >>> res = estimate_nonlinear_did_for_panel(
    ...     load_mpdta(),
    ...     "lemp",
    ...     "first.treat",
    ...     "year",
    ...     "countyreal",
    ...     counterfactual_model=CounterfactualModel.QDID,
    ... )
    """

    CIC = "cic"
    QDID = "qdid"


@dataclass(frozen=True)
class _NonlinearDidAggregation:
    qtt: pl.DataFrame
    att: pl.DataFrame
    group: tuple[str, ...] | None


_NonlinearDidAggregations = dict[str, _NonlinearDidAggregation]

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
