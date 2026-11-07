"""Result containers for the distributional difference-in-differences estimator."""

from dataclasses import dataclass

from qte.distributional_did.custom_types import (
    BasePeriod,
    ControlGroup,
    Estimator,
    SamplingScheme,
)
from qte.results import _BasicQteResult
from qte.stats import Ecdf


@dataclass(frozen=True)
class GroupTimeEffect:
    group: int
    tp: int
    ecdf_observed: Ecdf
    ecdf_counterfact: Ecdf
    mean_observed: float
    mean_counterfact: float
    group_size_observed: int
    group_size_counterfactual: int


@dataclass()
class DistributionalDidResult(_BasicQteResult):
    """Result for one aggregation of a distributional difference-in-differences estimation.

    Use `plot`, `tabulate`, `summarize` or `get_as_dataframe` to present the
    estimates. Instances are returned by `estimate_changes_in_changes_effects` or `estimate_quantile_did_effects`; the
    constructor is not part of the public API.

    Attributes
    ----------
    qtt : polars.DataFrame
        Quantile-specific effects.
    att : polars.DataFrame
        Average treatment effects.
    outcome : ColumnName
        Outcome variable the effect was estimated for.
    group : tuple[str, ...] or None
        Columns the estimates are grouped by.
    base_period : `BasePeriod`
        Base period used for the comparisons.
    control_group : `ControlGroup`
        Comparison group used for the counterfactual.
    sampling_scheme : `SamplingScheme`
        Sampling scheme of the input data.
    estimator : `Estimator`
        Estimator that produced the result.
    """

    group: tuple[str, ...] | None
    base_period: BasePeriod
    control_group: ControlGroup
    sampling_scheme: SamplingScheme
    estimator: Estimator


@dataclass()
class DistributionalDidResults:
    """The four aggregations returned by `estimate_changes_in_changes_effects` or `estimate_quantile_did_effects`.

    Attributes
    ----------
    group : `DistributionalDidResult`
        Effects aggregated by treatment group.
    event_study : `DistributionalDidResult`
        Effects by event-study period (time since treatment).
    overall : `DistributionalDidResult`
        Single overall effect.
    group_time : `DistributionalDidResult`
        Effects by treatment group and time period.
    """

    group: DistributionalDidResult
    event_study: DistributionalDidResult
    overall: DistributionalDidResult
    group_time: DistributionalDidResult
