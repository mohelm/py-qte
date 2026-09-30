"""Result containers for the nonlinear difference-in-differences estimator."""

from dataclasses import dataclass

from qte.nonlinear_did.custom_types import (
    BasePeriod,
    ControlGroup,
    CounterfactualModel,
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
class NonlinearDidResult(_BasicQteResult):
    """Result for one aggregation of a nonlinear difference-in-differences estimation.

    Use `plot`, `tabulate`, `summarize` or `get_as_dataframe` to present the
    estimates. Instances are returned by `estimate_nonlinear_did_for_panel`; the
    constructor is not part of the public API.

    Attributes
    ----------
    qtt : polars.DataFrame
        Quantile-specific effects.
    att : polars.DataFrame
        Average treatment effects.
    outcome : str
        Outcome variable the effect was estimated for.
    group : tuple[str, ...] or None
        Columns the estimates are grouped by.
    base_period : `BasePeriod`
        Base period used for the comparisons.
    control_group : `ControlGroup`
        Comparison group used for the counterfactual.
    sampling_scheme : `SamplingScheme`
        Sampling scheme of the input data.
    counterfactual_model : `CounterfactualModel`
        Model used to construct the counterfactual distribution.
    """

    group: tuple[str, ...] | None
    base_period: BasePeriod
    control_group: ControlGroup
    sampling_scheme: SamplingScheme
    counterfactual_model: CounterfactualModel


@dataclass()
class NonlinearDidResults:
    """The four aggregations returned by `estimate_nonlinear_did_for_panel`.

    Attributes
    ----------
    group : `NonlinearDidResult`
        Effects aggregated by treatment group.
    event_study : `NonlinearDidResult`
        Effects by event-study period (time since treatment).
    overall : `NonlinearDidResult`
        Single overall effect.
    group_time : `NonlinearDidResult`
        Effects by treatment group and time period.
    """

    group: NonlinearDidResult
    event_study: NonlinearDidResult
    overall: NonlinearDidResult
    group_time: NonlinearDidResult
