from dataclasses import dataclass
from typing import NamedTuple

from qte.nonlinear_did.custom_types import (
    BasePeriod,
    ControlGroup,
    CounterfactualModel,
    SamplingScheme,
)
from qte.results import _BasicQteResult
from qte.stats import Ecdf


class GroupTimeEffect(NamedTuple):
    group: int
    tp: int
    ecdf_observed: Ecdf
    ecdf_counterfact: Ecdf
    mean_observed: float
    mean_countfact: float
    group_size_observed: int
    group_size_counterfactual: int


@dataclass()
class NonlinearDidResult(_BasicQteResult):
    group: tuple[str, ...] | None
    base_period: BasePeriod
    control_group: ControlGroup
    sampling_scheme: SamplingScheme
    counterfactual_model: CounterfactualModel


@dataclass()
class NonlinearDidResults:
    group: NonlinearDidResult
    event_study: NonlinearDidResult
    overall: NonlinearDidResult
    group_time: NonlinearDidResult
