from dataclasses import dataclass
from typing import NamedTuple

from qte.non_linear_did.custom_types import (
    BasePeriod,
    ControlGroup,
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
class CicResult(_BasicQteResult):
    group: str | None
    base_period: BasePeriod
    control_group: ControlGroup
    sampling_scheme: SamplingScheme


@dataclass()
class CicResults:
    group: CicResult
    event_study: CicResult
    overall: CicResult
