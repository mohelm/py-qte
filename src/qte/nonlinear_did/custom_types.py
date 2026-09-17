from enum import StrEnum
from typing import NamedTuple

import polars as pl

from qte.custom_types import ColumnName


class ControlGroup(StrEnum):
    NOT_YET_TREATED = "not_yet_treated"
    NEVER_TREATED = "never_treated"


class BasePeriod(StrEnum):
    UNIVERSAL = "universal"
    VARYING = "varying"


class SamplingScheme(StrEnum):
    PANEL = "panel"
    REPEATED_CROSS_SECTIONS = "repeated_cross_sections"


class CounterfactualModel(StrEnum):
    CIC = "cic"
    QDID = "qdid"


class NonlinearDidAggregation(NamedTuple):
    qtt: pl.DataFrame
    att: pl.DataFrame
    group: str | None


NonlinearDidAggregations = dict[str, NonlinearDidAggregation]

TreatmentGroup = int
TimePeriod = int
WeightsLookup = dict[tuple[TreatmentGroup, TimePeriod], float]


class TrtGroupConfig(NamedTuple):
    name: ColumnName
    never_treated_identifier: float = float("inf")
