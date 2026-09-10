from enum import StrEnum
from typing import NamedTuple

import polars as pl


class ControlGroup(StrEnum):
    NOT_YET_TREATED = "not_yet_treated"
    NEVER_TREATED = "never_treated"


class BasePeriod(StrEnum):
    UNIVERSAL = "universal"
    VARYING = "varying"


class CicAggregation(NamedTuple):
    qtt: pl.DataFrame
    att: pl.DataFrame
    group: str | None


CicAggregations = dict[str, CicAggregation]
