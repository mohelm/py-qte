from enum import StrEnum


class ControlGroup(StrEnum):
    NOT_YET_TREATED = "not_yet_treated"
    NEVER_TREATED = "never_treated"


class BasePeriod(StrEnum):
    UNIVERSAL = "universal"
    VARYING = "varying"
