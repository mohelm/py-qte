from enum import StrEnum

import polars as pl

type ColumnName = str
type DataFrame = pl.DataFrame
type FormularRhs = str
type Series = pl.Series


class CausalTarget(StrEnum):
    QTT = "qtt"
    QTE = "qte"


class Estimator(StrEnum):
    SIMPLE = "simple"
    IPW = "ipw"
    OR = "or"
    AIPW = "aipw"
