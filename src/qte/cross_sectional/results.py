from dataclasses import dataclass
from typing import NamedTuple

import polars as pl

from qte.custom_types import CausalTarget, Estimator, FormularRhs
from qte.results import _BasicQteResult


class _QteIntermediateResult(NamedTuple):
    qtt: pl.DataFrame
    att: pl.DataFrame


@dataclass
class QteResult(_BasicQteResult):
    estimator: Estimator
    causal_target: CausalTarget
    ps_x_formular: FormularRhs | None = None
    or_x_formular: FormularRhs | None = None
