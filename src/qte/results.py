import dataclasses
from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import NDArray


@dataclass()
class QteResult:
    qs: NDArray[np.float64]
    q_val_t: NDArray[np.float64]
    q_val_c: NDArray[np.float64]
    effects: None | NDArray[np.float64] = None

    def __post_init__(self) -> None:
        self.effects = self.q_val_t - self.q_val_c

    def to_polars(self) -> pl.DataFrame:
        return pl.from_dict(dataclasses.asdict(self)).rename({
            "effects": "effect",
            "qs": "q",
        })
