import numpy as np
import polars as pl

from qte.qte import estimate_qte


def make_data(n_treated: int = 500, n_control: int | None = None) -> pl.DataFrame:
    if n_control is None:
        n_control = n_treated
    rng = np.random.default_rng()
    treated_outcome = rng.normal(0, 1, n_treated)
    control_outcome = rng.normal(1, 1, n_control)
    return pl.from_dict({
        "outcome": np.r_[treated_outcome, control_outcome],
        "treated": np.r_[np.ones((n_treated,)), np.zeros((n_control,))],
    })


def test_estimate_qte():
    ds = make_data(5000)
    assert isinstance(ds, pl.DataFrame)

    res = estimate_qte(ds, "outcome", "treated", qs=(0.05, 0.5, 0.95))
    assert isinstance(res, pl.DataFrame)
    print(res)
