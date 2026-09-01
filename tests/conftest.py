import polars as pl
from pytest import fixture

from qte.datasets import load_lalonde


@fixture(scope="session")
def lalonde_psid() -> pl.DataFrame:
    return load_lalonde(experimental=False, panel=False)
