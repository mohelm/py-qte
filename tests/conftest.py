import polars as pl
from pytest import fixture

from qte.datasets import load_lalonde, load_mpdta


@fixture(scope="session")
def lalonde_psid() -> pl.DataFrame:
    return load_lalonde(experimental=False, panel=False)


@fixture(scope="session")
def mpdata() -> pl.DataFrame:
    return load_mpdta()
