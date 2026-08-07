import polars as pl
import pyreadr
from pytest import fixture


@fixture(scope="session")
def lalonde():
    return pyreadr.read_r("~/Downloads/lalonde.RData")


@fixture(scope="session")
def lalonde_psid(lalonde) -> pl.DataFrame:
    return pl.from_pandas(lalonde["lalonde.psid"])
