import polars as pl
from pytest import fixture

from qte.datasets import load_card, load_lalonde, load_mpdta


@fixture(scope="session")
def lalonde_psid() -> pl.DataFrame:
    return load_lalonde(controls_source="psid", use_panel_structure=False)


@fixture(scope="session")
def mpdata() -> pl.DataFrame:
    return load_mpdta()


@fixture(scope="session")
def card() -> pl.DataFrame:
    """Card (1995) with the ivqte/AAI treatment definition ``college = educ >= 13``."""
    return load_card(treatment_cutoff=13).drop_nulls(["fatheduc", "motheduc"])
