import polars as pl
import pytest

from qte.datasets import (
    load_card,
    load_ddk_2011,
    load_engel_with_weights,
    load_jtpa,
    load_lalonde,
    load_mpdta,
    load_pension,
)


@pytest.mark.parametrize(
    "controls_source, use_panel_structure, expected_shape, expected_cols",
    [
        ("psid", False, (2675, 13), ["age", "education", "treat", "re78"]),
        ("psid", True, (8025, 13), ["year", "id", "treat", "re"]),
        ("experiment", False, (445, 13), ["age", "education", "treat", "re78"]),
        ("experiment", True, (1335, 13), ["year", "id", "treat", "re"]),
    ],
)
def test_load_lalonde(
    controls_source,
    use_panel_structure,
    expected_shape,
    expected_cols,
):
    df = load_lalonde(
        controls_source=controls_source,
        use_panel_structure=use_panel_structure,
    )

    assert isinstance(df, pl.DataFrame)

    assert df.shape == expected_shape

    for col in expected_cols:
        assert col in df.columns


def test_mpdta():
    ds = load_mpdta()
    assert isinstance(ds, pl.DataFrame)
    assert ds.shape == (2500, 6)
    expected_columns = [
        "lemp",
        "first.treat",
        "year",
        "countyreal",
        "lpop",
        "countyreal",
    ]
    for c in expected_columns:
        assert c in ds.columns


def test_engel_with_weights():
    ds = load_engel_with_weights()
    assert isinstance(ds, pl.DataFrame)
    expected_columns = ["w", "log_foodexp", "log_income"]
    for c in expected_columns:
        assert c in ds.columns


def test_load_card():
    ds = load_card()
    assert isinstance(ds, pl.DataFrame)
    assert ds.height == 3010
    assert {"nearc4", "educ", "lwage", "college"} <= set(ds.columns)
    assert set(ds["college"].unique().to_list()) <= {0, 1}
    low = load_card(treatment_cutoff=13)
    assert low["college"].mean() > ds["college"].mean()


def test_load_jtpa():
    ds = load_jtpa()
    assert isinstance(ds, pl.DataFrame)
    assert ds.height == 11_204
    assert {"assignmt", "training", "earnings", "sex", "age"} <= set(ds.columns)
    # Almost one-sided non-compliance: very few controls enrol in training.
    controls = ds.filter(pl.col("assignmt") == 0)
    assert controls["training"].mean() < 0.02


def test_load_pension():
    ds = load_pension()
    assert isinstance(ds, pl.DataFrame)
    assert ds.shape == (9_915, 44)
    assert {
        "net_tfa",
        "p401",
        "e401",
        "icat",
        "ecat",
        "a1",
        "marr",
        "pira",
        "hown",
    } <= set(ds.columns)
    assert set(ds["p401"].unique().to_list()) == {0, 1}
    assert set(ds["e401"].unique().to_list()) == {0, 1}


def test_load_ddk_2011():
    ds = load_ddk_2011()
    assert isinstance(ds, pl.DataFrame)
    assert ds.shape == (5_795, 10)
    assert {
        "totalscore",
        "tracking",
        "percentile",
        "highstream",
        "ts_std",
    } <= set(ds.columns)
    assert set(ds["tracking"].unique().to_list()) <= {0, 1}
    assert set(ds["highstream"].unique().to_list()) <= {0, 1}
