import polars as pl
import pytest

from qte.datasets import load_lalonde, load_mpdta


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
