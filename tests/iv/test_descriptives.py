import polars as pl
from great_tables import GT
from rich.table import Table

from qte.datasets import load_card
from qte.iv.descriptives import (
    OutputFormat,
    _make_display_frame,
    _RowType,
    make_descriptive_statistics,
)


def test_make_display_frame_observation_row_and_row_types():
    ds = load_card()
    display, row_meta = _make_display_frame(ds, "lwage", "college", "nearc4", ["educ", "age"], {})

    observations = display.row(0, named=True)
    assert observations["group"] == "Sample"
    assert observations["variable"] == "Number of observations"
    assert observations["entire"] == ds.height
    assert observations["assign_treated"] == ds.filter(pl.col("nearc4") == 1).height
    assert observations["assign_control"] == ds.filter(pl.col("nearc4") == 0).height
    assert observations["treat_treated"] == ds.filter(pl.col("college") == 1).height
    assert observations["treat_control"] == ds.filter(pl.col("college") == 0).height

    assert row_meta[0].kind is _RowType.N_OBS
    assert [meta.kind for meta in row_meta[1:]] == [
        _RowType.MEAN,
        _RowType.SD,
    ] * 4


def test_make_descriptive_statistics_returns_both_backends():
    ds = load_card()
    great_table = make_descriptive_statistics(
        ds,
        "lwage",
        "college",
        "nearc4",
        covariates=["educ", "age"],
        title="Card descriptive statistics",
    )
    assert isinstance(great_table, GT)
    html = great_table.as_raw_html()
    assert "Number of observations" in html
    assert "Standard deviations are shown in square brackets" in html

    rich_table = make_descriptive_statistics(
        ds,
        "lwage",
        "college",
        "nearc4",
        output=OutputFormat.rich_table,
        covariates=["educ", "age"],
        title="Card descriptive statistics",
    )
    assert isinstance(rich_table, Table)
