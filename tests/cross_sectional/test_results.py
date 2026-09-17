import altair as alt
import polars as pl
import pytest

from qte.cross_sectional.results import QteResult
from qte.custom_types import CausalTarget, Estimator
from qte.names import (
    EFFECT_ID,
    MEAN_CONTROL_ID,
    MEAN_TREATED_ID,
    QUANTILE_CONTROL_VAL_ID,
    QUANTILE_ID,
    QUANTILE_TREATED_VAL_ID,
    SE_ID,
)


@pytest.fixture
def mock_qte_result():
    ds = pl.DataFrame(
        {
            QUANTILE_ID: [0.25, 0.5, 0.75],
            EFFECT_ID: [-5.0, 0.0, 5.0],
            QUANTILE_TREATED_VAL_ID: [10.0, 15.0, 20.0],
            QUANTILE_CONTROL_VAL_ID: [15.0, 15.0, 15.0],
            SE_ID: [1.0, 1.0, 1.0],
        }
    )
    return QteResult(
        qtt=ds,
        att=pl.DataFrame(
            {
                MEAN_TREATED_ID: [10.0],
                MEAN_CONTROL_ID: [15.0],
                EFFECT_ID: [-5.0],
                SE_ID: [1.0],
            }
        ),
        outcome="re78",
        group=None,
        estimator=Estimator.SIMPLE,
        causal_target=CausalTarget.QTE,
    )


@pytest.fixture
def mock_single_quantile_qte_result():
    ds = pl.DataFrame(
        {
            QUANTILE_ID: [0.5],
            EFFECT_ID: [1.0],
            QUANTILE_TREATED_VAL_ID: [10.0],
            QUANTILE_CONTROL_VAL_ID: [15.0],
            SE_ID: [0.2],
        }
    )
    return QteResult(
        qtt=ds,
        att=pl.DataFrame(
            {
                MEAN_TREATED_ID: [10.0],
                MEAN_CONTROL_ID: [15.0],
                EFFECT_ID: [-5.0],
                SE_ID: [1.0],
            }
        ),
        outcome="re78",
        group=None,
        estimator=Estimator.SIMPLE,
        causal_target=CausalTarget.QTE,
    )


def test_plot_returns_altair_chart(mock_qte_result):
    chart = mock_qte_result.plot()

    assert isinstance(chart, alt.LayerChart)

    chart_dict = chart.to_dict()
    assert isinstance(chart_dict, dict)

    layers = chart_dict.get("layer", [])
    assert len(layers) == 2  # 2 aggregate layers (qte and att)

    # x-axis title is a single, non-bold "Q"
    x_axis = layers[0]["layer"][0]["encoding"]["x"]["axis"]
    assert x_axis["title"] == "Q"
    assert x_axis["titleFontWeight"] == "normal"


def test_plot_with_different_alpha(mock_qte_result):
    chart = mock_qte_result.plot(alpha=0.99)
    assert isinstance(chart, alt.LayerChart)
    chart_dict = chart.to_dict()
    assert isinstance(chart_dict, dict)


def test_plot_single_quantile_orders_and_colors(mock_single_quantile_qte_result):
    chart = mock_single_quantile_qte_result.plot()
    assert isinstance(chart, alt.LayerChart)

    layers = chart.to_dict()["layer"]
    # point estimate + CI rule + CI lower marker + CI upper marker
    assert len(layers) == 4

    # the title/subtitle must survive onto the layered chart
    assert chart.to_dict()["title"] == {
        "text": "Quantile and Average Treatment Effects",
        "subtitle": "The QTE is measured for quantile=0.5.",
        "offset": 10,
    }

    for layer in layers:
        # qte must come before att (left to right), not alphabetical order
        assert layer["encoding"]["x"]["sort"] == ["qte", "att"]
        # qte is black, att is red
        assert layer["encoding"]["color"]["scale"] == {
            "domain": ["qte", "att"],
            "range": ["black", "red"],
        }
        assert layer["encoding"]["color"]["legend"] is None

    # the point estimate is a solid dot, filled with its __kind color
    assert layers[0]["mark"]["type"] == "point"
    assert layers[0]["mark"]["filled"] is True

    # the CI endpoint markers are hollow, stroked with the same __kind color
    for layer in layers[2:]:
        assert layer["mark"]["shape"] == "stroke"


def test_summarize_returns_rich_table(mock_qte_result):
    from rich.console import Group

    table = mock_qte_result.summarize()
    assert isinstance(table, Group)


def test_get_as_dataframe(mock_qte_result):
    df = mock_qte_result.get_as_dataframe()
    assert isinstance(df, pl.DataFrame)

    assert "ci_lb" in df.columns
    assert "ci_ub" in df.columns

    assert df.shape[0] == 3


def test_tabulate_returns_gt(mock_qte_result):
    from great_tables import GT

    table = mock_qte_result.tabulate()
    assert isinstance(table, GT)
