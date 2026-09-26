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


def test_plot_returns_altair_chart_with_single_quantile_qte_res(mock_single_quantile_qte_result):
    chart = mock_single_quantile_qte_result.plot()

    assert isinstance(chart, alt.LayerChart)

    chart_dict = chart.to_dict()
    assert isinstance(chart_dict, dict)

    layers = chart_dict.get("layer", [])
    assert (
        len(layers) == 4
    )  # Four layers: one line (confidence interval) and one point (estimate) for att and qte (so time 2)


def test_plot_returns_altair_chart_with_multiple_quantile_qte_result(
    mock_qte_result,
):
    chart = mock_qte_result.plot()

    assert isinstance(chart, alt.LayerChart)

    chart_dict = chart.to_dict()
    assert isinstance(chart_dict, dict)

    layers = chart_dict.get("layer", [])
    assert len(layers) == 2  # 2 aggregate layers (qte and att)


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
