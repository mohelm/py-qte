import altair as alt
import polars as pl
import pytest

from qte.cross_sectional.results import QteResult
from qte.custom_types import CausalTarget, Estimator


@pytest.fixture
def mock_qte_result():
    ds = pl.DataFrame({
        "q": [0.25, 0.5, 0.75],
        "effect": [-5.0, 0.0, 5.0],
        "q_val_t": [10.0, 15.0, 20.0],
        "q_val_c": [15.0, 15.0, 15.0],
        "se": [1.0, 1.0, 1.0],
    })
    return QteResult(Estimator.SIMPLE, CausalTarget.QTE, ds)


def test_plot_returns_altair_chart(mock_qte_result):
    chart = mock_qte_result.plot()

    assert isinstance(chart, alt.LayerChart)

    chart_dict = chart.to_dict()
    assert isinstance(chart_dict, dict)

    layers = chart_dict.get("layer", [])
    assert len(layers) == 3  # 3 lines (effect, ci_lb, ci_ub)


def test_plot_with_different_alpha(mock_qte_result):
    chart = mock_qte_result.plot(alpha=0.99)
    assert isinstance(chart, alt.LayerChart)
    chart_dict = chart.to_dict()
    assert isinstance(chart_dict, dict)


def test_summarize_returns_rich_table(mock_qte_result):
    from rich.table import Table

    table = mock_qte_result.summarize()
    assert isinstance(table, Table)

    columns = [col.header for col in table.columns]
    assert "q" in columns
    assert "effect" in columns
    assert "se" in columns
    assert "ci_lb" in columns
    assert "ci_ub" in columns
    assert "q_val_t" not in columns
    assert "q_val_c" not in columns


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
