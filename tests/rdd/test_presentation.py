"""Tests for the RD design plots."""

import altair as alt
import pytest

from qte.rdd import (
    LocalPolynomialConfig,
    RunningVariableConfig,
    plot_rdd_design,
    plot_rdd_treatment,
    simulate_rdd_data,
)


def test_plot_rdd_design_layers_points_and_fit():
    ds = simulate_rdd_data(n=2_000, seed=0)
    chart = plot_rdd_design(
        ds,
        "y",
        RunningVariableConfig("running", 0.0),
        n_bins=10,
        local_polynomial_config=LocalPolynomialConfig(bandwidth=0.75),
    )

    assert isinstance(chart, alt.LayerChart)
    assert len(chart.layer) >= 4


def test_plot_rdd_treatment_and_no_fit():
    ds = simulate_rdd_data(n=2_000, seed=0)
    chart = plot_rdd_treatment(
        ds,
        "treatment",
        RunningVariableConfig("running", 0.0),
        n_bins=10,
        show_fit=False,
    )

    assert isinstance(chart, alt.LayerChart)
    assert len(chart.layer) == 2


def test_plot_rdd_design_rejects_invalid_bins():
    ds = simulate_rdd_data(n=100, seed=0)
    with pytest.raises(ValueError, match="n_bins"):
        plot_rdd_design(ds, "y", RunningVariableConfig("running", 0.0), n_bins=1)
