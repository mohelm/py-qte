"""Design plots for regression-discontinuity estimates."""

import altair as alt
import numpy as np
import polars as pl
from numpy.typing import NDArray

from qte._validation import (
    check_columns_are_in_dataset,
    check_missing_values_in_dataset,
    check_numeric,
    check_outcome,
    check_treatment_is_binary,
)
from qte.custom_types import ColumnName
from qte.rdd.custom_types import LocalPolynomialConfig, RunningVariableConfig, _Side
from qte.results import AltairChart

_SIDE_DOMAIN = [_Side.LEFT.value, _Side.RIGHT.value]
_SIDE_RANGE = ["#4C78A8", "#E45756"]


def _compute_side_bin_means(
    r: NDArray[np.float64],
    values: NDArray[np.float64],
    *,
    side: _Side,
    n_bins: int,
) -> pl.DataFrame:
    """Bin one side of the cutoff and return bin means of the running variable and value."""
    mask = r >= 0.0 if side is _Side.RIGHT else r < 0.0
    if not mask.any():
        return pl.DataFrame(
            schema={"running": pl.Float64, "value": pl.Float64, "side": pl.String, "n": pl.Int64}
        )

    side_r = r[mask]
    side_values = values[mask]
    bins = np.linspace(side_r.min(), side_r.max(), n_bins + 1)
    bin_index = np.clip(np.digitize(side_r, bins, right=False) - 1, 0, n_bins - 1)

    return (
        pl.DataFrame(
            {
                "running": side_r,
                "value": side_values,
                "bin": bin_index,
            }
        )
        .group_by("bin")
        .agg(
            pl.col("running").mean(),
            pl.col("value").mean(),
            pl.len().alias("n"),
        )
        .sort("running")
        .with_columns(pl.lit(side.value).alias("side"))
        .select("running", "value", "side", "n")
    )


def _compute_bin_means(
    r: NDArray[np.float64],
    values: NDArray[np.float64],
    *,
    n_bins: int,
) -> pl.DataFrame:
    """Bin the running variable separately on either side of the cutoff."""
    return pl.concat(
        [
            _compute_side_bin_means(r, values, side=_Side.LEFT, n_bins=n_bins),
            _compute_side_bin_means(r, values, side=_Side.RIGHT, n_bins=n_bins),
        ]
    )


def _compute_local_linear_line(
    r: NDArray[np.float64],
    values: NDArray[np.float64],
    *,
    side: _Side,
    local_polynomial_config: LocalPolynomialConfig | None,
    n_points: int = 100,
) -> pl.DataFrame:
    """Fit a weighted local-linear line on one side of the cutoff."""
    mask = r >= 0.0 if side is _Side.RIGHT else r < 0.0
    if local_polynomial_config is not None:
        mask &= np.abs(r) <= local_polynomial_config.bandwidth
    if mask.sum() < 2:
        return pl.DataFrame(schema={"running": pl.Float64, "value": pl.Float64, "side": pl.String})

    side_r = r[mask]
    side_values = values[mask]
    weights = (
        local_polynomial_config.kernel._compute_weights(side_r / local_polynomial_config.bandwidth)
        if local_polynomial_config is not None
        else np.ones_like(side_r)
    )
    weights = weights / weights.max()
    design = np.column_stack((np.ones(side_r.size), side_r))
    coefficients = np.linalg.lstsq(
        design * np.sqrt(weights)[:, None],
        side_values * np.sqrt(weights),
        rcond=None,
    )[0]

    running_grid = np.linspace(side_r.min(), side_r.max(), n_points)
    fitted = coefficients[0] + coefficients[1] * running_grid
    return pl.DataFrame(
        {
            "running": running_grid,
            "value": fitted,
            "side": side.value,
        }
    )


def _restore_running_scale(frame: pl.DataFrame, cutoff: float) -> pl.DataFrame:
    """Convert centred running-variable values back to the original scale."""
    return frame.with_columns((pl.col("running") + cutoff).alias("running"))


def _make_design_chart(
    points: pl.DataFrame,
    fits: pl.DataFrame,
    running: RunningVariableConfig,
    *,
    y_title: str,
    title: str,
    bandwidth: float | None,
    y_domain: tuple[float, float] | None = None,
) -> AltairChart:
    """Layer binned points, local-linear fits, the cutoff and the bandwidth window."""
    layers: list[alt.Chart] = []

    if bandwidth is not None:
        window = pl.DataFrame(
            {
                "x0": [running.cutoff - bandwidth],
                "x1": [running.cutoff + bandwidth],
            }
        )
        layers.append(
            alt.Chart(window)
            .mark_rect(opacity=0.08)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("x0:Q"), x2="x1:Q"
            )
        )

    y_encoding = alt.Y("value:Q", title=y_title)
    if y_domain is not None:
        y_encoding = alt.Y("value:Q", title=y_title, scale=alt.Scale(domain=y_domain))

    points_chart = (
        alt.Chart(points)  # ty: ignore[unresolved-attribute]
        .mark_circle(size=55, opacity=0.9)
        .encode(
            x=alt.X("running:Q", title=running.name),
            y=y_encoding,
            color=alt.Color(
                "side:N",
                scale=alt.Scale(domain=_SIDE_DOMAIN, range=_SIDE_RANGE),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("side:N"),
                alt.Tooltip("running:Q", title=running.name),
                alt.Tooltip("value:Q", title=y_title),
                alt.Tooltip("n:Q", title="Observations"),
            ],
        )
    )
    layers.append(points_chart)

    if not fits.is_empty():
        layers.append(
            alt.Chart(fits)  # ty: ignore[unresolved-attribute]
            .mark_line(strokeWidth=2.5)
            .encode(
                x=alt.X("running:Q"),
                y=alt.Y("value:Q"),
                color=alt.Color(
                    "side:N",
                    scale=alt.Scale(domain=_SIDE_DOMAIN, range=_SIDE_RANGE),
                    legend=None,
                ),
            )
        )

    layers.append(
        alt.Chart(pl.DataFrame({"cutoff": [running.cutoff]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[4, 4], color="#333333")
        .encode(x=alt.X("cutoff:Q"))
    )

    return alt.layer(*layers).properties(title=title, width=640, height=320)


def plot_rdd_design(
    ds: pl.DataFrame,
    outcome: ColumnName,
    running: RunningVariableConfig,
    *,
    n_bins: int = 30,
    local_polynomial_config: LocalPolynomialConfig | None = None,
    show_fit: bool = True,
) -> AltairChart:
    """Plot the outcome against the running variable around the cutoff.

    The chart shows binned outcome means on either side of the cutoff and,
    optionally, local-linear fits. Supplying ``bandwidth`` shades the estimation
    window and restricts the fitted lines to it.

    Parameters
    ----------
    ds : polars.DataFrame
        Input data.
    outcome : ColumnName
        Name of the outcome column.
    running : `RunningVariableConfig`
        Running variable and cutoff.
    n_bins : int, default=30
        Number of bins on each side of the cutoff.
    local_polynomial_config : `LocalPolynomialConfig`, optional
        Bandwidth and kernel for the local-linear fits. When omitted, the
        fits use the full support on each side and no bandwidth is shaded.
    show_fit : bool, default=True
        Whether to draw the local-linear fits.

    Returns
    -------
    altair.Chart
        Design plot of the outcome against the running variable.
    """
    check_columns_are_in_dataset(ds, [outcome, running.name])
    check_missing_values_in_dataset(ds, [outcome, running.name])
    check_outcome(ds[outcome])
    check_numeric(ds[running.name])
    if n_bins < 2:
        raise ValueError("`n_bins` must be at least 2.")
    bandwidth = local_polynomial_config.bandwidth if local_polynomial_config is not None else None

    r = ds[running.name].to_numpy() - running.cutoff
    y = ds[outcome].to_numpy()
    points = _restore_running_scale(_compute_bin_means(r, y, n_bins=n_bins), running.cutoff)
    fits = (
        _restore_running_scale(
            pl.concat(
                [
                    _compute_local_linear_line(
                        r, y, side=_Side.LEFT, local_polynomial_config=local_polynomial_config
                    ),
                    _compute_local_linear_line(
                        r, y, side=_Side.RIGHT, local_polynomial_config=local_polynomial_config
                    ),
                ]
            ),
            running.cutoff,
        )
        if show_fit
        else pl.DataFrame(schema={"running": pl.Float64, "value": pl.Float64, "side": pl.String})
    )
    return _make_design_chart(
        points,
        fits,
        running,
        y_title=outcome,
        title=f"{outcome} vs. {running.name}",
        bandwidth=bandwidth,
    )


def plot_rdd_treatment(
    ds: pl.DataFrame,
    treatment: ColumnName,
    running: RunningVariableConfig,
    *,
    n_bins: int = 30,
    local_polynomial_config: LocalPolynomialConfig | None = None,
    show_fit: bool = True,
) -> AltairChart:
    """Plot the treatment probability against the running variable.

    This is the first-stage plot of the RD design. It shows binned treatment
    rates on either side of the cutoff and, optionally, local-linear fits.

    Parameters
    ----------
    ds : polars.DataFrame
        Input data.
    treatment : ColumnName
        Name of the binary treatment column.
    running : `RunningVariableConfig`
        Running variable and cutoff.
    n_bins : int, default=30
        Number of bins on each side of the cutoff.
    local_polynomial_config : `LocalPolynomialConfig`, optional
        Bandwidth and kernel for the local-linear fits. When omitted, the
        fits use the full support on each side and no bandwidth is shaded.
    show_fit : bool, default=True
        Whether to draw the local-linear fits.

    Returns
    -------
    altair.Chart
        First-stage plot of the treatment probability against the running
        variable.
    """
    check_columns_are_in_dataset(ds, [treatment, running.name])
    check_missing_values_in_dataset(ds, [treatment, running.name])
    check_treatment_is_binary(ds[treatment])
    check_numeric(ds[running.name])
    if n_bins < 2:
        raise ValueError("`n_bins` must be at least 2.")
    bandwidth = local_polynomial_config.bandwidth if local_polynomial_config is not None else None

    r = ds[running.name].to_numpy() - running.cutoff
    d = ds[treatment].to_numpy().astype(float)
    points = _restore_running_scale(_compute_bin_means(r, d, n_bins=n_bins), running.cutoff)
    fits = (
        _restore_running_scale(
            pl.concat(
                [
                    _compute_local_linear_line(
                        r, d, side=_Side.LEFT, local_polynomial_config=local_polynomial_config
                    ),
                    _compute_local_linear_line(
                        r, d, side=_Side.RIGHT, local_polynomial_config=local_polynomial_config
                    ),
                ]
            ),
            running.cutoff,
        )
        if show_fit
        else pl.DataFrame(schema={"running": pl.Float64, "value": pl.Float64, "side": pl.String})
    )
    return _make_design_chart(
        points,
        fits,
        running,
        y_title="Treatment probability",
        title=f"Treatment probability vs. {running.name}",
        bandwidth=bandwidth,
        y_domain=(0.0, 1.0),
    )
