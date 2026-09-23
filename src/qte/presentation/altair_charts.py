import altair as alt
import polars as pl

from qte.names import CI_LB_ID, CI_UB_ID, EFFECT_ID, QUANTILE_ID

_ESTIMATE_TYPE_ORDER = ["qte", "att"]
_ESTIAMTE_TYPE_COLOR = alt.Scale(domain=_ESTIMATE_TYPE_ORDER, range=["black", "red"])
_EFFECT_STROKE = (1, 0)
_CONFIDENCE_INTERVAL_STROKE = (8, 8)

_LINE_CONFIG_IN_PLOT_WITH_MULTIPLE_QUANTILES = [
    (EFFECT_ID, _EFFECT_STROKE),
    (CI_LB_ID, _CONFIDENCE_INTERVAL_STROKE),
    (CI_UB_ID, _CONFIDENCE_INTERVAL_STROKE),
]


def _plot_against_quantiles(
    base: alt.Chart,
    y: str,
    *,
    tick_values: list[float],
    stroke_dash: tuple[int, int],
    x_label_pos: str | None,
) -> alt.LayerChart | alt.FacetChart:
    x = alt.X(
        f"{QUANTILE_ID}:Q",
        title="Quantile",
        axis=alt.Axis(
            titleOpacity=alt.expr(x_label_pos) if x_label_pos is not None else alt.Undefined,
            titleFontWeight="normal",
            values=tick_values,
            format=".3~f",  # At MOST three decimals
            titleAnchor="start",
        ),
    )

    line_chart = base.mark_line(color="black", strokeDash=list(stroke_dash)).encode(  # ty: ignore[unresolved-attribute]
        x=x, y=alt.Y(y, title="")
    )
    point_chart = base.mark_point(color="black", fill="black").encode(  # ty: ignore[unresolved-attribute]
        x=x, y=alt.Y(y, title="")
    )
    return alt.layer(line_chart, point_chart)


def _make_rule_layer(
    base: alt.Chart, y: str, *, stroke_dash: tuple[int, int] = (1, 0)
) -> alt.Chart:
    return base.mark_rule(color="red", strokeDash=stroke_dash).encode(y=alt.Y(f"{y}:Q"))  # ty: ignore[unresolved-attribute]


def _make_facets(grouped_by: tuple[str] | tuple[str, str]) -> dict:
    # If there is a single group (tuple[str] case), we want to put the associated dimension in the
    # column. If there are two group, we put the the first group in the rows and the second in the
    # columns. The call has then to ensure that the groups order reflect how the plot should look
    # like.
    n_grouped_by = len(grouped_by)
    row_group = 0 if n_grouped_by == 1 else 1
    facets = {"column": alt.Column(f"{grouped_by[row_group]}:N")}
    if n_grouped_by > 1:
        facets["row"] = alt.Row(f"{grouped_by[0]}:N")
    return facets


def _make_plot_with_multiple_quantiles(
    combined: pl.DataFrame, group: tuple[str] | tuple[str, str] | None, *, tick_values: list[float]
) -> alt.LayerChart | alt.FacetChart:
    base = alt.Chart(combined)

    # In facetted charts we only want to show a single label on the x-axis (the quantile axis).
    # In Altair, that is not totally simple. We opt for only showing the leftmost label. We achieve
    # this by essentially making the other labels transparent. For this we need a query that
    # identifies the left most panel in the column dimension (not the row dimension is always
    # irrelevan t here). The column dimension is always the last one.
    x_label_position_query = (
        None if group is None else f"parent['{group[-1]}'] == {combined[group[-1]].min()} ? 1 : 0"
    )
    qtes_ds = base.transform_filter(alt.datum.__kind == "qte")
    qte_layer = alt.layer(*[
        _plot_against_quantiles(
            qtes_ds, y, tick_values=tick_values, stroke_dash=sd, x_label_pos=x_label_position_query
        )
        for (y, sd) in _LINE_CONFIG_IN_PLOT_WITH_MULTIPLE_QUANTILES
    ])

    atts_ds = base.transform_filter(alt.datum.__kind == "att")
    att_layer = alt.layer(*[
        _make_rule_layer(atts_ds, y, stroke_dash=sd)
        for y, sd in _LINE_CONFIG_IN_PLOT_WITH_MULTIPLE_QUANTILES
    ])

    layer = alt.layer(qte_layer, att_layer)
    return layer if group is None else layer.facet(**_make_facets(group)).configure_facet(spacing=1)


def _make_plot_for_single_quantile(
    combined: pl.DataFrame, group: tuple[str] | tuple[str, str] | None, *, quantile: float
) -> alt.LayerChart | alt.FacetChart:
    base = alt.Chart(combined)
    x = alt.X("__kind", title=None, sort=_ESTIMATE_TYPE_ORDER)
    color = alt.Color("__kind:N", scale=_ESTIAMTE_TYPE_COLOR, legend=None)
    title = alt.Title(
        text="Quantile and Average Treatment Effects",
        subtitle=f"The QTE is measured for {quantile=}.",
        offset=10,
    )
    layer = alt.layer(
        base.mark_point(filled=True).encode(  # ty: ignore[unresolved-attribute]
            x=x, y=alt.Y(EFFECT_ID, title=None), color=color
        ),
        base.mark_rule().encode(  # ty: ignore[unresolved-attribute]
            x=x,
            y=alt.Y(CI_LB_ID, title=None),
            y2=alt.Y2(CI_UB_ID, title=None),
            color=color,
        ),
        base.mark_point(shape="stroke", size=150, strokeWidth=3).encode(  # ty: ignore[unresolved-attribute]
            x=x, y=alt.Y(CI_LB_ID, title=None), color=color
        ),
        base.mark_point(shape="stroke", size=150, strokeWidth=3).encode(  # ty: ignore[unresolved-attribute]
            x=x, y=alt.Y(CI_UB_ID, title=None), color=color
        ),
    )
    return (
        layer if group is None else layer.facet(**_make_facets(group)).configure_facet(spacing=1)
    ).properties(title=title)


def make_plot(
    qtes: pl.DataFrame, atts: pl.DataFrame, group: tuple[str] | tuple[str, str] | None = None
) -> alt.LayerChart | alt.FacetChart:

    combined = pl.concat(
        (d.with_columns(pl.lit(k).alias("__kind")) for d, k in [(qtes, "qte"), (atts, "att")]),
        how="diagonal",
    )
    quantiles = qtes[QUANTILE_ID].unique().sort()
    n_quantiles = quantiles.shape[0]
    return (
        _make_plot_with_multiple_quantiles(combined, group, tick_values=quantiles.to_list())
        if n_quantiles > 1
        else _make_plot_for_single_quantile(combined, group, quantile=quantiles.item())
    )
