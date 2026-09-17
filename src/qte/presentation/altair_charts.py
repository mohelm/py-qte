import altair as alt
import polars as pl

from qte.names import CI_LB_ID, CI_UB_ID, EFFECT_ID, QUANTILE_ID

_KIND_ORDER = ["qte", "att"]
_KIND_COLOR = alt.Scale(domain=_KIND_ORDER, range=["black", "red"])


def _make_line_layer(
    base: alt.Chart, y: str, tick_values: list[float], stroke_dash: tuple[int, int] = (1, 0)
) -> tuple[alt.Chart, alt.Chart]:
    x = alt.X(
        f"{QUANTILE_ID}:Q",
        axis=alt.Axis(
            title="Q",
            titleFontWeight="normal",
            values=tick_values,
            # At most 3 decimals (0.333 for 1/3); `~` trims trailing zeros.
            format=".3~f",
        ),
    )
    line_chart = base.mark_line(color="black", strokeDash=list(stroke_dash)).encode(  # ty: ignore[unresolved-attribute]
        x=x, y=alt.Y(y, title="")
    )
    point_chart = base.mark_point(color="black", fill="black").encode(  # ty: ignore[unresolved-attribute]
        x=x, y=alt.Y(y, title="")
    )
    return line_chart, point_chart


def _make_rule_layer(base: alt.Chart, y: str, stroke_dash: tuple[int, int] = (1, 0)) -> alt.Chart:
    return base.mark_rule(color="red", strokeDash=stroke_dash).encode(y=alt.Y(f"{y}:Q"))  # ty: ignore[unresolved-attribute]


def _make_plot_with_multiple_quantiles(
    combined: pl.DataFrame, group: str | None, tick_values: list[float]
) -> alt.LayerChart | alt.FacetChart:
    base = alt.Chart(combined)

    qtes_ds = base.transform_filter(alt.datum.__kind == "qte")
    qte_layer = alt.layer(
        *_make_line_layer(qtes_ds, EFFECT_ID, tick_values),
        *_make_line_layer(qtes_ds, CI_UB_ID, tick_values, stroke_dash=(8, 8)),
        *_make_line_layer(qtes_ds, CI_LB_ID, tick_values, stroke_dash=(8, 8)),
    )

    atts_ds = base.transform_filter(alt.datum.__kind == "att")
    att_layer = alt.layer(
        _make_rule_layer(atts_ds, EFFECT_ID),
        _make_rule_layer(atts_ds, CI_LB_ID, stroke_dash=(8, 8)),
        _make_rule_layer(atts_ds, CI_UB_ID, stroke_dash=(8, 8)),
    )

    layer = alt.layer(qte_layer, att_layer)
    return (
        layer
        if group is None
        else layer.facet(column=alt.Column(f"{group}:N")).configure_facet(spacing=1)
    )


def _make_plot_for_single_quantile(
    combined: pl.DataFrame, quantile: float, group: str | None
) -> alt.LayerChart | alt.FacetChart:
    base = alt.Chart(combined)
    x = alt.X("__kind", title=None, sort=_KIND_ORDER)
    color = alt.Color("__kind:N", scale=_KIND_COLOR, legend=None)
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
        layer
        if group is None
        else layer.facet(column=alt.Column(f"{group}:N")).configure_facet(spacing=1)
    ).properties(title=title)


def make_plot(
    qtes: pl.DataFrame, atts: pl.DataFrame, group: str | None = None
) -> alt.LayerChart | alt.FacetChart:

    # Need to have one dataset for Altair.
    combined = pl.concat(
        (d.with_columns(pl.lit(k).alias("__kind")) for d, k in [(qtes, "qte"), (atts, "att")]),
        how="diagonal",
    )
    quantiles = qtes[QUANTILE_ID].unique().sort()
    n_quantiles = quantiles.shape[0]
    return (
        _make_plot_with_multiple_quantiles(combined, group, quantiles.to_list())
        if n_quantiles > 1
        else _make_plot_for_single_quantile(combined, quantile=quantiles.item(), group=group)
    )
