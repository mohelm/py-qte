import altair as alt
import polars as pl

from qte.names import CI_LB_ID, CI_UB_ID, EFFECT_ID, QUANTILE_ID


def _make_line_layer(
    ds, y: str, stroke_dash: tuple[int, int] = (1, 0)
) -> tuple[alt.Chart, alt.Chart]:
    x = alt.X(f"{QUANTILE_ID}:Q", title="Quantile")
    line_chart = ds.mark_line(color="black", strokeDash=list(stroke_dash)).encode(
        x=x, y=alt.Y(y, title="")
    )
    point_chart = ds.mark_point(color="black", fill="black").encode(
        x=x, y=alt.Y(y, title="")
    )
    # Add zero?
    return line_chart, point_chart


def _make_rule_layer(ds, y, stroke_dash: tuple[int, int] = (1, 0)) -> alt.Chart:
    return ds.mark_rule(color="red", strokeDash=stroke_dash).encode(y=alt.Y(f"{y}:Q"))


def make_plot(qtes: pl.DataFrame, atts: pl.DataFrame, group: str | None = None):

    # Need to have one dataset for Altair.
    combined = pl.concat(
        (
            d.with_columns(pl.lit(k).alias("__kind"))
            for d, k in [(qtes, "qte"), (atts, "att")]
        ),
        how="diagonal",
    )

    base = alt.Chart(combined)

    qtes_ds = base.transform_filter(alt.datum.__kind == "qte")
    qte_layer = alt.layer(
        *_make_line_layer(qtes_ds, EFFECT_ID),
        *_make_line_layer(qtes_ds, CI_UB_ID, stroke_dash=(8, 8)),
        *_make_line_layer(qtes_ds, CI_LB_ID, stroke_dash=(8, 8)),
    )

    atts_ds = base.transform_filter(alt.datum.__kind == "att")
    att_layer = alt.layer(
        _make_rule_layer(atts_ds, EFFECT_ID),
        _make_rule_layer(atts_ds, CI_LB_ID, stroke_dash=(8, 8)),
        _make_rule_layer(atts_ds, CI_UB_ID, stroke_dash=(8, 8)),
    )

    chart = alt.layer(qte_layer, att_layer)

    return chart if group is None else chart.facet(column=alt.Column(f"{group}:N"))
