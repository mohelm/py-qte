"""Descriptive statistics for the instrumental-variables estimators."""

from collections.abc import Iterator, Mapping, Sequence
from enum import StrEnum
from functools import partial
from typing import NamedTuple

import polars as pl
from great_tables import GT, loc, md, style
from rich import box
from rich.table import Table

type _CellValue = float | int | None


class _RowType(StrEnum):
    N_OBS = "n_obs"
    MEAN = "mean"
    SD = "sd"


class _RowMeta(NamedTuple):
    """Formatting metadata for one row in the display frame."""

    row: int
    kind: _RowType
    decimals: int


class _RawTableRow(NamedTuple):
    """One fully assembled row of the descriptive-statistics table."""

    group: str
    variable: str
    entire: _CellValue
    assign_treated: _CellValue
    assign_control: _CellValue
    assign_diff: _CellValue
    treat_treated: _CellValue
    treat_control: _CellValue
    treat_diff: _CellValue


_ENTIRE_ID = "entire"
_ASSIGN_TREATED_ID = "assign_treated"
_ASSIGN_CONTROL_ID = "assign_control"
_ASSIGN_DIFF_ID = "assign_diff"
_TREAT_TREATED_ID = "treat_treated"
_TREAT_CONTROL_ID = "treat_control"
_TREAT_DIFF_ID = "treat_diff"

_VALUE_IDS = list(_RawTableRow._fields[2:])
_DIFF_IDS = [_ASSIGN_DIFF_ID, _TREAT_DIFF_ID]
_FOOTNOTE = (
    "Standard deviations are shown in square brackets. Differences in means are followed "
    "by Welch t-statistics in parentheses."
)
_NON_DIFF_IDS = [
    _ENTIRE_ID,
    _ASSIGN_TREATED_ID,
    _ASSIGN_CONTROL_ID,
    _TREAT_TREATED_ID,
    _TREAT_CONTROL_ID,
]


class OutputFormat(StrEnum):
    great_tables = "GREAT_TABLE"
    rich_table = "RICH_TABLE"


def _get_nicer_name(name: str) -> str:
    return name.replace("_", " ").capitalize()


def _welch_t_stat_expr(prefix: str) -> pl.Expr:
    """Welch t-statistic from already-aggregated ``prefix`` columns."""
    treated_mean = pl.col(f"{prefix}_treated_mean")
    control_mean = pl.col(f"{prefix}_control_mean")
    treated_var = pl.col(f"{prefix}_treated_var")
    control_var = pl.col(f"{prefix}_control_var")
    treated_n = pl.col(f"{prefix}_treated_n")
    control_n = pl.col(f"{prefix}_control_n")

    standard_error = (treated_var / treated_n + control_var / control_n).sqrt()
    valid = (treated_n >= 2) & (control_n >= 2) & ~((treated_var == 0) & (control_var == 0))
    return (
        pl.when(valid)
        .then((treated_mean - control_mean) / standard_error)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )


def _format_decimals(mean: float | None, sd: float | None) -> int:
    scale = max(abs(mean or 0.0), abs(sd or 0.0))
    return 0 if scale >= 1_000 else 2


def _format_t_stat(value: float | None) -> str:
    return "" if value is None else f"({value:.2f})"


def _format_standard_deviation(value: float | None, decimals: int) -> str:
    return "" if value is None else f"[{value:,.{decimals}f}]"


def _get_variable_stats(
    ds: pl.DataFrame,
    variable: str,
    treatment: str,
    instrument: str,
    *,
    group: str,
    label: str,
) -> tuple[_RawTableRow, _RawTableRow]:
    """Build the mean and SD rows for one variable."""
    value = pl.col(variable)
    assigned = pl.col(instrument) == 1
    not_assigned = pl.col(instrument) == 0
    treated = pl.col(treatment) == 1
    not_treated = pl.col(treatment) == 0

    stats = (
        ds.select(
            value.mean().alias("entire_sample_mean"),
            value.std().alias("entire_sample_sd"),
            value.filter(assigned).mean().alias("assignment_treated_mean"),
            value.filter(assigned).std().alias("assignment_treated_sd"),
            value.filter(assigned).var().alias("assignment_treated_var"),
            value.filter(assigned).count().alias("assignment_treated_n"),
            value.filter(not_assigned).mean().alias("assignment_control_mean"),
            value.filter(not_assigned).std().alias("assignment_control_sd"),
            value.filter(not_assigned).var().alias("assignment_control_var"),
            value.filter(not_assigned).count().alias("assignment_control_n"),
            value.filter(treated).mean().alias("treatment_treated_mean"),
            value.filter(treated).std().alias("treatment_treated_sd"),
            value.filter(treated).var().alias("treatment_treated_var"),
            value.filter(treated).count().alias("treatment_treated_n"),
            value.filter(not_treated).mean().alias("treatment_control_mean"),
            value.filter(not_treated).std().alias("treatment_control_sd"),
            value.filter(not_treated).var().alias("treatment_control_var"),
            value.filter(not_treated).count().alias("treatment_control_n"),
        )
        .with_columns(
            (pl.col("assignment_treated_mean") - pl.col("assignment_control_mean")).alias(
                "assignment_diff"
            ),
            _welch_t_stat_expr("assignment").alias("assignment_t_stat"),
            (pl.col("treatment_treated_mean") - pl.col("treatment_control_mean")).alias(
                "treatment_diff"
            ),
            _welch_t_stat_expr("treatment").alias("treatment_t_stat"),
        )
        .row(0, named=True)
    )

    means = _RawTableRow(
        group,
        label,
        stats["entire_sample_mean"],
        stats["assignment_treated_mean"],
        stats["assignment_control_mean"],
        stats["assignment_diff"],
        stats["treatment_treated_mean"],
        stats["treatment_control_mean"],
        stats["treatment_diff"],
    )
    sds = _RawTableRow(
        group,
        "",
        stats["entire_sample_sd"],
        stats["assignment_treated_sd"],
        stats["assignment_control_sd"],
        stats["assignment_t_stat"],
        stats["treatment_treated_sd"],
        stats["treatment_control_sd"],
        stats["treatment_t_stat"],
    )
    return means, sds


def _default_covariates(
    ds: pl.DataFrame, outcome: str, treatment: str, instrument: str
) -> Iterator[str]:
    return (
        name
        for name, dtype in ds.schema.items()
        if dtype.is_numeric() and name not in {outcome, treatment, instrument}
    )


def _make_covariate_rows(
    outcome: str,
    treatment: str,
    covariates: Sequence[str],
    labels: Mapping[str, str],
) -> Iterator[tuple[str, str, str]]:
    """Yield the ``(group, covariate, label)`` specs for the table rows."""
    yield ("Treatment", treatment, labels.get(treatment, _get_nicer_name(treatment)))
    yield ("Outcome variable", outcome, labels.get(outcome, _get_nicer_name(outcome)))
    for covariate in covariates:
        yield (
            "Baseline Characteristics",
            covariate,
            labels.get(covariate, _get_nicer_name(covariate)),
        )


def _make_display_frame(
    ds: pl.DataFrame,
    outcome: str,
    treatment: str,
    instrument: str,
    covariates: Sequence[str],
    labels: Mapping[str, str],
) -> tuple[pl.DataFrame, list[_RowMeta]]:
    """Assemble the two-row-per-column display frame and its row metadata.

    The metadata entries are ``(row_index, kind, decimals)`` where ``kind`` is
    a `_RowType` member.
    """
    counts = ds.select(
        pl.len().alias("entire"),
        (pl.col(instrument) == 1).sum().alias("assigned"),
        (pl.col(instrument) == 0).sum().alias("not_assigned"),
        (pl.col(treatment) == 1).sum().alias("treated"),
        (pl.col(treatment) == 0).sum().alias("not_treated"),
    ).row(0, named=True)

    rows: list[_RawTableRow] = []
    row_meta: list[_RowMeta] = []

    # Number of observations
    rows.append(
        _RawTableRow(
            "Sample",
            "Number of observations",
            counts["entire"],
            counts["assigned"],
            counts["not_assigned"],
            None,
            counts["treated"],
            counts["not_treated"],
            None,
        )
    )
    row_meta.append(_RowMeta(len(rows) - 1, _RowType.N_OBS, 0))

    for group_name, covariate, label in _make_covariate_rows(
        outcome, treatment, covariates, labels
    ):
        means, sds = _get_variable_stats(
            ds, covariate, treatment, instrument, group=group_name, label=label
        )
        decimals = _format_decimals(means.entire, sds.entire)
        start = len(rows)
        rows.extend([means, sds])
        row_meta.extend(
            [
                _RowMeta(start, _RowType.MEAN, decimals),
                _RowMeta(start + 1, _RowType.SD, decimals),
            ]
        )

    return pl.DataFrame(rows), row_meta


def _make_great_table(
    display: pl.DataFrame,
    row_meta: Sequence[_RowMeta],
    title: str,
) -> GT:
    table = (
        GT(display, rowname_col="variable", groupname_col="group")
        .tab_header(title=title)
        .tab_footnote(footnote=_FOOTNOTE)
        .tab_stubhead(label="")
        .cols_label(
            entire="Entire Sample",
            assign_treated="Treatment",
            assign_control="Control",
            assign_diff=md("Diff.<br>(t-stat.)"),
            treat_treated="Treatment",
            treat_control="Control",
            treat_diff=md("Diff.<br>(t-stat.)"),
        )
        .tab_spanner(
            label="Assignment", columns=[_ASSIGN_TREATED_ID, _ASSIGN_CONTROL_ID, _ASSIGN_DIFF_ID]
        )
        .tab_spanner(
            label="Treatment", columns=[_TREAT_TREATED_ID, _TREAT_CONTROL_ID, _TREAT_DIFF_ID]
        )
        .sub_missing(columns=_VALUE_IDS, missing_text="")
    )

    for meta in row_meta:
        table = table.fmt_number(
            columns=_VALUE_IDS, rows=meta.row, decimals=meta.decimals, use_seps=True
        )
        if meta.kind is _RowType.SD:
            table = table.fmt(
                columns=_NON_DIFF_IDS,
                rows=meta.row,
                fns=partial(_format_standard_deviation, decimals=meta.decimals),
            )

    sd_rows = [meta.row for meta in row_meta if meta.kind is _RowType.SD]
    mean_rows = [meta.row for meta in row_meta if meta.kind is _RowType.MEAN]
    table = table.fmt(
        columns=_DIFF_IDS,
        rows=sd_rows,
        fns=_format_t_stat,
    )

    # Keep each mean and its standard deviation visually together: no rule
    # between them and only a little breathing room.
    table = table.tab_style(
        style=style.css("padding-bottom: 2px;"),
        locations=[loc.body(rows=mean_rows), loc.stub(rows=mean_rows)],
    ).tab_style(
        style=[style.borders(sides="top", style="hidden"), style.css("padding-top: 2px;")],
        locations=[loc.body(rows=sd_rows), loc.stub(rows=sd_rows)],
    )

    table = table.tab_style(style=style.css("text-align: left;"), locations=loc.footnotes())

    return table.tab_style(
        style=style.borders(sides="right", style="solid", weight="1px", color="#d9d9d9"),
        locations=loc.body(columns=[_ENTIRE_ID, _ASSIGN_DIFF_ID]),
    ).tab_options(
        heading_title_font_size="16px",
        column_labels_font_weight="bold",
        row_group_font_weight="bold",
        row_group_background_color="#f8f9fa",
        table_width="100%",
    )


def _make_rich_table(
    display: pl.DataFrame,
    row_meta: Sequence[_RowMeta],
    title: str,
) -> Table:
    table = Table(
        title=title,
        caption=_FOOTNOTE,
        caption_justify="left",
        box=box.SIMPLE,
        header_style="bold",
        expand=False,
    )
    table.add_column("", justify="left")
    for label in [
        "Entire\nSample",
        "Assignment:\nTreatment",
        "Assignment:\nControl",
        "Assignment diff.\n(t-stat.)",
        "Treatment:\nTreatment",
        "Treatment:\nControl",
        "Treatment diff.\n(t-stat.)",
    ]:
        table.add_column(label, justify="right")

    row_meta_by_row = {meta.row: (meta.kind, meta.decimals) for meta in row_meta}
    previous_group = None
    for row_index, row in enumerate(display.iter_rows(named=True)):
        if row["group"] != previous_group:
            table.add_row(row["group"], style="bold")
            previous_group = row["group"]
        kind, decimals = row_meta_by_row.get(row_index, (_RowType.MEAN, 2))
        values = []
        for column in _VALUE_IDS:
            value = row[column]
            if value is None:
                values.append("")
            elif kind is _RowType.SD and column in _DIFF_IDS:
                values.append(f"({value:.2f})")
            elif kind is _RowType.SD:
                values.append(f"[{value:,.{decimals}f}]")
            elif kind is _RowType.N_OBS:
                values.append(f"{value:,.0f}")
            else:
                values.append(f"{value:,.{decimals}f}")
        table.add_row(row["variable"], *values)
    return table


def make_descriptive_statistics(
    ds: pl.DataFrame,
    outcome: str,
    treatment: str,
    instrument: str,
    *,
    output: OutputFormat = OutputFormat.great_tables,
    covariates: Sequence[str] | None = None,
    labels: Mapping[str, str] | None = None,
    title: str = "Means and Standard Deviations",
) -> GT | Table:
    """Build a table of means and standard deviations by assignment and treatment.

    The rows describe the treatment, the outcome and the requested baseline
    covariates. Each column gets a mean row and a standard-deviation row; the
    difference columns report the mean difference in the first row and the
    Welch t-statistic in parentheses in the second.

    Parameters
    ----------
    ds : polars.DataFrame
        Input data.
    outcome : str
        Outcome column.
    treatment : str
        Binary treatment column.
    instrument : str
        Binary instrument column.
    output : `OutputFormat`, default=`OutputFormat.great_tables`
        Table backend. Great Tables is the full layout; the Rich table is a
        compact terminal/notebook fallback.
    covariates : sequence of str, optional
        Baseline characteristics to list after the outcome. When omitted, all
        numeric columns other than the outcome, treatment and instrument are used.
    labels : mapping of str to str, optional
        Display labels keyed by column name.
    title : str, default="Means and Standard Deviations"
        Table title.

    Returns
    -------
    great_tables.GT or rich.table.Table
        The formatted descriptive-statistics table.
    """
    labels = labels or {}
    if covariates is None:
        covariate_names = list(_default_covariates(ds, outcome, treatment, instrument))
    else:
        covariate_names = list(covariates)
    covariate_names = [covariate for covariate in covariate_names if covariate != outcome]

    display, row_meta = _make_display_frame(
        ds, outcome, treatment, instrument, covariate_names, labels
    )
    if output == OutputFormat.rich_table:
        return _make_rich_table(display, row_meta, title)
    return _make_great_table(display, row_meta, title)


if __name__ == "__main__":
    from qte.datasets import load_card

    card = load_card()
    covariates = ["educ", "exper", "age", "black", "smsa", "south", "married"]
    labels = {
        "lwage": "Log wage",
        "college": "College",
        "educ": "Years of education",
        "exper": "Experience",
        "age": "Age",
        "black": "Black",
        "smsa": "SMSA",
        "south": "South",
        "married": "Married",
    }
    title = "Table 1. Card (1995) descriptive statistics"

    descriptive_table = make_descriptive_statistics(
        card,
        "lwage",
        "college",
        "nearc4",
        covariates=covariates,
        labels=labels,
        title=title,
    )
    if not isinstance(descriptive_table, GT):  # pragma: no cover - __main__ guard
        raise TypeError("expected a Great Tables table")
    output_path = "card_descriptives.html"
    descriptive_table.write_raw_html(output_path, make_page=True)
    print(f"Wrote {output_path}")

    rich_table = make_descriptive_statistics(
        card,
        "lwage",
        "college",
        "nearc4",
        output=OutputFormat.rich_table,
        covariates=covariates,
        labels=labels,
        title=title,
    )
    try:
        from IPython import get_ipython
        from IPython.display import display
    except ImportError:
        from rich import print as rich_print

        rich_print(rich_table)
    else:
        if get_ipython() is not None:
            display(rich_table)
        else:
            from rich import print as rich_print

            rich_print(rich_table)
