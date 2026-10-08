"""Datasets bundled with py-qte."""

import importlib.resources
from typing import Literal

import polars as pl


def _make_lalonde_filename(
    controls_source: Literal["psid", "experiment"], use_panel_structure: bool
) -> str:
    ds_name = ["lalonde"]
    controls_source_in_filename = "exp" if controls_source == "experiment" else controls_source
    ds_name.append(controls_source_in_filename)

    if use_panel_structure:
        ds_name.append("panel")

    return "_".join(ds_name) + ".parquet"


def load_lalonde(
    controls_source: Literal["psid", "experiment"] = "psid",
    use_panel_structure: bool = False,
) -> pl.DataFrame:
    """Load the LaLonde (1986) job training dataset.

    Parameters
    ----------
    controls_source : {"psid", "experiment"}, default="psid"
        Source of the control group: the PSID comparison sample or the
        experimental controls.
    use_panel_structure : bool, default=False
        Return the panel version of the data instead of the cross-section.

    Returns
    -------
    polars.DataFrame
        The requested LaLonde dataset.
    """
    with importlib.resources.path(
        "qte.datasets.lalonde",
        _make_lalonde_filename(controls_source, use_panel_structure),
    ) as path:
        return pl.read_parquet(path)


def load_mpdta() -> pl.DataFrame:
    """Load the minimum wage panel dataset (mpdta).

    County-level teen employment data from 2003 to 2007, used in the ``did`` and
    ``qte`` R packages.

    Returns
    -------
    polars.DataFrame
        County-year panel of teen employment and minimum wage variables.
    """
    with importlib.resources.path("qte.datasets", "mpdta.parquet") as path:
        return pl.read_parquet(path)


def load_engel_with_weights() -> pl.DataFrame:
    """Load Engel (1857) food expenditure data with a sampling weights column.

    Mainly used to compare weighted and unweighted quantile regression against
    ``quantreg::rq`` in R. The ``w`` column is the inverse probability of
    selection, where the selection probability decreases with income.

    Returns
    -------
    polars.DataFrame
        Food expenditure data with an additional ``w`` column.
    """
    with importlib.resources.path("qte.datasets", "engel_with_weights.parquet") as path:
        return pl.read_parquet(path)


def load_card(treatment_cutoff: int = 16) -> pl.DataFrame:
    """Load the Card (1995) college-proximity dataset.

    The data cover 3010 men from the 1976 National Longitudinal Survey of Young
    Men. The classical instrument ``nearc4`` indicates whether a man grew up in
    a local labour market with a four-year college, and ``educ`` records years
    of schooling. Because the local quantile treatment effect requires a binary
    treatment, a column ``college`` equal to ``1{educ >= treatment_cutoff}`` is
    appended.

    Original source: ``wooldridge::card`` (data used in Card, 1995, "Using
    Geographic Variation in College Proximity to Estimate the Return to
    Schooling"). Bundled from the Rdatasets mirror.

    Parameters
    ----------
    treatment_cutoff : int, default=16
        Years of schooling at or above which ``college`` is one. The default
        treats completed college (16 years) as the treatment.

    Returns
    -------
    polars.DataFrame
        The Card data with an additional binary ``college`` column.
    """
    with importlib.resources.path("qte.datasets", "card.parquet") as path:
        ds = pl.read_parquet(path)
    return ds.with_columns((pl.col("educ") >= treatment_cutoff).cast(pl.Int8).alias("college"))


def load_jtpa() -> pl.DataFrame:
    """Load the JTPA job-training data used in Abadie, Angrist & Imbens (2002).

    The Job Training Partnership Act (JTPA) study randomly assigned applicants
    to a treatment group eligible for training or to a control group. The
    binary instrument ``assignmt`` records assignment, ``training`` records
    actual enrolment and ``earnings`` is total 30-month earnings. Non-compliance
    is almost entirely one-sided: only 54 of 3717 controls enrol in training.
    The data are the 11,204-observation extract distributed with the Stata
    ``ivqte`` documentation (Froelich & Melly, 2010).

    Returns
    -------
    polars.DataFrame
        Individual-level JTPA data with assignment, training and earnings.
    """
    with importlib.resources.path("qte.datasets", "jtpa.parquet") as path:
        return pl.read_parquet(path)
