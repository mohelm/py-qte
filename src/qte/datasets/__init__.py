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
