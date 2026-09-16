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

    with importlib.resources.path(
        "qte.datasets.lalonde",
        _make_lalonde_filename(controls_source, use_panel_structure),
    ) as path:
        return pl.read_parquet(path)


def load_mpdta() -> pl.DataFrame:
    """Load the minimum wage panel dataset (mpdta) used in the did and qte packages.

    Contains county-level teen employment data from 2003-2007.
    """
    with importlib.resources.path("qte.datasets", "mpdta.parquet") as path:
        return pl.read_parquet(path)


def load_engel_with_with_weights() -> pl.DataFrame:
    """Load Engel (1857) food expenditure data with an additional weights column.

    It is mainly used for comparing the quantile regression with and without weights against the
    results obtained from quantreg::rq in R. The columns were generated as

    ```{r}
    library(dplyr)
    data(engel)
    engel_weighted <- engel %>%
    mutate(
    # Probability of selection decreases as income grows
    prob_selection = 1 / (1 + exp((income - mean(income)) / sd(income))),
    # Sampling weight is the inverse selection probability
    w = 1 / prob_selection,

    log_income = log(income),
    log_foodexp = log(foodexp)
    )
    ```

    """
    with importlib.resources.path("qte.datasets", "engel_with_weights.parquet") as path:
        return pl.read_parquet(path)
