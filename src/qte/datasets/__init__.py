import importlib.resources
from typing import Literal

import polars as pl


def _make_lalonde_filename(
    controls_source: Literal["psid", "experiment"], use_panel_structure: bool
):
    ds_name = ["lalonde"]
    controls_source_in_filename = (
        "exp" if controls_source == "experiment" else controls_source
    )
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
