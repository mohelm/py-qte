"""Result container for the local quantile treatment effect estimator."""

from dataclasses import dataclass

from qte.results import _BasicQteResult


@dataclass
class LocalQteResult(_BasicQteResult):
    """Result of a local quantile treatment effect estimation.

    Use `plot`, `tabulate`, `summarize` or `get_as_dataframe` to present the
    estimates. Instances are returned by `estimate_local_effects`; the
    constructor is not part of the public API.

    Attributes
    ----------
    qtt : polars.DataFrame
        Quantile-specific effects, one row per quantile.
    att : polars.DataFrame
        Average effects for compliers (the local average treatment effect).
    outcome : ColumnName
        Outcome variable the effect was estimated for.
    group : tuple[str, ...] or None
        Columns the estimates are grouped by.
    complier_share : float
        Estimated share of compliers, ``E[kappa]``.
    first_stage : float
        Difference in treatment take-up between instrument arms,
        ``E[D | Z = 1] - E[D | Z = 0]``.
    """

    complier_share: float
    first_stage: float
