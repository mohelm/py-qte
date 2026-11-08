"""Fail-fast input checks for the public estimators."""

from collections.abc import Sequence

import numpy as np
import polars as pl
import polars.selectors as cs
from formulaic import Formula
from formulaic.utils.variables import Variable, get_required_variables
from numpy.typing import ArrayLike, NDArray

from qte.custom_types import ColumnName, FormulaRhs


class BinaryTreatmentError(Exception): ...


class MissingValuesError(Exception): ...


class MissingColumnsError(Exception): ...


class InvalidQuantilesError(Exception): ...


class InvalidWeightsError(Exception): ...


def check_treatment_is_binary(s: pl.Series) -> None:
    values = set(s.unique().to_list())
    if values != {0, 1}:
        raise BinaryTreatmentError(f"Treatment {s.name!r} must be 0/1, got {values}.")


def check_outcome(s: pl.Series) -> None:
    if not s.dtype.is_numeric() or s.null_count() or not s.is_finite().all():
        raise ValueError(f"Outcome {s.name!r} must be numeric and finite.")


def check_grid(grid: int | ArrayLike) -> None:
    if isinstance(grid, (int, np.integer)):
        if grid < 2:
            raise ValueError("Integer `grid` must be at least 2.")
        return

    qs = np.asarray(grid, dtype=float)
    if (
        qs.ndim != 1
        or not qs.size
        or not np.isfinite(qs).all()
        or ((qs <= 0) | (qs >= 1)).any()
        or np.unique(qs).size != qs.size
    ):
        raise ValueError("Explicit `grid` must be unique, finite and inside (0, 1).")


def check_columns_are_in_dataset(ds: pl.DataFrame, required_columns: Sequence[ColumnName]) -> None:
    if not isinstance(ds, pl.DataFrame):
        raise TypeError(f"`ds` must be a polars.DataFrame, got {type(ds).__name__}.")
    if ds.height == 0:
        raise ValueError("`ds` must contain at least one row.")
    missing = [c for c in required_columns if c not in ds.columns]
    if missing:
        raise MissingColumnsError(
            "The following required columns are missing from your dataset:\n"
            + ",\n".join(repr(c) for c in missing)
            + "\n\nAvailable columns (alphabetical):\n"
            + ", ".join(sorted(ds.columns))
        )


def get_formula_columns(formula: FormulaRhs) -> list[ColumnName]:
    """Return the data columns referenced by a right-hand-side formula."""
    columns: list[ColumnName] = []
    for term in Formula.from_spec(f"~ {formula}"):
        for factor in term.factors:
            if factor.eval_method.value == "lookup":
                names = {factor.expr}
            else:
                try:
                    names = {
                        str(v)
                        for v in get_required_variables(factor.expr)
                        if Variable.Role.VALUE in v.roles
                    }
                except (SyntaxError, TypeError, ValueError):
                    names = set()
                names = names or {str(v) for v in factor.required_variables}
            columns.extend(names)
    return list(dict.fromkeys(columns))


def check_missing_values_in_dataset(ds: pl.DataFrame, columns: Sequence[ColumnName]) -> None:
    nulls = (
        ds.select(cs.by_name(columns).null_count())
        .unpivot(variable_name="Column", value_name="Nulls")
        .filter(pl.col("Nulls") > 0)
    )
    if not nulls.is_empty():
        raise MissingValuesError(f"Missing values detected:\n\n{nulls}")


def check_quantiles(qs: NDArray[np.float64]) -> None:
    if (
        qs.ndim != 1
        or not qs.size
        or not np.isfinite(qs).all()
        or ((qs <= 0) | (qs >= 1)).any()
        or np.unique(qs).size != qs.size
    ):
        raise InvalidQuantilesError(f"Invalid quantiles: {qs.tolist()}")


def check_weights(s: pl.Series) -> None:
    if (
        not s.dtype.is_numeric()
        or s.null_count()
        or not s.is_finite().all()
        or (s < 0).any()
        or s.sum() <= 0
    ):
        raise InvalidWeightsError(f"Invalid weights in {s.name!r}.")


class InvalidTreatmentGroupError(Exception): ...


class DuplicateUnitTimeError(Exception): ...


class TimeVaryingTreatmentGroupError(Exception): ...


class UnbalancedPanelError(Exception): ...


def check_numeric(s: pl.Series) -> None:
    """Check that a column is numeric, finite and free of missing values."""
    if not s.dtype.is_numeric() or s.null_count() or not s.is_finite().all():
        raise ValueError(f"{s.name!r} must be numeric and finite.")


def check_treatment_group(s: pl.Series) -> None:
    """Check that a treatment-group column is numeric, with ``inf`` marking never-treated units."""
    if not s.dtype.is_numeric():
        raise InvalidTreatmentGroupError(
            f"Treatment group {s.name!r} must be numeric, got {s.dtype}. Map your never-treated "
            "code to +inf, either in the data or through `TrtGroupConfig`."
        )
    groups = s.cast(pl.Float64)
    if groups.null_count() or groups.is_nan().any() or (groups.is_infinite() & (groups < 0)).any():
        raise InvalidTreatmentGroupError(
            f"Treatment group {s.name!r} must not contain missing values or negative infinity; "
            "use +inf for units that are never treated."
        )


def check_unit_time_is_unique(ds: pl.DataFrame, unit: ColumnName, time: ColumnName) -> None:
    """Check that every unit appears at most once per period, as the panel estimators require."""
    duplicated = (
        ds.group_by([unit, time]).len().filter(pl.col("len") > 1).sort([unit, time]).head(10)
    )
    if not duplicated.is_empty():
        raise DuplicateUnitTimeError(
            f"Units must appear at most once per period in {time!r}, but {unit!r} repeats within "
            "a period:\n\n"
            f"{duplicated}"
        )


def check_treatment_group_is_time_invariant(
    ds: pl.DataFrame, unit: ColumnName, treatment_group: ColumnName
) -> None:
    """Check that a unit's treatment group does not change over its observations."""
    varying = (
        ds.group_by(unit)
        .agg(pl.col(treatment_group).n_unique().alias("n_groups"))
        .filter(pl.col("n_groups") > 1)
        .sort(unit)
        .head(10)
    )
    if not varying.is_empty():
        raise TimeVaryingTreatmentGroupError(
            f"Treatment group {treatment_group!r} must be constant within {unit!r}, but it varies "
            "for these units:\n\n"
            f"{varying}"
        )


def check_units_are_observed_in_both_periods(earlier: NDArray, later: NDArray) -> None:
    """Check that two period cells of a group-time cell contain the same units.

    Estimators that form a unit's change over time from its outcomes in two periods need every unit
    of the cell they are indexed on to appear in both of them. Neither the data alone nor a
    dataset-wide check can establish this: which units a cell contains depends on the treatment
    group, the period, the comparison group and the anticipation window.
    """
    if len(earlier) == len(later) and np.array_equal(np.sort(earlier), np.sort(later)):
        return
    raise UnbalancedPanelError(
        "Every unit must be observed in both periods of a group-time cell to form its change over "
        f"time, but the earlier period has {len(earlier)} units and the later one has "
        f"{len(later)}. Drop units that are not observed in both periods, or use an estimator that "
        "does not need unit-level changes."
    )
