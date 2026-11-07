import numpy as np
import polars as pl
import pytest

from qte._validation import (
    BinaryTreatmentError,
    InvalidQuantilesError,
    InvalidWeightsError,
    MissingColumnsError,
    MissingValuesError,
    check_columns_are_in_dataset,
    check_grid,
    check_missing_values_in_dataset,
    check_outcome,
    check_quantiles,
    check_treatment_is_binary,
    check_weights,
    get_formula_columns,
)
from qte.cross_sectional import (
    OutcomeRegressionConfig,
    estimate_aipw_effects,
    estimate_ipw_effects,
    estimate_outcome_regression_effects,
    estimate_unadjusted_effects,
)


@pytest.mark.parametrize(
    ("formula", "expected"),
    [
        ("1", set()),
        ("age + education", {"age", "education"}),
        ("age:region", {"age", "region"}),
        ("I(age**2) + C(region) + poly(earnings, 5)", {"age", "region", "earnings"}),
        ("`my-col` + `their region`", {"my-col", "their region"}),
        ('Q("age")', {"age"}),
        ("poly(`their region`, 3)", {"their region"}),
        ("age + I(age**2)", {"age"}),
    ],
)
def test_get_formula_columns(formula, expected):
    assert set(get_formula_columns(formula)) == expected


def test_check_columns_are_in_dataset_passes():
    ds = pl.DataFrame({"a": [1], "b": [2]})
    check_columns_are_in_dataset(ds, ["a", "b"])


def test_check_columns_are_in_dataset_rejects_wrong_type():
    with pytest.raises(TypeError):
        check_columns_are_in_dataset([], ["a"])


def test_check_columns_are_in_dataset_rejects_empty_frame():
    ds = pl.DataFrame(schema={"a": pl.Int64})
    with pytest.raises(ValueError):
        check_columns_are_in_dataset(ds, ["a"])


def test_check_columns_are_in_dataset_reports_missing_and_available():
    ds = pl.DataFrame({"b": [2], "a": [1]})
    with pytest.raises(MissingColumnsError) as exc:
        check_columns_are_in_dataset(ds, ["a", "missing"])
    message = str(exc.value)
    assert "'missing'" in message
    assert "a, b" in message


def test_check_missing_values_in_dataset():
    ds = pl.DataFrame({"a": [1.0, 2.0], "b": [1.0, None]})
    check_missing_values_in_dataset(ds, ["a"])
    with pytest.raises(MissingValuesError):
        check_missing_values_in_dataset(ds, ["a", "b"])


def test_check_treatment_is_binary():
    check_treatment_is_binary(pl.Series("d", [0, 1, 1, 0]))
    with pytest.raises(BinaryTreatmentError):
        check_treatment_is_binary(pl.Series("d", [0, 2]))
    with pytest.raises(BinaryTreatmentError):
        check_treatment_is_binary(pl.Series("d", [1, 1]))
    with pytest.raises(BinaryTreatmentError):
        check_treatment_is_binary(pl.Series("d", [0, 1, None]))


def test_check_outcome_accepts_numeric_finite():
    check_outcome(pl.Series("y", [1.0, 2.0]))


@pytest.mark.parametrize(
    "series",
    [
        pl.Series("y", ["a", "b"]),
        pl.Series("y", [1.0, None]),
        pl.Series("y", [1.0, float("inf")]),
        pl.Series("y", [1.0, float("nan")]),
    ],
)
def test_check_outcome_rejects_invalid(series):
    with pytest.raises(ValueError):
        check_outcome(series)


def test_check_grid_accepts_valid():
    check_grid(2)
    check_grid(100)
    check_grid(np.array([0.25, 0.5, 0.75]))


@pytest.mark.parametrize(
    "grid",
    [
        0,
        1,
        -1,
        np.array([0.0, 0.5]),
        np.array([0.5, 1.0]),
        np.array([0.5, 0.5]),
        np.array([0.5, np.nan]),
        np.array([]),
        np.array([[0.5]]),
    ],
)
def test_check_grid_rejects_invalid(grid):
    with pytest.raises(ValueError):
        check_grid(grid)


def test_check_quantiles_accepts_valid():
    check_quantiles(np.array([0.25, 0.5, 0.75]))


@pytest.mark.parametrize(
    "qs",
    [
        np.array([0.0, 0.5]),
        np.array([0.5, 1.0]),
        np.array([0.5, 0.5]),
        np.array([0.5, np.nan]),
        np.array([]),
        np.array([[0.5]]),
    ],
)
def test_check_quantiles_rejects_invalid(qs):
    with pytest.raises(InvalidQuantilesError):
        check_quantiles(qs)


def test_check_weights_accepts_valid():
    check_weights(pl.Series("w", [1.0, 2.0]))


@pytest.mark.parametrize(
    "series",
    [
        pl.Series("w", ["a", "b"]),
        pl.Series("w", [1.0, None]),
        pl.Series("w", [1.0, -1.0]),
        pl.Series("w", [0.0, 0.0]),
    ],
)
def test_check_weights_rejects_invalid(series):
    with pytest.raises(InvalidWeightsError):
        check_weights(series)


def test_estimate_unadjusted_effects_rejects_non_binary_treatment():
    ds = pl.DataFrame({"y": [1.0, 2.0], "d": [0, 2]})
    with pytest.raises(BinaryTreatmentError):
        estimate_unadjusted_effects(ds, "y", "d", bootstrap_config=1)


def test_estimate_unadjusted_effects_rejects_invalid_quantiles():
    ds = pl.DataFrame({"y": [1.0, 2.0], "d": [0, 1]})
    with pytest.raises(InvalidQuantilesError):
        estimate_unadjusted_effects(ds, "y", "d", qs=(0.0, 0.5), bootstrap_config=1)


def test_estimate_unadjusted_effects_rejects_invalid_weights():
    ds = pl.DataFrame({"y": [1.0, 2.0], "d": [0, 1], "w": [1.0, -1.0]})
    with pytest.raises(InvalidWeightsError):
        estimate_unadjusted_effects(ds, "y", "d", weights="w", bootstrap_config=1)


def test_estimate_unadjusted_effects_rejects_non_numeric_outcome():
    ds = pl.DataFrame({"y": ["a", "b"], "d": [0, 1]})
    with pytest.raises(ValueError):
        estimate_unadjusted_effects(ds, "y", "d", bootstrap_config=1)


def test_estimate_ipw_effects_rejects_invalid_target():
    ds = pl.DataFrame({"y": [1.0, 2.0], "d": [0, 1]})
    with pytest.raises(ValueError):
        estimate_ipw_effects(
            ds, "y", "d", propensity_score_formula="1", target="nope", bootstrap_config=1
        )


def test_outcome_regression_config_rejects_invalid_grid():
    with pytest.raises(ValueError):
        OutcomeRegressionConfig("1", grid=1)


def test_estimate_ipw_effects_rejects_missing_formula_column():
    ds = pl.DataFrame({"y": [1.0, 2.0], "d": [0, 1]})
    with pytest.raises(MissingColumnsError):
        estimate_ipw_effects(ds, "y", "d", propensity_score_formula="age", bootstrap_config=1)


def test_estimate_outcome_regression_effects_rejects_missing_formula_column():
    ds = pl.DataFrame({"y": [1.0, 2.0], "d": [0, 1]})
    with pytest.raises(MissingColumnsError):
        estimate_outcome_regression_effects(
            ds, "y", "d", outcome_regression_config="age", bootstrap_config=1
        )


def test_estimate_aipw_effects_rejects_missing_propensity_formula_column():
    ds = pl.DataFrame({"y": [1.0, 2.0], "d": [0, 1]})
    with pytest.raises(MissingColumnsError):
        estimate_aipw_effects(
            ds,
            "y",
            "d",
            propensity_score_formula="age",
            outcome_regression_config="1",
            bootstrap_config=1,
        )


def test_estimate_aipw_effects_rejects_missing_outcome_formula_column():
    ds = pl.DataFrame({"y": [1.0, 2.0], "d": [0, 1]})
    with pytest.raises(MissingColumnsError):
        estimate_aipw_effects(
            ds,
            "y",
            "d",
            propensity_score_formula="1",
            outcome_regression_config="age",
            bootstrap_config=1,
        )
