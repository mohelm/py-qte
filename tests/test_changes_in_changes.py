import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal
from pytest import fixture, mark

from qte.non_linear_did.changes_in_changes import (
    estimate_changes_in_changes_for_panel,
)
from qte.non_linear_did.custom_types import BasePeriod, ControlGroup


@fixture(scope="session")
def mpdata_prepared(mpdata):
    return mpdata.with_columns(
        pl.col("first.treat").replace(0, float("inf")).alias("first.treat"),
        pl.col("year").cast(pl.Float64).alias("year"),
    )


TEST_CASES = [
    (
        {"base_period": BasePeriod.VARYING, "control_group": ControlGroup.NEVER_TREATED},
        {
            "overall": {"group": None, "res": pl.Series("effect", [-0.01924603])},
            "treatment_group": {
                "group": "treatment_group",
                "res": pl.DataFrame(
                    {
                        "treatment_group": [2004, 2006, 2007],
                        "effect": [-0.0678029241, 0.0008487484, -0.0179685726],
                    }
                ),
            },
            "event_study": {
                "group": "event_study_period",
                "res": pl.DataFrame(
                    {
                        "event_study_period": [-3, -2, -1, 0, 1, 2, 3],
                        "effect": [
                            0.059032449,
                            0.023266515,
                            -0.011340997,
                            -0.007069727,
                            -0.033344279,
                            -0.125003869,
                            -0.092958251,
                        ],
                    }
                ),
            },
        },
    ),
    (
        {"base_period": BasePeriod.VARYING, "control_group": ControlGroup.NOT_YET_TREATED},
        {
            "overall": {"group": None, "res": pl.Series("effect", [-0.01966596])},
            "treatment_group": {
                "group": "treatment_group",
                "res": pl.DataFrame(
                    {
                        "treatment_group": [2004, 2006, 2007],
                        "effect": [-0.0718628885, 0.0008735407, -0.0179685726],
                    }
                ),
            },
            "event_study": {
                "group": "event_study_period",
                "res": pl.DataFrame(
                    {
                        "event_study_period": [-3, -2, -1, 0, 1, 2, 3],
                        "effect": [
                            0.050846598,
                            0.015848106,
                            -0.012817207,
                            -0.008055713,
                            -0.036396125,
                            -0.122572855,
                            -0.092958251,
                        ],
                    }
                ),
            },
        },
    ),
]


@mark.parametrize("estimate_params,expected_results", TEST_CASES)
def test_estimate_changes_in_changes_for_panel(mpdata_prepared, estimate_params, expected_results):
    qs = [0.25, 0.5, 0.75]
    res = estimate_changes_in_changes_for_panel(
        mpdata_prepared,
        "lemp",
        "first.treat",
        "year",
        "countyreal",
        qs=[0.25, 0.5, 0.75],
        **estimate_params,
        n_bootstrap_iter=3,
    )

    # ATT
    overall_expected_results = expected_results["overall"]
    assert res.overall.group is overall_expected_results["group"]
    assert_series_equal(res.overall.att["effect"], overall_expected_results["res"])

    agg_by_treatment_group_expected_results = expected_results["treatment_group"]
    assert res.group.group == agg_by_treatment_group_expected_results["group"]
    assert_frame_equal(
        res.group.att.select("treatment_group", "effect"),
        agg_by_treatment_group_expected_results["res"],
        check_dtypes=False,
    )

    agg_by_event_study_period_expected_results = expected_results["event_study"]
    assert res.event_study.group == agg_by_event_study_period_expected_results["group"]
    assert_frame_equal(
        res.event_study.att.select("event_study_period", "effect"),
        agg_by_event_study_period_expected_results["res"],
        check_dtypes=False,
    )

    # QTE
    assert_series_equal(res.overall.qtt["q"], pl.Series("q", qs))
    assert_series_equal(res.group.qtt["q"].unique(), pl.Series("q", qs))
    assert_series_equal(res.event_study.qtt["q"].unique(), pl.Series("q", qs))
