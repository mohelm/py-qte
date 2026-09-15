import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal
from pytest import fixture, mark

from qte.non_linear_did.changes_in_changes import (
    estimate_changes_in_changes_for_panel,
)
from qte.non_linear_did.custom_types import BasePeriod, ControlGroup, CounterfactualModel


@fixture(scope="session")
def mpdata_prepared(mpdata):
    return mpdata.with_columns(
        pl.col("first.treat").replace(0, float("inf")).alias("first.treat"),
        pl.col("year").cast(pl.Float64).alias("year"),
    )


TEST_CASES_CIC = [
    (
        {
            "base_period": BasePeriod.VARYING,
            "control_group": ControlGroup.NEVER_TREATED,
            "counterfactual_model": CounterfactualModel.CIC,
        },
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
        {
            "base_period": BasePeriod.VARYING,
            "control_group": ControlGroup.NOT_YET_TREATED,
            "counterfactual_model": CounterfactualModel.CIC,
        },
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


@mark.parametrize("estimate_params,expected_results", TEST_CASES_CIC)
def test_estimate_cic_for_panel_data_with_unconditional_parallel_trends(
    mpdata_prepared, estimate_params, expected_results
):
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


TEST_CASES_QDID = [
    (
        {
            "base_period": BasePeriod.VARYING,
            "control_group": ControlGroup.NEVER_TREATED,
            "counterfactual_model": CounterfactualModel.QDID,
        },
        {
            "overall": {"group": None, "res": pl.Series("effect", [-0.02707699])},
            "treatment_group": {
                "group": "treatment_group",
                "res": pl.DataFrame(
                    {
                        "treatment_group": [2004, 2006, 2007],
                        "effect": [-0.08154130, -0.02274363, -0.02008499],
                    }
                ),
            },
            "event_study": {
                "group": "event_study_period",
                "res": pl.DataFrame(
                    {
                        "event_study_period": [-3, -2, -1, 0, 1, 2, 3],
                        "effect": [
                            0.023111462,
                            -0.000333753,
                            -0.024289244,
                            -0.015327331,
                            -0.052323450,
                            -0.134207653,
                            -0.111142403,
                        ],
                    }
                ),
            },
        },
    ),
    (
        {
            "base_period": BasePeriod.VARYING,
            "control_group": ControlGroup.NOT_YET_TREATED,
            "counterfactual_model": CounterfactualModel.QDID,
        },
        {
            "overall": {"group": None, "res": pl.Series("effect", [-0.02821676])},
            "treatment_group": {
                "group": "treatment_group",
                "res": pl.DataFrame(
                    {
                        "treatment_group": [2004, 2006, 2007],
                        "effect": [-0.09136508, -0.02327415, -0.02008499],
                    }
                ),
            },
            "event_study": {
                "group": "event_study_period",
                "res": pl.DataFrame(
                    {
                        "event_study_period": [-3, -2, -1, 0, 1, 2, 3],
                        "effect": [
                            0.026582485,
                            -0.004005279,
                            -0.024655569,
                            -0.016425656,
                            -0.058366873,
                            -0.147005585,
                            -0.111142403,
                        ],
                    }
                ),
            },
        },
    ),
]


@mark.parametrize("estimate_params,expected_results", TEST_CASES_QDID)
def test_estimate_qdid_for_panel_data_with_unconditional_parallel_trends(
    mpdata_prepared, estimate_params, expected_results
):
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
