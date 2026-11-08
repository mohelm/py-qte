import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal
from pytest import approx, fixture, mark, raises

from qte._validation import UnbalancedPanelError
from qte.distributional_did.custom_types import (
    BasePeriod,
    ControlGroup,
    TrtGroupConfig,
)
from qte.distributional_did.estimator import (
    estimate_changes_in_changes_effects,
    estimate_distributional_did_effects,
    estimate_mean_did_effects,
    estimate_quantile_did_effects,
)


@fixture(scope="session")
def mpdata_prepared(mpdata):
    return mpdata.with_columns(
        pl.col("first.treat").replace(0, float("inf")).alias("first.treat"),
        pl.col("year").cast(pl.Float64).alias("year"),
    )


TEST_CASES_CHANGES_IN_CHANGES = [
    (
        {
            "base_period": BasePeriod.VARYING,
            "control_group": ControlGroup.NEVER_TREATED,
        },
        {
            "overall": {"group": None, "res": pl.Series("effect", [-0.01924603])},
            "treatment_group": {
                "group": ("treatment_group",),
                "res": pl.DataFrame(
                    {
                        "treatment_group": [2004, 2006, 2007],
                        "effect": [-0.0678029241, 0.0008487484, -0.0179685726],
                    }
                ),
            },
            "event_study": {
                "group": ("event_study_period",),
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
            "group_time_effects": {
                "group": ("group", "time"),
                "res": pl.DataFrame(
                    {
                        "group": [
                            2004,
                            2004,
                            2004,
                            2004,
                            2006,
                            2006,
                            2006,
                            2006,
                            2007,
                            2007,
                            2007,
                            2007,
                        ],
                        "time": [
                            2004,
                            2007,
                            2006,
                            2005,
                            2005,
                            2004,
                            2007,
                            2006,
                            2006,
                            2005,
                            2004,
                            2007,
                        ],
                        "effect": [
                            0.003064769,
                            -0.092958251,
                            -0.125003869,
                            -0.056314345,
                            0.015993557,
                            0.015244073,
                            -0.021859246,
                            0.023556742,
                            -0.019687426,
                            0.025716116,
                            0.059032449,
                            -0.017968573,
                        ],
                    }
                ).sort(["group", "time"]),
            },
        },
    ),
    (
        {
            "base_period": BasePeriod.VARYING,
            "control_group": ControlGroup.NOT_YET_TREATED,
        },
        {
            "overall": {"group": None, "res": pl.Series("effect", [-0.01966596])},
            "treatment_group": {
                "group": ("treatment_group",),
                "res": pl.DataFrame(
                    {
                        "treatment_group": [2004, 2006, 2007],
                        "effect": [-0.0718628885, 0.0008735407, -0.0179685726],
                    }
                ),
            },
            "event_study": {
                "group": ("event_study_period",),
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
            "group_time_effects": {
                "group": ("group", "time"),
                "res": pl.DataFrame(
                    {
                        "group": [
                            2004,
                            2004,
                            2004,
                            2004,
                            2006,
                            2006,
                            2006,
                            2006,
                            2007,
                            2007,
                            2007,
                            2007,
                        ],
                        "time": [
                            2004,
                            2007,
                            2006,
                            2005,
                            2005,
                            2004,
                            2007,
                            2006,
                            2006,
                            2005,
                            2004,
                            2007,
                        ],
                        "effect": [
                            -0.006450563,
                            -0.092958251,
                            -0.122572855,
                            -0.065469885,
                            0.009682760,
                            0.006067035,
                            -0.021859246,
                            0.023606327,
                            -0.019687426,
                            0.018834693,
                            0.050846598,
                            -0.017968573,
                        ],
                    }
                ).sort(["group", "time"]),
            },
        },
    ),
]


def _assert_att_aggregations(res, expected_results) -> None:
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

    agg_by_group_time_expected_results = expected_results["group_time_effects"]
    assert res.group_time.group == agg_by_group_time_expected_results["group"]
    assert_frame_equal(
        res.group_time.att.select("group", "time", "effect"),
        agg_by_group_time_expected_results["res"],
        check_dtypes=False,
    )


@mark.parametrize("estimate_params,expected_results", TEST_CASES_CHANGES_IN_CHANGES)
def test_estimate_changes_in_changes_for_panel_data_with_unconditional_parallel_trends(
    mpdata_prepared, estimate_params, expected_results
):
    qs = [0.25, 0.5, 0.75]
    res = estimate_changes_in_changes_effects(
        mpdata_prepared,
        "lemp",
        "first.treat",
        "year",
        "countyreal",
        qs=[0.25, 0.5, 0.75],
        **estimate_params,
        bootstrap_config=3,
    )

    _assert_att_aggregations(res, expected_results)

    # QTE
    assert_series_equal(res.overall.qtt["q"], pl.Series("q", qs))
    assert_series_equal(res.group.qtt["q"].unique(), pl.Series("q", qs))
    assert_series_equal(res.event_study.qtt["q"].unique(), pl.Series("q", qs))


TEST_CASES_QUANTILE_DID = [
    (
        {
            "base_period": BasePeriod.VARYING,
            "control_group": ControlGroup.NEVER_TREATED,
        },
        {
            "overall": {"group": None, "res": pl.Series("effect", [-0.02707699])},
            "treatment_group": {
                "group": ("treatment_group",),
                "res": pl.DataFrame(
                    {
                        "treatment_group": [2004, 2006, 2007],
                        "effect": [-0.08154130, -0.02274363, -0.02008499],
                    }
                ),
            },
            "event_study": {
                "group": ("event_study_period",),
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
            "group_time_effects": {
                "group": ("group", "time"),
                "res": pl.DataFrame(
                    {
                        "group": [
                            2004,
                            2004,
                            2004,
                            2004,
                            2006,
                            2006,
                            2006,
                            2006,
                            2007,
                            2007,
                            2007,
                            2007,
                        ],
                        "time": [
                            2004,
                            2007,
                            2006,
                            2005,
                            2005,
                            2004,
                            2007,
                            2006,
                            2006,
                            2005,
                            2004,
                            2007,
                        ],
                        "effect": [
                            -0.0129295649,
                            -0.1111424035,
                            -0.1342076532,
                            -0.0678855722,
                            0.0056240926,
                            -0.0047225781,
                            -0.0445423888,
                            -0.0009448729,
                            -0.0334230867,
                            0.0010063462,
                            0.0231114618,
                            -0.0200849923,
                        ],
                    }
                ).sort(["group", "time"]),
            },
        },
    ),
    (
        {
            "base_period": BasePeriod.VARYING,
            "control_group": ControlGroup.NOT_YET_TREATED,
        },
        {
            "overall": {"group": None, "res": pl.Series("effect", [-0.02821676])},
            "treatment_group": {
                "group": ("treatment_group",),
                "res": pl.DataFrame(
                    {
                        "treatment_group": [2004, 2006, 2007],
                        "effect": [-0.09136508, -0.02327415, -0.02008499],
                    }
                ),
            },
            "event_study": {
                "group": ("event_study_period",),
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
            "group_time_effects": {
                "group": ("group", "time"),
                "res": pl.DataFrame(
                    {
                        "group": [
                            2004,
                            2004,
                            2004,
                            2004,
                            2006,
                            2006,
                            2006,
                            2006,
                            2007,
                            2007,
                            2007,
                            2007,
                        ],
                        "time": [
                            2004,
                            2007,
                            2006,
                            2005,
                            2005,
                            2004,
                            2007,
                            2006,
                            2006,
                            2005,
                            2004,
                            2007,
                        ],
                        "effect": [
                            -0.021296492,
                            -0.111142403,
                            -0.147005585,
                            -0.086015842,
                            0.004058053,
                            -0.008741508,
                            -0.044542389,
                            -0.002005911,
                            -0.033423087,
                            -0.002559103,
                            0.026582485,
                            -0.020084992,
                        ],
                    }
                ).sort(["group", "time"]),
            },
        },
    ),
]


@mark.parametrize("estimate_params,expected_results", TEST_CASES_QUANTILE_DID)
def test_estimate_quantile_did_for_panel_data_with_unconditional_parallel_trends(
    mpdata_prepared, estimate_params, expected_results
):
    qs = [0.25, 0.5, 0.75]
    res = estimate_quantile_did_effects(
        mpdata_prepared,
        "lemp",
        "first.treat",
        "year",
        "countyreal",
        qs=[0.25, 0.5, 0.75],
        **estimate_params,
        bootstrap_config=3,
    )

    _assert_att_aggregations(res, expected_results)

    # QTE
    assert_series_equal(res.overall.qtt["q"], pl.Series("q", qs))
    assert_series_equal(res.group.qtt["q"].unique(), pl.Series("q", qs))
    assert_series_equal(res.event_study.qtt["q"].unique(), pl.Series("q", qs))


def test_trt_group_config_on_raw_data_keeps_integer_identifiers(mpdata, mpdata_prepared):
    qs = [0.25, 0.5, 0.75]

    raw = estimate_changes_in_changes_effects(
        mpdata,
        "lemp",
        TrtGroupConfig("first.treat", 0),
        "year",
        "countyreal",
        qs=qs,
        bootstrap_config=3,
    )
    prepared = estimate_changes_in_changes_effects(
        mpdata_prepared,
        "lemp",
        "first.treat",
        "year",
        "countyreal",
        qs=qs,
        bootstrap_config=3,
    )

    # identifiers come back as integers even though the internals are float
    assert raw.group.att.schema["treatment_group"] == pl.Int64
    assert raw.event_study.att.schema["event_study_period"] == pl.Int64

    # the internal 0 -> inf mapping must reproduce the pre-mapped result exactly
    assert_frame_equal(
        raw.group.att.select("treatment_group", "effect"),
        prepared.group.att.select("treatment_group", "effect"),
    )
    assert_frame_equal(
        raw.event_study.att.select("event_study_period", "effect"),
        prepared.event_study.att.select("event_study_period", "effect"),
    )


GROUPS_AND_TIMES_VARYING = {
    "group": [g for g in (2004, 2006, 2007) for _ in range(4)],
    "time": [t for _ in (2004, 2006, 2007) for t in (2004, 2005, 2006, 2007)],
}

# Reference values from qte::mdid() (R) on mpdta, with base_period = "varying".
TEST_CASES_MEAN_DID = [
    (
        {
            "base_period": BasePeriod.VARYING,
            "control_group": ControlGroup.NEVER_TREATED,
        },
        {
            "overall": {"group": None, "res": pl.Series("effect", [-0.031018282])},
            "treatment_group": {
                "group": ("treatment_group",),
                "res": pl.DataFrame(
                    {
                        "treatment_group": [2004, 2006, 2007],
                        "effect": [-0.079749127, -0.022909539, -0.026054411],
                    }
                ),
            },
            "event_study": {
                "group": ("event_study_period",),
                "res": pl.DataFrame(
                    {
                        "event_study_period": [-3, -2, -1, 0, 1, 2, 3],
                        "effect": [
                            0.030506656,
                            -0.000563085,
                            -0.024458745,
                            -0.019931817,
                            -0.050957367,
                            -0.137258739,
                            -0.100811363,
                        ],
                    }
                ),
            },
            "group_time_effects": {
                "group": ("group", "time"),
                "res": pl.DataFrame(
                    {
                        **GROUPS_AND_TIMES_VARYING,
                        "effect": [
                            -0.010503246,
                            -0.070423158,
                            -0.137258739,
                            -0.100811363,
                            0.006520112,
                            -0.002750819,
                            -0.004594607,
                            -0.041224472,
                            0.030506656,
                            -0.002725893,
                            -0.031087119,
                            -0.026054411,
                        ],
                    }
                ).sort(["group", "time"]),
            },
        },
    ),
    (
        {
            "base_period": BasePeriod.VARYING,
            "control_group": ControlGroup.NOT_YET_TREATED,
        },
        {
            "overall": {"group": None, "res": pl.Series("effect", [-0.030462228])},
            "treatment_group": {
                "group": ("treatment_group",),
                "res": pl.DataFrame(
                    {
                        "treatment_group": [2004, 2006, 2007],
                        "effect": [-0.083694293, -0.018281798, -0.026054411],
                    }
                ),
            },
            "event_study": {
                "group": ("event_study_period",),
                "res": pl.DataFrame(
                    {
                        "event_study_period": [-3, -2, -1, 0, 1, 2, 3],
                        "effect": [
                            0.029759365,
                            -0.002446154,
                            -0.024268903,
                            -0.018922199,
                            -0.053589347,
                            -0.136274346,
                            -0.100811363,
                        ],
                    }
                ),
            },
            "group_time_effects": {
                "group": ("group", "time"),
                "res": pl.DataFrame(
                    {
                        **GROUPS_AND_TIMES_VARYING,
                        "effect": [
                            -0.019372364,
                            -0.078319099,
                            -0.136274346,
                            -0.100811363,
                            -0.002562551,
                            -0.001939246,
                            0.004660876,
                            -0.041224472,
                            0.029759365,
                            -0.002410613,
                            -0.031087119,
                            -0.026054411,
                        ],
                    }
                ).sort(["group", "time"]),
            },
        },
    ),
]

# Reference values from qte::ddid() (R) on mpdta, with base_period = "varying".
TEST_CASES_DISTRIBUTIONAL_DID = [
    (
        {
            "base_period": BasePeriod.VARYING,
            "control_group": ControlGroup.NEVER_TREATED,
        },
        {
            "overall": {"group": None, "res": pl.Series("effect", [-0.047531111])},
            "treatment_group": {
                "group": ("treatment_group",),
                "res": pl.DataFrame(
                    {
                        "treatment_group": [2004, 2006, 2007],
                        "effect": [-0.087593078, -0.037135165, -0.044589115],
                    }
                ),
            },
            "event_study": {
                "group": ("event_study_period",),
                "res": pl.DataFrame(
                    {
                        "event_study_period": [-3, -2, -1, 0, 1, 2, 3],
                        "effect": [
                            0.014069153,
                            -0.015827781,
                            -0.042197054,
                            -0.036444646,
                            -0.063055768,
                            -0.145102691,
                            -0.108655315,
                        ],
                    }
                ),
            },
            "group_time_effects": {
                "group": ("group", "time"),
                "res": pl.DataFrame(
                    {
                        **GROUPS_AND_TIMES_VARYING,
                        "effect": [
                            -0.018347198,
                            -0.078267110,
                            -0.145102691,
                            -0.108655315,
                            -0.006286586,
                            -0.013031193,
                            -0.018820233,
                            -0.055450098,
                            0.014069153,
                            -0.018741124,
                            -0.051102661,
                            -0.044589115,
                        ],
                    }
                ).sort(["group", "time"]),
            },
        },
    ),
    (
        {
            "base_period": BasePeriod.VARYING,
            "control_group": ControlGroup.NOT_YET_TREATED,
        },
        {
            "overall": {"group": None, "res": pl.Series("effect", [-0.045276275])},
            "treatment_group": {
                "group": ("treatment_group",),
                "res": pl.DataFrame(
                    {
                        "treatment_group": [2004, 2006, 2007],
                        "effect": [-0.087009453, -0.026660135, -0.044589115],
                    }
                ),
            },
            "event_study": {
                "group": ("event_study_period",),
                "res": pl.DataFrame(
                    {
                        "event_study_period": [-3, -2, -1, 0, 1, 2, 3],
                        "effect": [
                            0.013369614,
                            -0.014576319,
                            -0.040343357,
                            -0.032371859,
                            -0.063733045,
                            -0.137731355,
                            -0.108655315,
                        ],
                    }
                ),
            },
            "group_time_effects": {
                "group": ("group", "time"),
                "res": pl.DataFrame(
                    {
                        **GROUPS_AND_TIMES_VARYING,
                        "effect": [
                            -0.021352204,
                            -0.080298939,
                            -0.137731355,
                            -0.108655315,
                            -0.005904732,
                            -0.005106639,
                            0.002129828,
                            -0.055450098,
                            0.013369614,
                            -0.017224132,
                            -0.051102661,
                            -0.044589115,
                        ],
                    }
                ).sort(["group", "time"]),
            },
        },
    ),
]


@mark.parametrize("estimate_params,expected_results", TEST_CASES_MEAN_DID)
def test_estimate_mean_did_for_panel_data(mpdata_prepared, estimate_params, expected_results):
    qs = [0.25, 0.5, 0.75]
    res = estimate_mean_did_effects(
        mpdata_prepared,
        "lemp",
        "first.treat",
        "year",
        "countyreal",
        qs=qs,
        **estimate_params,
        bootstrap_config=3,
    )

    _assert_att_aggregations(res, expected_results)

    qtt = res.group_time.qtt
    assert_series_equal(qtt["effect"], qtt["q_t"] - qtt["q_c"], check_names=False)


@mark.parametrize("estimate_params,expected_results", TEST_CASES_DISTRIBUTIONAL_DID)
def test_estimate_distributional_did_for_panel_data(
    mpdata_prepared, estimate_params, expected_results
):
    qs = [0.25, 0.5, 0.75]
    res = estimate_distributional_did_effects(
        mpdata_prepared,
        "lemp",
        "first.treat",
        "year",
        "countyreal",
        qs=qs,
        **estimate_params,
        bootstrap_config=3,
    )

    _assert_att_aggregations(res, expected_results)

    # Reference cell-level quantiles from the change-plus-rank-matched-level construction in R.
    reference = {
        (ControlGroup.NEVER_TREATED, 2004, 2004): (
            [5.003946306, 5.488937726, 6.859614904],
            [5.073441546, 5.843241538, 6.857931623],
        ),
        (ControlGroup.NEVER_TREATED, 2007, 2007): (
            [4.969813300, 5.713732806, 6.765038977],
            [4.996461241, 5.836805608, 6.721262137],
        ),
        (ControlGroup.NOT_YET_TREATED, 2004, 2004): (
            [5.003946306, 5.488937726, 6.859614904],
            [5.078964962, 5.743003188, 6.854868486],
        ),
        (ControlGroup.NOT_YET_TREATED, 2007, 2007): (
            [4.969813300, 5.713732806, 6.765038977],
            [4.996461241, 5.836805608, 6.721262137],
        ),
    }
    for (control_group, g, tp), (q_t, q_c) in reference.items():
        if estimate_params["control_group"] != control_group:
            continue
        cell = res.group_time.qtt.filter((pl.col("group") == g) & (pl.col("time") == tp)).sort("q")
        assert cell["q_t"].to_list() == approx(q_t)
        assert cell["q_c"].to_list() == approx(q_c)


def test_distributional_did_group_time_qtt_is_the_quantile_difference(mpdata_prepared):
    res = estimate_distributional_did_effects(
        mpdata_prepared,
        "lemp",
        "first.treat",
        "year",
        "countyreal",
        qs=[0.1, 0.25, 0.5, 0.75, 0.9],
        base_period=BasePeriod.VARYING,
        bootstrap_config=2,
    )
    qtt = res.group_time.qtt
    assert_series_equal(qtt["effect"], qtt["q_t"] - qtt["q_c"], check_names=False)


def test_distributional_did_requires_control_units_in_both_periods(mpdata, mpdata_prepared):
    never_treated_unit = mpdata.filter(pl.col("first.treat") == 0)["countyreal"].min()
    unbalanced = mpdata_prepared.filter(
        ~((pl.col("countyreal") == never_treated_unit) & (pl.col("year") == 2004.0))
    )
    with raises(UnbalancedPanelError, match="observed in both periods"):
        estimate_distributional_did_effects(
            unbalanced,
            "lemp",
            "first.treat",
            "year",
            "countyreal",
            control_group=ControlGroup.NEVER_TREATED,
            bootstrap_config=2,
        )
