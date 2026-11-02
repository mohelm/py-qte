"""Replication of the empirical example in Chen (2018) and Chernozhukov & Hansen (2004)."""

import numpy as np
import polars as pl
import pytest
from great_tables import GT
from polars.testing import assert_frame_equal, assert_series_equal
from rich.console import Group

from qte.datasets import load_pension
from qte.iv import IvqrTreatment, estimate_ivqr_effects
from qte.names import EFFECT_ID, ESTIMATE_ID, QUANTILE_ID, SE_ID

# Specification of the 401(k) example in Chen (2018): net financial assets on
# 401(k) participation, instrumented by 401(k) eligibility, with income and
# education category dummies plus the demographic and pension controls.
_PENSION_COVARIATES = (
    "C(icat) + C(ecat) + a1 + a2 + a3 + a4 + marr + fsize + twoearn + db + pira + hown"
)

# ``summary(fit_median)`` in Chen (2018), table of endogenous coefficients:
# coefficient 5700, standard error 619.0487. The paper's control coefficients
# are reproduced too but depend on the quantile-regression tie-break; the
# endogenous coefficient is the object of interest.
_CHEN_MEDIAN_EFFECT = 5700.0
_CHEN_MEDIAN_SE = 619.0487


def test_estimate_ivqr_effects_matches_chen_pension_example():
    res = estimate_ivqr_effects(
        load_pension(),
        "net_tfa",
        treatment=IvqrTreatment("p401", np.arange(0, 25_001, 100)),
        instruments="e401",
        qs=[0.5],
        covariates=_PENSION_COVARIATES,
    )

    # First: look at the quantile treatment effect
    effect = res.qtt.filter(pl.col(QUANTILE_ID) == 0.5)
    assert effect[EFFECT_ID][0] == pytest.approx(_CHEN_MEDIAN_EFFECT, abs=1.0)
    assert effect[SE_ID][0] == pytest.approx(_CHEN_MEDIAN_SE, abs=0.1)

    # Second: in that setup the mean effect with a single quantile is identical to the median
    # effect since it is an average over the quantiles.
    assert res.att[EFFECT_ID][0] == pytest.approx(_CHEN_MEDIAN_EFFECT, abs=1.0)
    assert res.att[SE_ID][0] == pytest.approx(res.qtt[SE_ID][0], rel=1e-10)

    # Third: Check that our two-stage least square number does not change.
    assert res.two_sls[0] == pytest.approx(13_086.849, abs=1.0)
    assert res.two_sls[1] > 0
    assert res.covariate_coefficients[ESTIMATE_ID].is_finite().all()
    assert res.covariate_coefficients[SE_ID].is_finite().all()
    assert res.n_obs == load_pension().height
    assert isinstance(res.get_as_dataframe(), pl.DataFrame)
    assert isinstance(res.tabulate(), GT)
    assert isinstance(res.summarize(), Group)


def test_estimate_ivqr_effects_matches_r_pension_quartiles():
    res = estimate_ivqr_effects(
        load_pension(),
        "net_tfa",
        treatment=IvqrTreatment("p401", np.arange(0, 25_001, 100)),
        instruments="e401",
        qs=[0.25, 0.75],
        covariates=_PENSION_COVARIATES,
    )

    expected_effects = pl.Series(EFFECT_ID, [3_700.0, 13_200.0])
    expected_standard_errors = pl.Series(SE_ID, [528.619703298070, 1_209.065740145327])
    assert_series_equal(res.qtt[EFFECT_ID], expected_effects, abs_tol=1.0)
    assert_series_equal(res.qtt[SE_ID], expected_standard_errors, abs_tol=0.1)


def test_estimate_ivqr_effects_allows_overidentification():
    """A single endogenous variable may be instrumented by several instruments."""
    res = estimate_ivqr_effects(
        load_pension(),
        "net_tfa",
        treatment=IvqrTreatment("p401", np.arange(5_000, 6_501, 100)),
        instruments=["e401", "age"],
        qs=[0.45, 0.5, 0.55],
        covariates=_PENSION_COVARIATES,
    )

    effect = res.qtt.filter(pl.col(QUANTILE_ID) == 0.5)
    assert effect[EFFECT_ID][0] == pytest.approx(_CHEN_MEDIAN_EFFECT, abs=1.0)
    assert np.isfinite(effect[SE_ID][0])
    assert effect[SE_ID][0] > 0
    assert res.att[EFFECT_ID][0] == pytest.approx(res.qtt[EFFECT_ID].mean())
    assert np.isfinite(res.att[SE_ID][0])
    assert res.att[SE_ID][0] > 0
    assert np.isfinite(res.two_sls[0])
    assert np.isfinite(res.two_sls[1])


def test_ivqr_parallel_grid_matches_serial():
    kwargs = {
        "ds": load_pension(),
        "outcome": "net_tfa",
        "treatment": IvqrTreatment("p401", np.array([5_000.0, 5_700.0, 6_500.0])),
        "instruments": "e401",
        "qs": [0.25, 0.5, 0.75],
        "covariates": _PENSION_COVARIATES,
    }
    serial = estimate_ivqr_effects(**kwargs, n_workers=1)
    parallel = estimate_ivqr_effects(**kwargs, n_workers=2)

    assert_frame_equal(serial.qtt, parallel.qtt)
    assert_frame_equal(serial.att, parallel.att)
    assert serial.two_sls == pytest.approx(parallel.two_sls)
