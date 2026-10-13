"""Tests for the conditional (Abadie-Angrist-Imbens) local QTE estimator."""

import polars as pl
import pytest
from polars.testing import assert_series_equal

from qte.datasets import load_jtpa
from qte.iv import WeightAdjustment, estimate_conditional_local_effects
from qte.names import EFFECT_ID, QUANTILE_CONTROL_VAL_ID, QUANTILE_ID, QUANTILE_TREATED_VAL_ID

# Covariates of Abadie, Angrist & Imbens (2002), women subsample.
_AAI_WOMEN_COVARIATES = (
    "black + hispanic + married + hsorged + afdc + f2sms + wkless13 + class_tr "
    "+ ojt_jsa + age2225 + age2629 + age3035 + age3644 + age4554"
)
_AAI_QUANTILES = [0.15, 0.25, 0.50, 0.75, 0.85]


@pytest.mark.parametrize("weight_adjustment", list(WeightAdjustment))
def test_estimate_conditional_local_effects_runs(weight_adjustment):
    ds = load_jtpa().filter(pl.col("sex") == 0)
    res = estimate_conditional_local_effects(
        ds,
        "earnings",
        "training",
        "assignmt",
        qs=_AAI_QUANTILES,
        covariates=_AAI_WOMEN_COVARIATES,
        instrument_probability_formula=_AAI_WOMEN_COVARIATES,
        weight_adjustment=weight_adjustment,
        bootstrap_config=5,
    )

    assert_series_equal(pl.Series("q", _AAI_QUANTILES), res.qtt[QUANTILE_ID])
    assert res.qtt[EFFECT_ID].is_finite().all()
    assert res.qtt["se"].is_finite().all()
    assert 0.4 < res.first_stage < 0.8
    # The average training effect for women is positive and plausible.
    assert 1000 < res.att[EFFECT_ID][0] < 3000


# Stata ``ivqte_examples.do`` CASE 2 (Card): ``ivqte lwage (college=nearc4),
# q(0.1) ... aai``. The do-file leaves the positive-weight bandwidth at its
# default ``h = infinity``, so the local linear projection degenerates to the
# global linear projection our PROJECTION adjustment implements. Stata reports
# college = 0.6130352, Constant = 3.229647 and a complier share of 0.089.
_CARD_IVQTE_FORMULA = (
    "black + age + fatheduc + motheduc + reg662 + reg663 + reg664 + reg665 "
    "+ reg666 + reg667 + reg668 + reg669"
)
_CARD_AAI_CONDITIONAL = 0.6130352
_CARD_AAI_INTERCEPT = 3.229647


def test_card_aai_conditional_matches_stata(card):
    res = estimate_conditional_local_effects(
        card,
        "lwage",
        "college",
        "nearc4",
        qs=[0.1],
        covariates=_CARD_IVQTE_FORMULA,
        instrument_probability_formula=_CARD_IVQTE_FORMULA,
        weight_adjustment=WeightAdjustment.PROJECTION,
        bootstrap_config=1,
    )

    assert res.qtt[EFFECT_ID][0] == pytest.approx(_CARD_AAI_CONDITIONAL, abs=1e-5)
    assert res.qtt[QUANTILE_CONTROL_VAL_ID][0] == pytest.approx(_CARD_AAI_INTERCEPT, abs=1e-5)
    assert res.qtt[QUANTILE_TREATED_VAL_ID][0] == pytest.approx(
        _CARD_AAI_INTERCEPT + _CARD_AAI_CONDITIONAL, abs=1e-5
    )
    assert res.complier_share == pytest.approx(0.089, abs=1e-3)


# Stata ``ivqte_examples.do`` JTPA replication of AAI (2002) Table III
# (Froelich & Melly 2010, p. 444): constant instrument propensity
# ``pi = mean(assignment)``, weights from a degree-5 power series in earnings
# interacted with treatment (``reg kappa earnings treatment e2..e5 de1..de5``),
# negative weights dropped, then a weighted median regression with the men's
# covariates. Stata reports a training coefficient of 1544.059 and an
# intercept (Constant) of 7688.78.
_JTPA_MEN_COVARIATES = (
    "hsorged + black + hispanic + married + wkless13 + class_tr + ojt_jsa "
    "+ age2225 + age2629 + age3035 + age3644 + age4554 + f2sms"
)
_JTPA_AAI_SERIES_TREATMENT = 1544.059
_JTPA_AAI_INTERCEPT = 7688.78


def test_jtpa_aai_series_matches_stata():
    ds = load_jtpa().filter(pl.col("sex") == 1)
    res = estimate_conditional_local_effects(
        ds,
        "earnings",
        "training",
        "assignmt",
        qs=[0.5],
        covariates=_JTPA_MEN_COVARIATES,
        weight_formula="poly(earnings, 5)",  # the do-file's `reg kappa ...`, per arm
        instrument_probability_formula=None,  # constant pi = mean(assignment)
        weight_adjustment=WeightAdjustment.PLUGIN,
        bootstrap_config=1,
    )

    assert res.qtt[EFFECT_ID][0] == pytest.approx(_JTPA_AAI_SERIES_TREATMENT, abs=0.01)
    assert res.qtt[QUANTILE_CONTROL_VAL_ID][0] == pytest.approx(_JTPA_AAI_INTERCEPT, abs=0.01)
    assert res.qtt[QUANTILE_TREATED_VAL_ID][0] == pytest.approx(
        _JTPA_AAI_INTERCEPT + _JTPA_AAI_SERIES_TREATMENT, abs=0.01
    )


def test_conditional_effects_are_close_to_aai_table_iii():
    # AAI (2002) Table III, women: the plug-in route should be in the right
    # neighbourhood (the exact numbers need their propensity cell means).
    expected = [324, 680, 1742, 1984, 1900]
    ds = load_jtpa().filter(pl.col("sex") == 0)
    res = estimate_conditional_local_effects(
        ds,
        "earnings",
        "training",
        "assignmt",
        qs=_AAI_QUANTILES,
        covariates=_AAI_WOMEN_COVARIATES,
        instrument_probability_formula=_AAI_WOMEN_COVARIATES,
        weight_adjustment=WeightAdjustment.PLUGIN,
        bootstrap_config=5,
    )
    effects = res.qtt[EFFECT_ID].to_numpy()
    assert (abs(effects - expected) < 800).all()
