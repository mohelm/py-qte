import polars as pl
import pytest
from polars.testing import assert_series_equal

from qte.datasets import load_card, load_jtpa
from qte.iv import Estimator, estimate_local_effects
from qte.names import EFFECT_ID, QUANTILE_ID


def test_card_dataset_estimation_runs():
    ds = load_card()
    res = estimate_local_effects(
        ds,
        "lwage",
        "college",
        "nearc4",
        qs=[0.25, 0.5, 0.75],
        bootstrap_config=10,
    )
    assert res.qtt[EFFECT_ID].is_finite().all()
    assert res.first_stage > 0


def test_jtpa_local_effects_runs():
    ds = load_jtpa().filter(pl.col("sex") == 0)
    covariates = (
        "age2225 + age2629 + age3035 + age3644 + age4554 + black + hispanic "
        "+ married + afdc + hsorged + prevearn"
    )
    res = estimate_local_effects(
        ds,
        "earnings",
        "training",
        "assignmt",
        qs=[0.25, 0.5, 0.75],
        instrument_probability_formula=covariates,
        bootstrap_config=20,
    )
    assert res.first_stage > 0.5
    assert res.qtt[EFFECT_ID].is_finite().all()


# Stata ``ivqte`` reference, Frölich & Melly (2007) unconditional endogenous QTE:
#   ivqte lwage (college = nearc4), variance dummy(black) continuous(age fatheduc
#   motheduc) unordered(region)
# see ivqte_examples.log.
_CARD_IVQTE_FORMULA = (
    "black + age + fatheduc + motheduc + reg662 + reg663 + reg664 + reg665 "
    "+ reg666 + reg667 + reg668 + reg669"
)
_CARD_IVQTE_DECILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# Stata ``ivqte ...`` (signed) and ``ivqte ..., positive`` (positive weights).
_CARD_IVQTE_CASES = [
    (
        Estimator.SIGNED,
        [0.7099, 0.7323, 0.7083, 0.6355, 0.6863, 0.7277, 0.5733, 0.6024, 0.6312],
    ),
    (
        Estimator.POSITIVE,
        [0.7589, 0.7195, 0.6464, 0.5864, 0.5358, 0.5104, 0.4785, 0.4618, 0.5175],
    ),
]


@pytest.mark.parametrize("estimator,expected", _CARD_IVQTE_CASES)
def test_estimate_local_effects_matches_ivqte_card(card, estimator, expected):
    res = estimate_local_effects(
        card,
        "lwage",
        "college",
        "nearc4",
        qs=_CARD_IVQTE_DECILES,
        instrument_probability_formula=_CARD_IVQTE_FORMULA,
        estimator=estimator,
        bootstrap_config=10,
    )

    assert_series_equal(pl.Series("q", _CARD_IVQTE_DECILES), res.qtt[QUANTILE_ID])
    assert_series_equal(pl.Series("effect", expected), res.qtt[EFFECT_ID], abs_tol=1e-3)
    assert res.first_stage == pytest.approx(0.103, abs=1e-3)
    assert res.complier_share == pytest.approx(0.083, abs=1e-3)
