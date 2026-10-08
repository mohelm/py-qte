import polars as pl

from qte.datasets import load_card, load_jtpa
from qte.iv import estimate_local_effects
from qte.names import EFFECT_ID


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
