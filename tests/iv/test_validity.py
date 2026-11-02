"""Tests for the Kitagawa (2015) instrument-validity test."""

import numpy as np
import polars as pl
import pytest
from great_tables import GT
from rich.console import Group

from qte.bootstrap import BootstrapConfig
from qte.datasets import load_card
from qte.iv import InstrumentValidityResult, ValidityWeighting
from qte.iv.simulate import simulate_data
from qte.iv.validity import perform_instrument_validity_test as run_validity_test

TOY = pl.DataFrame(
    {
        "y": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.5, 3.0, 4.0, 1.5, 3.0, 4.0],
        "d": [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1],
        "z": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        "w": [1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 1.5, 1.0, 1.0, 1.0, 1.0, 2.0],
    }
)


def _compute_statistic_by_hand(
    ds, weights=None, weighting=ValidityWeighting.VARIANCE, se_floor=0.15
):
    """Evaluate Kitagawa (2015, equations 2.1/2.2) interval by interval, by hand."""
    y = ds["y"].to_numpy()
    d = ds["d"].to_numpy()
    z = ds["z"].to_numpy()
    w = ds[weights].to_numpy() if weights is not None else np.ones(y.size)
    w = w * y.size / w.sum()

    levels = np.unique(z)
    take_up = [np.average(d[z == level], weights=w[z == level]) for level in levels]
    low, high = (levels[i] for i in np.argsort(take_up))
    pool = w / w.sum()

    ends = [float("-inf"), *np.unique(y)]

    def mass(is_high, treatment_value, lower, upper):
        in_arm = z == high if is_high else z == low
        selected = in_arm & (d == treatment_value) & (y > lower) & (y <= upper)
        return np.sum(w[selected]) / np.sum(w[in_arm])

    n_low = np.sum(w[z == low])
    n_high = np.sum(w[z == high])
    total = n_low + n_high
    pair_scale = np.sqrt(n_low * n_high / total)
    best = 0.0
    for treatment_value, direction in ((1, 1.0), (0, -1.0)):
        for i, lower in enumerate(ends):
            for upper in ends[i + 1 :]:
                p_low = mass(False, treatment_value, lower, upper)
                p_high = mass(True, treatment_value, lower, upper)
                if weighting is ValidityWeighting.VARIANCE:
                    variance = (1 - n_high / total) * p_high * (1 - p_high)
                    variance += (n_high / total) * p_low * (1 - p_low)
                    scale = max(np.sqrt(variance), se_floor)
                else:
                    scale = 1.0
                best = max(best, pair_scale * direction * (p_low - p_high) / scale)
    return best, pool, (low, high)


def test_statistic_and_binding_match_the_paper_formula_on_toy_data():
    res = run_validity_test(TOY, "y", "d", "z", bootstrap_config=1)

    expected, _, _ = _compute_statistic_by_hand(TOY)
    assert res.statistic == pytest.approx(expected, abs=1e-10)
    assert res.statistic == pytest.approx(1.7320508, abs=1e-6)
    assert res.binding is not None
    assert res.binding["direction"] == "d = 0"
    assert res.binding["y_lower"] == 1.0
    assert res.binding["y_upper"] == 1.5


def test_weighted_and_unweighted_statistics_match_the_paper_formula():
    weighted = run_validity_test(TOY, "y", "d", "z", weights="w", bootstrap_config=1)
    expected, _, _ = _compute_statistic_by_hand(TOY, weights="w")
    assert weighted.statistic == pytest.approx(expected, abs=1e-10)

    unweighted = run_validity_test(
        TOY,
        "y",
        "d",
        "z",
        weights="w",
        weighting=ValidityWeighting.UNWEIGHTED,
        bootstrap_config=1,
    )
    expected_unweighted, _, _ = _compute_statistic_by_hand(
        TOY, weights="w", weighting=ValidityWeighting.UNWEIGHTED
    )
    assert unweighted.statistic == pytest.approx(expected_unweighted, abs=1e-10)


def test_instrument_coding_is_taken_as_given():
    # z = 1 is the "on" arm; flipping the coding swaps on and off and therefore
    # flips the sign of the first stage. The statistic is the same up to the
    # sign of the violation it looks for, so it need not be invariant.
    res = run_validity_test(TOY, "y", "d", "z", bootstrap_config=BootstrapConfig(20, seed=3))
    flipped = TOY.with_columns((1 - pl.col("z")).alias("z"))
    res_flipped = run_validity_test(
        flipped, "y", "d", "z", bootstrap_config=BootstrapConfig(20, seed=3)
    )

    assert res.first_stage == pytest.approx(
        (TOY["d"].to_numpy()[TOY["z"].to_numpy() == 1].mean())
        - (TOY["d"].to_numpy()[TOY["z"].to_numpy() == 0].mean())
    )
    assert res_flipped.first_stage == pytest.approx(-res.first_stage)
    assert (res.binding["z_low"], res.binding["z_high"]) == (0, 1)


def test_statistic_matches_r_reference_on_simulated_data():
    # Reference: ivcheck::iv_kitagawa(y, d, z, n_boot = 50)
    # statistic = 1.014194. The pooled bootstrap changes only the p value.
    ds = simulate_data(n=500, seed=1)
    res = run_validity_test(
        ds, "y", "treatment", "instrument", bootstrap_config=BootstrapConfig(200, seed=1)
    )

    assert isinstance(res, InstrumentValidityResult)
    assert res.statistic == pytest.approx(1.014194, abs=1e-5)
    assert res.first_stage > 0
    assert res.verdict == "Cannot reject IV validity"


def test_rejects_card_instrument_without_covariates():
    # Reference: ivcheck::iv_kitagawa(card1995$lwage, card1995$college,
    # card1995$near_college, n_boot = 100); statistic = 5.316233, p value = 0.
    ds = load_card(treatment_cutoff=16)
    res = run_validity_test(
        ds, "lwage", "college", "nearc4", bootstrap_config=BootstrapConfig(200, seed=4)
    )

    assert res.statistic == pytest.approx(5.316233, abs=1e-5)
    assert res.p_value < 0.05
    assert res.verdict == "Reject IV validity"
    assert res.binding is not None
    assert res.binding["direction"] == "d = 0"
    assert res.binding["y_lower"] == pytest.approx(6.249975, abs=1e-6)
    assert res.binding["y_upper"] == pytest.approx(7.784889, abs=1e-6)


def test_rejects_clear_exclusion_violation():
    rng = np.random.default_rng(2)
    instrument = rng.integers(0, 2, size=1500)
    treatment = rng.binomial(1, 0.3 + 0.4 * instrument)
    outcome = rng.normal(treatment + 2.0 * instrument, 1.0)
    ds = pl.DataFrame({"y": outcome, "treatment": treatment, "instrument": instrument})

    res = run_validity_test(
        ds, "y", "treatment", "instrument", bootstrap_config=BootstrapConfig(200, seed=2)
    )

    assert res.p_value < 0.05
    assert res.verdict == "Reject IV validity"
    assert res.binding is not None
    assert res.binding["direction"] == "d = 1"
    assert res.binding["y_lower"] == float("-inf")


def test_weight_scale_leaves_statistic_and_p_value_unchanged():
    # Mean-one normalisation: multiplying a weight column by a constant must not
    # move the statistic or the bootstrap distribution it is compared against.
    ds = load_card(treatment_cutoff=16).with_columns(
        (0.5 + pl.int_range(pl.len()).mod(3)).cast(pl.Float64).alias("w")
    )
    scaled = ds.with_columns(pl.col("w") * 5.0)
    config = BootstrapConfig(100, seed=7)

    base = run_validity_test(ds, "lwage", "college", "nearc4", weights="w", bootstrap_config=config)
    rescaled = run_validity_test(
        scaled, "lwage", "college", "nearc4", weights="w", bootstrap_config=config
    )

    assert rescaled.statistic == pytest.approx(base.statistic, rel=1e-12)
    assert rescaled.p_value == pytest.approx(base.p_value, abs=1e-12)
    np.testing.assert_allclose(rescaled.boot_stats, base.boot_stats, rtol=1e-12)


def test_unit_weights_reproduce_the_unweighted_test():
    # With unit weights the case bootstrap must be exactly Algorithm 2.1, so the
    # bootstrap realisations coincide with the no-weights run draw by draw.
    ds = simulate_data(n=400, seed=5)
    config = BootstrapConfig(50, seed=3)

    unweighted = run_validity_test(ds, "y", "treatment", "instrument", bootstrap_config=config)
    unit_weighted = run_validity_test(
        ds.with_columns(pl.lit(1.0).alias("w")),
        "y",
        "treatment",
        "instrument",
        weights="w",
        bootstrap_config=config,
    )

    assert unit_weighted.statistic == pytest.approx(unweighted.statistic, rel=1e-12)
    np.testing.assert_allclose(unit_weighted.boot_stats, unweighted.boot_stats, rtol=1e-12)


def test_weights_enter_the_bootstrap():
    # Guards against the weights silently dropping out of the bootstrap world.
    ds = simulate_data(n=400, seed=5).with_columns(
        (0.2 + pl.int_range(pl.len()).mod(7)).cast(pl.Float64).alias("w")
    )
    config = BootstrapConfig(50, seed=3)

    unweighted = run_validity_test(ds, "y", "treatment", "instrument", bootstrap_config=config)
    weighted = run_validity_test(
        ds, "y", "treatment", "instrument", weights="w", bootstrap_config=config
    )

    assert not np.allclose(unweighted.boot_stats, weighted.boot_stats)


def test_bootstrap_is_deterministic_and_uses_the_pool():
    # The bootstrap draws from the pooled (Y, D), so permuting the instrument
    # within the two arms leaves the pool untouched; only the observed CDFs and
    # hence the observed statistic change. The bootstrap distribution must not.
    ds = simulate_data(n=400, seed=5)
    res = run_validity_test(
        ds, "y", "treatment", "instrument", bootstrap_config=BootstrapConfig(50, seed=9)
    )
    again = run_validity_test(
        ds, "y", "treatment", "instrument", bootstrap_config=BootstrapConfig(50, seed=9)
    )

    assert res.statistic == again.statistic
    assert res.p_value == again.p_value
    np.testing.assert_array_equal(res.boot_stats, again.boot_stats)

    assert res.n_boot == 50
    assert res.boot_stats.shape == (50,)
    assert 0.0 <= res.p_value <= 1.0
    assert np.all(res.boot_stats >= 0.0)


def test_result_tables_render():
    res = run_validity_test(TOY, "y", "d", "z", bootstrap_config=20)
    assert isinstance(res.get_as_dataframe(), pl.DataFrame)
    assert isinstance(res.tabulate(), GT)
    assert isinstance(res.summarize(), Group)
