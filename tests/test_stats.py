import numpy as np
import polars as pl

from qte.stats import estimate_propensity_score, get_quantiles


def test_get_quantile():
    data = np.array([-1, 0, 1])
    res = get_quantiles(0.5, data)
    assert np.isclose(res, 0)


def test_get_quantiles():
    data = np.arange(-50, +51)
    expected_result = np.array([-25, 0, 25])
    quantiles = np.array([0.25, 0.5, 0.75])
    res = get_quantiles(quantiles, data)
    assert np.isclose(res, expected_result).all()


def test_weighted_propensity_score_matches_replication():
    rng = np.random.default_rng(0)
    n = 80
    ds = (
        pl.DataFrame(
            {
                "x": rng.normal(size=n),
                "w": rng.integers(1, 4, size=n).astype(float),
            }
        )
        .with_columns((1 / (1 + np.exp(-(0.5 + pl.col("x"))))).alias("p"))
        .with_columns(
            (pl.lit(rng.uniform(size=n)) < pl.col("p")).alias("y"),
        )
    )

    weighted = estimate_propensity_score(ds, "y", "x", "w").params.to_numpy()
    # Integer frequency weights are equivalent to replicating every row w_i times.
    replicated = ds.select(pl.all().repeat_by("w").explode()).drop("w")
    unweighted = estimate_propensity_score(replicated, "y", "x").params.to_numpy()

    assert np.allclose(weighted, unweighted, atol=1e-9)
