"""Tests for the shared bootstrap helpers in :mod:`qte.bootstrap`."""

import polars as pl
from pytest import fixture

from qte.bootstrap import (
    BootstrapConfig,
    _make_block_bootstrap_run,
    _make_bootstrap_config,
    _make_bootstrap_runs,
)


@fixture
def panel():
    return pl.DataFrame({"unit": [1, 1, 2, 2, 3, 3], "v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})


def get_rows(ds):
    return ds.height


def total(ds):
    return float(ds["v"].sum())


def block_sizes(ds):
    return ds.group_by("unit").len()["len"].sort().to_list()


def pairs(ds):
    return sorted(zip(ds["unit"], ds["v"]))


def test_make_bootstrap_config_passes_config_through():
    cfg = BootstrapConfig(n_iter=3, seed=7, n_workers=2)
    assert _make_bootstrap_config(5) == BootstrapConfig(n_iter=5)
    assert _make_bootstrap_config(cfg) is cfg


def test_make_bootstrap_runs_yields_one_resample_per_iteration(panel):
    runs = list(_make_bootstrap_runs(panel, get_rows, n_iter=5, seed=1))
    assert len(runs) == 5
    assert set(runs) == {panel.height}


def test_make_bootstrap_runs_is_seed_reproducible(panel):
    first = list(_make_bootstrap_runs(panel, total, n_iter=5, seed=1))
    assert first == list(_make_bootstrap_runs(panel, total, n_iter=5, seed=1))
    assert first != list(_make_bootstrap_runs(panel, total, n_iter=5, seed=2))


def test_make_block_bootstrap_run_resamples_whole_blocks(panel):
    runs = list(_make_block_bootstrap_run(panel, block_sizes, "unit", n_iter=5, seed=1))
    assert runs == [[2, 2, 2]] * 5  # All block sizes remain of size 2.


def test_make_block_bootstrap_run_is_seed_reproducible(panel):
    first = list(_make_block_bootstrap_run(panel, pairs, "unit", n_iter=5, seed=1))
    assert first == list(_make_block_bootstrap_run(panel, pairs, "unit", n_iter=5, seed=1))
    assert first != list(_make_block_bootstrap_run(panel, pairs, "unit", n_iter=5, seed=2))


def test_make_bootstrap_runs_parallel_matches_sequential(panel):
    serial = list(_make_bootstrap_runs(panel, total, n_iter=5, seed=1))
    parallel = list(_make_bootstrap_runs(panel, total, n_iter=5, seed=1, n_workers=2))
    assert parallel == serial


def test_make_block_bootstrap_run_parallel_matches_sequential(panel):
    serial = list(_make_block_bootstrap_run(panel, pairs, "unit", n_iter=5, seed=1))
    parallel = list(_make_block_bootstrap_run(panel, pairs, "unit", n_iter=5, seed=1, n_workers=2))
    assert parallel == serial
