"""Bootstrap resampling helpers and the public `BootstrapConfig`."""

import itertools
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Protocol, TypeVar

import numpy as np
import polars as pl

from qte.names import EFFECT_ID, QUANTILE_ID, SE_ID

T = TypeVar("T")


@dataclass(frozen=True)
class BootstrapConfig:
    """Configuration for the bootstrap.

    Parameters
    ----------
    n_iter : int, default=100
        Number of bootstrap replications.
    seed : int, optional
        Seed for the random number generator. ``None`` draws a fresh seed.
    n_workers : int, default=1
        Number of worker threads.
    """

    n_iter: int = 100
    seed: int | None = None
    n_workers: int = 1


class _BootstrapEstimates(Protocol):
    """Minimal interface of one bootstrap draw's estimates."""

    @property
    def qtt(self) -> pl.DataFrame: ...

    @property
    def att(self) -> pl.DataFrame: ...


def _bootstrap_standard_errors(
    runs: Iterable[_BootstrapEstimates],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Compute the standard error of the effect across bootstrap draws.

    Returns
    -------
    tuple of polars.DataFrame
        Per-quantile standard errors and average standard errors, both keyed by
        the corresponding point-estimate frame.
    """
    run_list = list(runs)
    standard_error = pl.col(EFFECT_ID).std().alias(SE_ID)
    qtt = pl.concat([r.qtt for r in run_list]).group_by(QUANTILE_ID).agg(standard_error)
    att = pl.concat([r.att for r in run_list]).select(standard_error)
    return qtt, att


def _attach_standard_errors(
    point: _BootstrapEstimates,
    runs: Iterable[_BootstrapEstimates],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Attach bootstrap standard errors of the effect to the point estimates."""
    qtt_se, att_se = _bootstrap_standard_errors(runs)
    qtt = point.qtt.join(qtt_se, on=QUANTILE_ID, how="left")
    att = point.att.with_columns(att_se[SE_ID])
    return qtt, att


def _make_bootstrap_config(config: BootstrapConfig | int) -> BootstrapConfig:
    """Coerce an integer shorthand into a `BootstrapConfig`."""
    return config if isinstance(config, BootstrapConfig) else BootstrapConfig(config)


def _make_seeds(seed: int | None, n_iter: int) -> Iterable[int | None]:
    if seed is None:
        return itertools.repeat(seed, n_iter)
    seed_seq = np.random.SeedSequence(seed)
    return (child.generate_state(1)[0] for child in seed_seq.spawn(n_iter))


def _threaded_map[T](
    task: Callable[[int | None], T], seeds: Iterable[int | None], n_workers: int
) -> Iterator[T]:
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        yield from executor.map(task, seeds)


def _map[T](
    task: Callable[[int | None], T],
    n_iter: int,
    seed: int | None,
    n_workers: int,
) -> Iterable[T]:
    """Run ``task`` once per bootstrap seed, optionally across threads.

    Results are always returned in seed order, so an arbitrary ``n_workers`` yields
    exactly the same sequence (and therefore the same estimates) as a serial
    run with the same ``seed``.
    """
    seeds = _make_seeds(seed, n_iter)
    if n_workers == 1:
        return (task(s) for s in seeds)
    return _threaded_map(task, seeds, n_workers)


def _bootstrap_once[T](ds: pl.DataFrame, fcn: Callable[[pl.DataFrame], T], seed: int | None) -> T:
    return fcn(ds.sample(fraction=1.0, with_replacement=True, seed=seed))


def _make_bootstrap_runs[T](
    ds: pl.DataFrame,
    fcn: Callable[[pl.DataFrame], T],
    *,
    n_iter: int = 100,
    seed: int | None = None,
    n_workers: int = 1,
) -> Iterable[T]:
    return _map(partial(_bootstrap_once, ds, fcn), n_iter, seed, n_workers)


def _block_bootstrap_once[T](
    ds: pl.DataFrame,
    units: pl.DataFrame,
    fcn: Callable[[pl.DataFrame], T],
    block_id: str,
    n_units: int,
    seed: int | None,
) -> T:
    # Resample unit IDs with replacement and assign new unique IDs. One seed per
    # iteration keeps the draws independent yet reproducible; one shared seed
    # would resample the same units and zero out the SE.
    sampled_units = units.sample(
        n=n_units,
        with_replacement=True,
        seed=seed,
    ).with_columns(pl.int_range(0, n_units).alias("new_id"))

    boot_ds = (
        sampled_units.join(ds, on=block_id, how="inner")
        .with_columns(pl.col("new_id").alias(block_id))
        .drop("new_id")
    )

    return fcn(boot_ds)


def _make_block_bootstrap_run[T](
    ds: pl.DataFrame,
    fcn: Callable[[pl.DataFrame], T],
    block_id: str,
    *,
    n_iter: int = 100,
    seed: int | None = None,
    n_workers: int = 1,
) -> Iterable[T]:
    # `maintain_order` matters: the default hash order varies between calls, which
    # would make a given seed non-reproducible.
    units = ds.select(block_id).unique(maintain_order=True)
    n_units = len(units)
    task = partial(_block_bootstrap_once, ds, units, fcn, block_id, n_units)
    return _map(task, n_iter, seed, n_workers)
