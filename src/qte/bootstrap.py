import itertools
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import NamedTuple, TypeVar

import numpy as np
import polars as pl

T = TypeVar("T")


class BootstrapConfig(NamedTuple):
    """Configuration for the bootstrap.

    Parameters
    ----------
    n_iter : int
        Number of bootstrap replications.
    seed : int, optional
    n_workers : int
        Number of worker threads.
    """

    n_iter: int = 100
    seed: int | None = None
    n_workers: int = 1


def make_bootstrap_config(config: BootstrapConfig | int) -> BootstrapConfig:
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


def perform_bootstrap[T](
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
        sampled_units
        .join(ds, on=block_id, how="inner")
        .with_columns(pl.col("new_id").alias(block_id))
        .drop("new_id")
    )

    return fcn(boot_ds)


def perform_block_bootstrap[T](
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
