import itertools
from collections.abc import Callable, Iterable
from typing import NamedTuple, TypeVar

import numpy as np
import polars as pl

T = TypeVar("T")


class BootstrapConfig(NamedTuple):
    n_iter: int = 100
    seed: int | None = None


def make_bootstrap_config(config: BootstrapConfig | int) -> BootstrapConfig:
    return config if isinstance(config, BootstrapConfig) else BootstrapConfig(config)


def _make_seeds(seed: int | None, n_iter: int) -> Iterable[int | None]:
    if seed is None:
        return itertools.repeat(seed, n_iter)
    seed_seq = np.random.SeedSequence(seed)
    return (child.generate_state(1)[0] for child in seed_seq.spawn(n_iter))


def perform_bootstrap[T](
    ds: pl.DataFrame,
    fcn: Callable[[pl.DataFrame], T],
    n_iter: int = 100,
    seed: int | None = None,
) -> Iterable[T]:
    return (
        fcn(ds.sample(fraction=1.0, with_replacement=True, seed=s))
        for s in _make_seeds(seed, n_iter)
    )


def perform_block_bootstrap[T](
    ds: pl.DataFrame,
    fcn: Callable[[pl.DataFrame], T],
    block_id: str,
    n_iter: int = 100,
    seed: int | None = None,
) -> Iterable[T]:
    # `maintain_order` matters: the default hash order varies between calls, which
    # would make a given seed non-reproducible.
    units = ds.select(block_id).unique(maintain_order=True)
    n_units = len(units)
    for s in _make_seeds(seed, n_iter):
        # Resample unit IDs with replacement and assign new unique IDs. `seed + i`
        # keeps the draws independent yet reproducible; one seed for every
        # iteration would resample the same units and zero out the SE.
        sampled_units = units.sample(
            n=n_units,
            with_replacement=True,
            seed=s,
        ).with_columns(pl.int_range(0, n_units).alias("new_id"))

        boot_ds = (
            sampled_units.join(ds, on=block_id, how="inner")
            .with_columns(pl.col("new_id").alias(block_id))
            .drop("new_id")
        )

        yield fcn(boot_ds)
