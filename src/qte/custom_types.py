"""Shared type aliases used across the package."""

import polars as pl

type ColumnName = str
"""Name of a column in a :class:`polars.DataFrame`."""

type DataFrame = pl.DataFrame
"""A :class:`polars.DataFrame`."""

type FormulaRhs = str
"""Right-hand side of a formula, e.g. ``"age + education"``."""

type Series = pl.Series
"""A :class:`polars.Series`."""
