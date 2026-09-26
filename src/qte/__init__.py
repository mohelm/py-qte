"""Quantile Treatment Effects (qte)."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("py-qte")
except PackageNotFoundError:  # pragma: no cover - uninstalled source tree
    __version__ = "0.0.0.dev0"
