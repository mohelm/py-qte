"""Private types shared by the RD estimator and its presentation helpers."""

from enum import StrEnum


class Side(StrEnum):
    """Side of the cutoff."""

    LEFT = "left"
    RIGHT = "right"
