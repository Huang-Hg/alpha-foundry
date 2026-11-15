"""Allocation semantics shared by formulas, arrays and portfolio weighting."""
from types import ModuleType

from foundry.types import Allocation, BoolArray, FloatArray, IntArray


def signal_to_book(values: FloatArray, universe: BoolArray, slots: FloatArray | IntArray,
                   allocation: Allocation, xp: ModuleType) -> FloatArray:
    """Map resident (T,S) signals to eligible exposure using (T,1) slot counts."""
    lower = -1. if allocation.allow_short else 0.
    return xp.where(universe & xp.isfinite(values), xp.clip(values, lower, 1.), 0.) / slots
