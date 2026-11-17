"""Explicit score sizing shared by formulas and registered score arrays."""
from types import ModuleType

from foundry.config import Sizing
from foundry.types import Allocation, BoolArray, FloatArray, IntArray


def score_to_book(values: FloatArray, universe: BoolArray, slots: FloatArray | IntArray,
                  allocation: Allocation, sizing: Sizing, xp: ModuleType) -> FloatArray:
    """Map resident (...,D,S) scores to equity fractions on eligible assets.

    Missing scores produce zero targets. Equal slots divide a budget capped at
    one by eligible asset count; the default preserves exact clip/divide math.
    Independent sizing never divides by asset count or rescales across assets.
    The evaluation boundary judges its resulting per-candidate gross budget.
    """
    lower = -1. if allocation.allow_short else 0.
    scores = xp.where(universe & xp.isfinite(values), xp.clip(values, lower, 1.), 0.)
    if sizing.rule == 'equal_slots':
        if allocation.max_gross >= 1.:
            return scores / slots
        return scores * allocation.max_gross / slots
    return scores * sizing.scale
