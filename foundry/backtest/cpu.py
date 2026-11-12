"""CPU execution: the reference kernel (foundry/backtest/src, extension _bt) runs every ExecutionSpec."""
from __future__ import annotations

import numpy as np

from foundry.backtest import _bt
from foundry.backtest.spec import ExecutionSpec


def simulate(W: np.ndarray, spec: ExecutionSpec) -> np.ndarray:
    """Target weights (T,S) decided at each close -> net daily returns (T,)."""
    return _bt.simulate(np.ascontiguousarray(W, dtype=np.float64), spec.open_raw, spec.prev_close, spec.open_ff0,
                        spec.close_ff0, spec.cost_buy, spec.cost_sell, spec.live, spec.block, spec.carry)


def book(signal: np.ndarray, universe: np.ndarray) -> np.ndarray:
    """Long-only book: exposure clip[0,1] in the day's universe (NaN = cash), one equal slot per asset."""
    slots = np.maximum(universe.sum(1, keepdims=True), 1).astype(np.float64)
    return np.where(universe & np.isfinite(signal), np.clip(signal, 0.0, 1.0), 0.0) / slots
