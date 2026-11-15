"""Compile daily cash-market prices into immutable execution arrays."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from foundry.types import ExecutionSpec

EPS = 1e-7


def ffill(x: NDArray[np.floating]) -> NDArray[np.floating]:
    """Forward fill over rows; leading NaN stays NaN, without back filling."""
    T, S = x.shape
    idx = np.where(np.isfinite(x), np.arange(T, dtype=np.int64)[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    return x[idx, np.arange(S, dtype=np.int64)[None, :]]


def previous(x: NDArray[np.floating]) -> NDArray[np.floating]:
    return np.vstack([np.full((1, x.shape[1]), np.nan), x[:-1]])


def cash_spec(open_: NDArray[np.floating], close: NDArray[np.floating], cost_buy: NDArray[np.floating],
              cost_sell: NDArray[np.floating] | None = None, block: NDArray[np.integer] | None = None,
              carry: NDArray[np.floating] | None = None) -> ExecutionSpec:
    """Prices in the kernel's form from raw open/close (NaN = no price that day)."""
    c = np.ascontiguousarray
    open_ff, close_ff = ffill(open_), ffill(close)
    open_ff0 = np.where(np.isnan(open_ff), 0.0, open_ff)
    live = np.isfinite(open_) & np.isfinite(cost_buy) & (open_ff0 > EPS)
    return ExecutionSpec(c(open_), c(ffill(previous(close))), c(open_ff0), c(np.where(np.isnan(close_ff), 0.0, close_ff)),
                         c(cost_buy), c(live.astype(np.uint8)), None if cost_sell is None else c(cost_sell),
                         None if block is None else c(block.astype(np.uint8)), None if carry is None else c(carry))
