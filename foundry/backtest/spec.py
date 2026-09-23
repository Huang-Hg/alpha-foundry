"""What a market's execution rules compile to: per (day, asset) arrays that the one market-agnostic kernel reads.

The kernel trades yesterday's target weights at today's open and marks to today's close (Quantiacs
calc_relative_return semantics, 1:1 with the platform). A market differs only in the arrays below:
    cost_buy / cost_sell   cost per share traded, in price units (a fee rate is rate x open)
    live                   the asset has a price today; without one it cannot trade and its position is closed at the
                           last price after the close
    block                  bit 0: cannot buy today, bit 1: cannot sell today (limit up / limit down / suspension);
                           a blocked asset keeps its position
    carry                  daily financing rate on the value of a short position
A spec without cost_sell / block / carry is symmetric: the CUDA kernel runs it; the CPU kernel runs every spec.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Optional

import numpy as np

EPS = 1e-7                                    # qnt.stats.EPS: an open below this is no price


def ffill(x: np.ndarray) -> np.ndarray:
    """Forward fill over rows; leading NaN stays NaN (qnt semantics, no back fill)."""
    T, S = x.shape
    idx = np.where(np.isfinite(x), np.arange(T, dtype=np.int64)[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    return x[idx, np.arange(S, dtype=np.int64)[None, :]]


def previous(x: np.ndarray) -> np.ndarray:
    return np.vstack([np.full((1, x.shape[1]), np.nan), x[:-1]])


@dataclass(frozen=True)
class ExecutionSpec:
    open_raw: np.ndarray
    prev_close: np.ndarray
    open_ff0: np.ndarray
    close_ff0: np.ndarray
    cost_buy: np.ndarray
    live: np.ndarray
    cost_sell: Optional[np.ndarray] = None
    block: Optional[np.ndarray] = None
    carry: Optional[np.ndarray] = None

    @staticmethod
    def build(open_: np.ndarray, close: np.ndarray, cost_buy: np.ndarray, cost_sell=None, block=None,
              carry=None) -> 'ExecutionSpec':
        """Prices in the kernel's form from raw open/close (NaN = no price that day)."""
        c = np.ascontiguousarray
        open_ff, close_ff = ffill(open_), ffill(close)
        open_ff0 = np.where(np.isnan(open_ff), 0.0, open_ff)
        live = np.isfinite(open_) & np.isfinite(close) & np.isfinite(cost_buy) & (open_ff0 > EPS)
        return ExecutionSpec(c(open_), c(ffill(previous(close))), c(open_ff0), c(np.where(np.isnan(close_ff), 0.0, close_ff)),
                             c(cost_buy), c(live.astype(np.uint8)), None if cost_sell is None else c(cost_sell),
                             None if block is None else c(block.astype(np.uint8)), None if carry is None else c(carry))

    @property
    def symmetric(self) -> bool:
        return self.cost_sell is None and self.block is None and self.carry is None

    def rows(self, end: int) -> 'ExecutionSpec':
        return replace(self, **{f.name: getattr(self, f.name)[:end] for f in fields(self) if getattr(self, f.name) is not None})
