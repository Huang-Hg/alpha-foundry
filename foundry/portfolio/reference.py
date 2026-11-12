"""The reference a candidate is judged against: the current portfolio, backtested.

Protocol v6: members are ADDED with their weights, U = sum_i w_i * book_i, and the held book is
capped(U) = U / max(1, day's total exposure) - the portfolio shrinks only on days the sum exceeds full exposure (the
Quantiacs treatment of weights). An empty library starts from the full-exposure equal-weight book.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cupy as cp

from foundry.config import PortfolioConfig
from foundry.core import metrics as M
from foundry.portfolio.library import SEED

CFG = PortfolioConfig()


def capped(U):
    """Books are nonnegative, so a day's exposure is the plain row sum."""
    return U / cp.maximum(1, U.sum(-1, keepdims=True))


@dataclass
class Reference:
    U: object                      # raw additive sum of member books (T,S), zero before the window opens
    W: object                      # held book capped(U)
    rr: object                     # net daily returns over the window
    p: object                      # recency weights over the window
    metrics: dict                  # report statistics of the held book
    sharpe: float
    cagr: float
    maxdd: float
    wsharpe: float                 # recency-weighted Sharpe
    recent_growth: float           # recency-weighted annualised log growth
    n: int                         # members (the equal-weight seed counts as one)
    seed: bool
    keys: frozenset                # member formulas
    mixed_error: Optional[object] = None      # measured FP32-vs-FP64 error of this reference (precision path)
    force_f64: bool = field(default=False)    # a calibration breach seen on this reference: stay in FP64

    @staticmethod
    def build(engine, members, null_shift=False) -> 'Reference':
        seed = not members
        members = members or [SEED]
        formulas = [m['expr'] for m in members if m['expr'] != SEED['expr']]
        known = dict(zip(formulas, (b for b, _ in engine.members(formulas, null_shift))))
        books = [engine.equal_weight() if m['expr'] == SEED['expr'] else known[m['expr']] for m in members]
        U = cp.sum(cp.stack(books) * cp.asarray([m['weight'] for m in members])[:, None, None], axis=0)
        U[:engine.start] = 0                  # as in the backtest: nothing held before the window opens
        st = engine.metrics([capped(U)])[0]
        p = M.recency(engine.dates[engine.start:], CFG.recent_halflife_days, cp)
        ws, growth = cp.asnumpy(cp.stack((M.weighted_sharpe(st['rr'], p, engine.ppy), M.recent_growth(st['rr'], p, engine.ppy)))).tolist()
        return Reference(U=U, W=st['W'], rr=st['rr'], p=p, metrics={k: v for k, v in st.items() if k not in ('rr', 'W')},
                         sharpe=st['sharpe'], cagr=st['cagr'], maxdd=st['maxdd'], wsharpe=ws, recent_growth=growth,
                         n=len(books), seed=seed, keys=frozenset(m['expr'] for m in members))
