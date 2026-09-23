"""Statistics of daily net returns - the one definition every report and every admission check uses.

Functions take a batch of return rows r (B,T) (or one row) on numpy or cupy and return per-row values. Sharpe is
annualised with the market's periods per year `ppy`; the arithmetic form (mean / population std) is the Quantiacs
crypto convention, the geometric form uses log returns. A zero-variance row has Sharpe 0. Recency weights p (T,)
sum to 1 and halve every `halflife` days back from the last day.
"""
from __future__ import annotations

import numpy as np


def xp_of(a):
    return np if isinstance(a, np.ndarray) else __import__('cupy')


def sharpe(r, ppy, geometric=False):
    xp = xp_of(r)
    x = xp.log1p(r) if geometric else r
    sd = x.std(-1)
    return xp.where(sd > 0, x.mean(-1) / xp.where(sd > 0, sd, 1) * ppy ** 0.5, 0)


def recency(dates: np.ndarray, halflife: float, xp=np):
    """Weights over the days of `dates` (datetime64[D]), p[-1] largest, sum 1."""
    d = xp.asarray(dates.astype('int64'))
    p = xp.power(0.5, (d[-1] - d) / halflife)
    return p / p.sum()


def weighted_sharpe(r, p, ppy):
    xp = xp_of(r)
    mu = xp.sum(r * p, axis=-1)
    var = xp.sum((r - mu[..., None]) ** 2 * p, axis=-1)
    return xp.where(var > 0, mu / xp.sqrt(xp.where(var > 0, var, 1)) * ppy ** 0.5, 0)


def cagr(r, ppy):
    xp = xp_of(r)
    return xp.expm1(xp.log1p(r).mean(-1) * ppy)


def recent_growth(r, p, ppy):
    """Recency-weighted annualised log growth."""
    xp = xp_of(r)
    return xp.sum(xp.log1p(r) * p, axis=-1) * ppy


def max_drawdown(r):
    """Deepest fall of cumulative wealth from its running peak (the start counts as a peak of 1)."""
    xp = xp_of(r)
    cum = xp.cumprod(1 + r, axis=-1)
    peak = xp.maximum.accumulate(xp.maximum(cum, 1), axis=-1)
    return xp.minimum((cum / peak - 1).min(-1), 0)


def sound(r):
    """Every day finite and no day losing everything."""
    xp = xp_of(r)
    return xp.isfinite(r).all(-1) & (r > -1).all(-1)


class Years:
    """Calendar years of a statistics window as an indicator matrix, so per-year statistics of a batch are one
    matrix product."""

    def __init__(self, dates: np.ndarray, xp=np):
        y = dates.astype('datetime64[Y]').astype(int) + 1970
        self.years = np.unique(y)
        self.days = np.array([int((y == v).sum()) for v in self.years])
        self.M = xp.asarray((y[None, :] == self.years[:, None]).astype(np.float64))
        self.ndays = xp.asarray(self.days.astype(np.float64))

    def sharpe(self, r, ppy, last=None):
        """(B, Y) arithmetic Sharpe per calendar year; `last` keeps only the most recent years."""
        xp = xp_of(r)
        M, n = (self.M, self.ndays) if last is None else (self.M[len(self.M) - last:], self.ndays[len(self.ndays) - last:])
        mean = r @ M.T / n
        sd = xp.sqrt(xp.maximum((r * r) @ M.T / n - mean * mean, 0.0))
        return xp.where(sd > 0, mean / xp.where(sd > 0, sd, 1.0) * ppy ** 0.5, 0.0)

    def share(self, x):
        """(B, Y) mean of x per year (e.g. the share of days holding a position)."""
        return (x.astype(np.float64) @ self.M.T) / self.ndays

    def total(self, x):
        return x @ self.M.T


def summary(r: np.ndarray, ppy: float, geometric=False) -> dict:
    """Headline figures of one return series (reports)."""
    return dict(sharpe=float(sharpe(r, ppy, geometric)), cagr=float(cagr(r, ppy)), maxdd=float(max_drawdown(r)),
                ann_ret=float(r.mean() * ppy), volatility=float(r.std() * ppy ** 0.5), total_ret=float(np.prod(1 + r) - 1))
