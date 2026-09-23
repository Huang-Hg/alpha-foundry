"""China A-shares, daily (CSI 300 constituents, point-in-time membership applied when the parquet is built), long only.

Data: data/a_share_daily/panel.parquet (date, symbol, open, close, high, low, volume, amount, pct_change), prices
forward-adjusted (qfq). Execution:
  - fees per share: commission + transfer fee on both sides, stamp tax on sells only;
  - price limits: an open at the upper limit cannot be bought, at the lower limit cannot be sold (limit 20% for
    ChiNext 300/301 and STAR 688 codes, 10% otherwise; the check is on the open, which is known when trading);
  - suspension: a symbol-day without a row inside the symbol's data span is a suspension: prices carry forward,
    nothing can be traded, the position is held (T+1 needs no extra rule: yesterday's decision trades at today's open).
Geometric Sharpe over 242 days. Universe: the day's members that trade.
"""
from __future__ import annotations

import numpy as np

from foundry.backtest.spec import ExecutionSpec, ffill, previous
from foundry.config import AShareConfig, path
from foundry.core.panel import Field, Panel
from foundry.market.base import Market, Window, long_table, ohlcv_fields, trend_fields

NO_BUY, NO_SELL = 1, 2
TOLERANCE = 1e-3                              # adjusted prices are no longer on the 0.01 tick


def within_span(x: np.ndarray) -> np.ndarray:
    """Rows between a column's first and last value (a gap inside is a suspension, outside is no listing)."""
    fin = np.isfinite(x)
    rows = np.arange(len(x))[:, None]
    first = np.where(fin.any(0), fin.argmax(0), len(x))
    last = len(x) - 1 - fin[::-1].argmax(0)
    return (rows >= first) & (rows <= last)


class AShare(Market):
    name = 'a_share'
    ppy = 242.0
    geometric = True
    categories = {'pricevolume': 'scale-free daily price/volume primitives',
                  'trend': 'distance to moving average, trailing return'}

    def __init__(self):
        self.cfg = AShareConfig()

    def load(self, backend):
        dates, symbols, g, start = long_table(path(self.cfg.data_dir) / 'panel.parquet', self.cfg.start, self.cfg.end,
                                              ('open', 'high', 'low', 'close', 'volume'))
        src = dict(g)
        src['trading'] = np.isfinite(g['close']).astype(np.float64)
        fields = [Field(k, 'price', formula=False) for k in src] + ohlcv_fields() + trend_fields()
        span = within_span(g['close'])
        open_, close = (np.where(span, ffill(g[k]), np.nan) for k in ('open', 'close'))
        suspended = span & ~np.isfinite(g['close'])
        limit = np.where(np.array([s.startswith(('300', '301', '688')) for s in symbols]), .20, .10)
        gap = g['open'] / previous(close) - 1
        block = (np.where(gap >= limit - TOLERANCE, NO_BUY, 0) | np.where(gap <= -limit + TOLERANCE, NO_SELL, 0)
                 | np.where(suspended, NO_BUY | NO_SELL, 0))
        fee = self.cfg.commission + self.cfg.transfer_fee
        spec = ExecutionSpec.build(open_, close, fee * open_, cost_sell=(fee + self.cfg.stamp_tax_sell) * open_, block=block)
        return Panel(dates, symbols, fields, src, backend), spec, Window(dates, start)

    def universe(self, panel):
        return panel['trading'] > 0


MARKET = AShare()
