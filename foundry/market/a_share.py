"""China A-shares, daily (historical CSI 300 constituents), long only.

Data: data/a_share_pit/panel.parquet (date, symbol, open, close, high, low, volume, exec_open, exec_close), raw prices for signals, cumulative historical adjustment (hfq) for execution. Execution:
  - fees per share: commission + transfer fee on both sides, stamp tax on sells only;
  - price limits: an open at the upper limit cannot be bought, at the lower limit cannot be sold (limit 20% for
    ChiNext 300/301 and STAR 688 codes, 10% otherwise; the check is on the open, which is known when trading);
  - suspension: a missing open after the first observed open is a suspension: prices carry forward,
    nothing can be traded, the position is held (T+1 needs no extra rule: yesterday's decision trades at today's open).
Geometric Sharpe over 242 days. Universe: the day's members that trade.
"""
from __future__ import annotations

import numpy as np

from foundry.backtest.spec import ExecutionSpec, ffill, previous
from foundry.contracts import Allocation, LoadedStudy, Window
from foundry.core.panel import Field, Panel
from foundry.market.base import Market, ohlcv_fields, trend_fields

NO_BUY, NO_SELL = 1, 2
TOLERANCE = 1e-3                              # adjusted prices are no longer on the 0.01 tick


class AShare(Market):
    allocation = Allocation(1., False)
    name = 'a_share'
    ppy = 242.0
    geometric = True
    categories = {'pricevolume': 'scale-free daily price/volume primitives',
                  'trend': 'distance to moving average, trailing return'}

    def __init__(self, commission: float, transfer_fee: float, stamp_tax_sell: float):
        self.commission, self.transfer_fee, self.stamp_tax_sell = commission, transfer_fee, stamp_tax_sell

    def compile(self, data, backend, study):
        dates, symbols, g = data.dates, data.assets, data.values
        start = int(np.searchsorted(dates, np.datetime64(study.start)))
        src = {k:g[k] for k in ('open','high','low','close','volume')}
        src['trading'] = np.isfinite(g['open'])
        src['member'] = data.members
        fields = [Field(k, 'price', formula=False) for k in src] + ohlcv_fields() + trend_fields()
        # Quote gaps are suspensions after the first observed open. No future
        # last-quote date is consulted and index exits do not erase price history.
        seen = np.maximum.accumulate(np.isfinite(g['exec_open']),axis=0)
        open_, close = (np.where(seen, ffill(g[k]), np.nan) for k in ('exec_open', 'exec_close'))
        suspended = seen & ~np.isfinite(g['exec_open'])
        limit = np.where(np.array([s.startswith(('300', '301', '688')) for s in symbols]), .20, .10)
        gap = g['exec_open'] / previous(close) - 1
        block = (np.where(gap >= limit - TOLERANCE, NO_BUY, 0) | np.where(gap <= -limit + TOLERANCE, NO_SELL, 0)
                 | np.where(suspended, NO_BUY | NO_SELL, 0))
        fee = self.commission + self.transfer_fee
        spec = ExecutionSpec.build(open_, close, fee * open_, cost_sell=(fee + self.stamp_tax_sell) * open_, block=block)
        return LoadedStudy(study, self, Panel(dates, symbols, fields, src, backend), spec,
                           Window(dates, start), data.contract)

    def universe(self, panel):
        return panel['member'] & panel['trading']
