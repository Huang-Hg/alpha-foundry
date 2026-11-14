"""US equities, daily, with constituent eligibility independent of quote history.

Data: data/us_equity_daily/panel.parquet (date, symbol, open, high, low, close, volume, adj_close, adj_factor).
Prices are total-return adjusted (raw x adj_factor). Execution: a symmetric fee of cost_rate x open per share,
borrow cost on short value; geometric Sharpe over 252 days. Universe: the day's members with a price.
"""
from __future__ import annotations

import numpy as np

from foundry.backtest.spec import ExecutionSpec
from foundry.contracts import Allocation, LoadedStudy, Window
from foundry.core.panel import Field, Panel
from foundry.market.base import Market, ohlcv_fields, trend_fields


class UsEquity(Market):
    allocation = Allocation(1., False)
    name = 'us_equity'
    ppy = 252.0
    geometric = True
    categories = {'pricevolume': 'scale-free daily price/volume primitives',
                  'trend': 'distance to moving average, trailing return'}

    def __init__(self, cost_rate: float, daily_borrow: float):
        self.cost_rate, self.daily_borrow = cost_rate, daily_borrow

    def compile(self, data, backend, study):
        dates, symbols, g = data.dates, data.assets, data.values
        start = int(np.searchsorted(dates, np.datetime64(study.start)))
        src = {k: g[k] * g['adj_factor'] for k in ('open', 'high', 'low', 'close')}
        src['volume'] = g['volume']
        src['member'] = data.members
        fields = [Field(k, 'price', formula=False) for k in src] + ohlcv_fields() + trend_fields()
        carry = None if self.daily_borrow == 0 else np.full(src['close'].shape, self.daily_borrow)
        spec = ExecutionSpec.build(src['open'], src['close'], self.cost_rate * src['open'], carry=carry)
        return LoadedStudy(study, self, Panel(dates, symbols, fields, src, backend), spec,
                           Window(dates, start), data.contract)

    def universe(self, panel):
        return panel['member'] & panel.backend.xp.isfinite(panel['close'])
