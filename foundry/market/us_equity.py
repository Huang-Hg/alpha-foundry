"""US equities, daily, with constituent eligibility independent of quote history.

Data: data/us_equity_daily/panel.parquet (date, symbol, open, high, low, close, volume, adj_close, adj_factor).
Prices are total-return adjusted (raw x adj_factor). Execution: a symmetric fee of cost_rate x open per share,
borrow cost on short value; geometric Sharpe over 252 days. Universe: the day's members with a price.
"""
from __future__ import annotations

import numpy as np

from foundry.market.execution import cash_spec
from foundry.market.clock import daily_clock, daily_schedule
from foundry.types import Allocation, Backend, BoolArray, ExecutionSpec, LoadedStudy, MarketData, Study, Window
from foundry.market.panel import Field, Panel
from foundry.market.base import Market, ohlcv_fields, trend_fields


class UsEquity(Market[MarketData, ExecutionSpec]):
    name = 'us_equity'
    ppy = 252.0
    geometric = True
    categories = {'pricevolume': 'scale-free daily price/volume primitives',
                  'trend': 'distance to moving average, trailing return'}

    def __init__(self, cost_rate: float, daily_borrow: float, gross_limit: float = 1.):
        self.cost_rate, self.daily_borrow = cost_rate, daily_borrow
        self.allocation = Allocation(gross_limit, False)

    def compile(self, data: MarketData, backend: Backend, study: Study) -> LoadedStudy[UsEquity, Panel, ExecutionSpec]:
        dates, symbols, g = data.dates, data.assets, data.values
        clock = daily_clock(dates, self.name)
        start = int(np.searchsorted(dates, np.datetime64(study.start)))
        window = Window(clock, start, dates, start, self.ppy)
        src = {k: g[k] * g['adj_factor'] for k in ('open', 'high', 'low', 'close')}
        src['volume'] = g['volume']
        src['member'] = data.members
        fields = [Field(k, 'price', formula=False) for k in src] + ohlcv_fields() + trend_fields()
        carry = None if self.daily_borrow == 0 else np.full(src['close'].shape, self.daily_borrow)
        spec = cash_spec(src['open'], src['close'], self.cost_rate * src['open'], carry=carry,
                         schedule=daily_schedule(dates, self.name, clock))
        return LoadedStudy(study, self, Panel(dates, symbols, fields, src, backend), spec,
                           window, data.contract)

    def universe(self, panel: Panel) -> BoolArray:
        return panel['member'] & panel.backend.xp.isfinite(panel['close'])
