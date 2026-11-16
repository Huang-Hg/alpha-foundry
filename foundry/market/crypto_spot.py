"""Daily long-only crypto spot research on a continuous UTC calendar.

A completed UTC day's decision executes at the next UTC day's open. The
symmetric rate times open is a research cost assumption, not an exchange quote.
Missing opens after the first quote block both trade directions and carry prices
forward; absent delisting settlement events cannot be inferred from future data.
No funding, borrowing, leverage or price-limit rules are modeled.
"""
from __future__ import annotations

import numpy as np

from foundry.market.execution import cash_spec, ffill
from foundry.market.clock import daily_clock
from foundry.types import Allocation, Backend, BoolArray, ExecutionSpec, LoadedStudy, MarketData, Study, Window
from foundry.market.panel import Field, Panel
from foundry.market.base import Market, ohlcv_fields, trend_fields


class CryptoSpot(Market[MarketData, ExecutionSpec]):
    name = 'crypto_spot'
    decision_clock = 'utc_day_close'
    execution_clock = 'next_utc_day_open'
    ppy = 365.0
    geometric = True
    categories = {'pricevolume': 'scale-free daily price/volume primitives',
                  'trend': 'distance to moving average, trailing return'}

    def __init__(self, cost_rate: float, gross_limit: float = 1.):
        self.cost_rate = cost_rate
        self.allocation = Allocation(gross_limit, False)

    def compile(self, data: MarketData, backend: Backend, study: Study) -> LoadedStudy[CryptoSpot, Panel, ExecutionSpec]:
        dates, symbols, g = data.dates, data.assets, data.values
        clock = daily_clock(dates, self.name)
        start = int(np.searchsorted(dates, np.datetime64(study.start)))
        window = Window(dates, start, clock)
        src = {k: g[k] for k in ('open', 'high', 'low', 'close', 'volume')}
        src['trading'] = np.isfinite(g['open'])
        src['member'] = data.members
        fields = [Field(k, 'price', formula=False) for k in src] + ohlcv_fields() + trend_fields()
        seen = np.maximum.accumulate(src['trading'], axis=0)
        open_, close = (np.where(seen, ffill(g[k]), np.nan) for k in ('open', 'close'))
        block = np.where(seen & ~src['trading'], 3, 0)
        spec = cash_spec(open_, close, self.cost_rate * open_, block=block)
        return LoadedStudy(study, self, Panel(dates, symbols, fields, src, backend), spec,
                           window, data.contract)

    def universe(self, panel: Panel) -> BoolArray:
        return panel['member'] & panel['trading']
