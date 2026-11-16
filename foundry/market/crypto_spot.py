"""Daily long-only crypto spot research on a continuous UTC calendar.

A completed UTC day's decision executes at the next UTC day's open. The
symmetric rate times open is a research cost assumption, not an exchange quote.
Missing opens after the first quote block both trade directions and carry prices
forward; absent delisting settlement events cannot be inferred from future data.
No funding, borrowing, leverage or price-limit rules are modeled.
"""
from __future__ import annotations

import numpy as np

from foundry.config import Clock, Execution
from foundry.market.execution import cash_spec, ffill
from foundry.market.clock import daily_clock, daily_schedule
from foundry.market.intraday import IntradayData
from foundry.market.orders import crypto_bars
from foundry.types import Allocation, Backend, BoolArray, ExecutionSpec, LoadedStudy, MarketData, Study, Window
from foundry.market.panel import Field, Panel
from foundry.market.base import Market, ohlcv_fields, trend_fields


class CryptoSpot(Market[MarketData | IntradayData, ExecutionSpec]):
    name = 'crypto_spot'
    decision_clock = 'utc_day_close'
    execution_clock = 'next_utc_day_open'
    ppy = 365.0
    geometric = True
    categories = {'pricevolume': 'scale-free daily price/volume primitives',
                  'trend': 'distance to moving average, trailing return'}

    def __init__(self, cost_rate: float, gross_limit: float = 1., clock: Clock = Clock(),
                 execution: Execution = Execution()):
        self.cost_rate = cost_rate
        self.allocation = Allocation(gross_limit, False)
        self.clock = clock
        self.execution = execution

    def compile(self, data: MarketData | IntradayData, backend: Backend, study: Study) -> LoadedStudy[CryptoSpot, Panel, ExecutionSpec]:
        if isinstance(data, IntradayData):
            return self._intraday(data, backend, study)
        if self.clock != Clock():
            raise ValueError('Daily spot artifacts require 1D decision and statistics clocks')
        dates, symbols, g = data.dates, data.assets, data.values
        clock = daily_clock(dates, self.name)
        start = int(np.searchsorted(dates, np.datetime64(study.start)))
        window = Window(clock, start, dates, start, self.ppy)
        src = {k: g[k] for k in ('open', 'high', 'low', 'close', 'volume')}
        src['trading'] = np.isfinite(g['open'])
        src['member'] = data.members
        fields = [Field(k, 'price', formula=False) for k in src] + ohlcv_fields() + trend_fields()
        seen = np.maximum.accumulate(src['trading'], axis=0)
        open_, close = (np.where(seen, ffill(g[k]), np.nan) for k in ('open', 'close'))
        block = np.where(seen & ~src['trading'], 3, 0)
        spec = cash_spec(open_, close, self.cost_rate * open_, block=block,
                         schedule=daily_schedule(dates, self.name, clock))
        program_spec = crypto_bars(self.name, spec.schedule, clock.decision_ns, g, self.execution, self.cost_rate)
        return LoadedStudy(study, self, Panel(dates, symbols, fields, src, backend), spec,
                           window, data.contract, program_spec)

    def _intraday(self, data: IntradayData, backend: Backend, study: Study) -> LoadedStudy[CryptoSpot, Panel, ExecutionSpec]:
        bars = data.panel
        start_time = np.datetime64(study.start.removesuffix('Z'))
        start = int(np.searchsorted(bars.dates, start_time))
        measurement_start = int(np.searchsorted(data.measurement_times, start_time))
        window = Window(data.clock, start, data.measurement_times, measurement_start, data.periods_per_year)
        src = dict(bars.values, member=bars.members, trading=np.isfinite(bars.values['open']))
        fields = [Field(k, 'price', formula=False) for k in src] + ohlcv_fields() + trend_fields()
        g = data.execution
        trading = np.isfinite(g['open'])
        seen = np.maximum.accumulate(trading, axis=0)
        open_, close = (np.where(seen, ffill(g[k]), np.nan) for k in ('open', 'close'))
        block = np.where(seen & ~trading, 3, 0)
        spec = cash_spec(open_, close, self.cost_rate * open_, block=block, schedule=data.schedule)
        self.ppy = data.periods_per_year
        self.decision_clock = data.clock.provenance
        self.execution_clock = 'next_base_bar_open'
        program_spec = crypto_bars(self.name, data.schedule, data.event_close_ns, g, self.execution, self.cost_rate)
        return LoadedStudy(study, self, Panel(bars.dates, bars.assets, fields, src, backend), spec,
                           window, bars.contract, program_spec)

    def universe(self, panel: Panel) -> BoolArray:
        return panel['member'] & panel['trading']
