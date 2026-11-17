"""US equities, daily, with constituent eligibility independent of quote history.

Data: data/us_equity_daily/panel.parquet (date, symbol, open, high, low, close, volume, adj_close, adj_factor).
Prices are total-return adjusted (raw x adj_factor). Execution: a symmetric fee of cost_rate x open per share,
borrow cost on short value; geometric Sharpe over 252 days. Universe: the day's members with a price.
"""
from __future__ import annotations

import numpy as np

from foundry.config import Clock, Execution
from foundry.market.intraday import IntradayData
from foundry.market.orders import equity_bars

from foundry.market.execution import cash_spec
from foundry.market.clock import daily_clock, daily_schedule
from foundry.types import Allocation, Backend, BoolArray, ExecutionSpec, LoadedStudy, MarketData, Study, Window
from foundry.market.panel import Field, Panel
from foundry.market.base import Market, ohlcv_fields, trend_fields


class UsEquity(Market[MarketData | IntradayData, ExecutionSpec]):
    name = 'us_equity'
    ppy = 252.0
    geometric = True
    categories = {'pricevolume': 'scale-free daily price/volume primitives',
                  'trend': 'distance to moving average, trailing return'}

    def __init__(self, cost_rate: float, daily_borrow: float, gross_limit: float = 1.,
                 clock: Clock = Clock(), execution: Execution = Execution()):
        self.cost_rate, self.daily_borrow = cost_rate, daily_borrow
        self.allocation = Allocation(gross_limit, False)
        self.clock, self.execution = clock, execution

    def compile(self, data: MarketData | IntradayData, backend: Backend, study: Study) -> LoadedStudy[UsEquity, Panel, ExecutionSpec]:
        if isinstance(data, IntradayData):
            return self._intraday(data, backend, study)
        if self.execution.model == 'orders':
            raise ValueError('Equity orders require market-bars-v2 with explicit raw prices and execution rules')
        if self.clock != Clock():
            raise ValueError('Daily equity artifacts require 1D decision and statistics clocks')
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

    def _intraday(self, data: IntradayData, backend: Backend, study: Study) -> LoadedStudy[UsEquity, Panel, ExecutionSpec]:
        if self.execution.model != 'orders':
            raise ValueError('Equity intraday artifacts require the orders execution model')
        bars = data.panel
        start_time = np.datetime64(study.start.removesuffix('Z'))
        start = int(np.searchsorted(bars.dates, start_time))
        measurement_start = int(np.searchsorted(data.measurement_times, start_time))
        window = Window(data.clock, start, data.measurement_times, measurement_start, data.periods_per_year)
        src = dict(bars.values, member=bars.members, trading=np.isfinite(bars.values['open']))
        fields = [Field(k, 'price', formula=False) for k in src] + ohlcv_fields() + trend_fields()
        program_spec = equity_bars(self.name, data.schedule, data.event_close_ns, data.execution)
        # The cash specification carries common schedule/shape metadata only;
        # every candidate executes through the order account composition path.
        spec = cash_spec(program_spec.mark_open, program_spec.mark_close,
                         np.zeros(program_spec.mark_open.shape), schedule=data.schedule)
        self.ppy = data.periods_per_year
        self.decision_clock = data.clock.provenance
        self.execution_clock = 'next_regular_base_bar_open'
        return LoadedStudy(study, self, Panel(bars.dates, bars.assets, fields, src, backend), spec,
                           window, bars.contract, program_spec)

    def universe(self, panel: Panel) -> BoolArray:
        return panel['member'] & panel.backend.xp.isfinite(panel['close'])
