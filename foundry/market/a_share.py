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

from foundry.config import Clock, Execution
from foundry.market.intraday import IntradayData
from foundry.market.orders import equity_bars

from foundry.market.execution import cash_spec, ffill, previous
from foundry.market.clock import daily_clock, daily_schedule
from foundry.types import Allocation, Backend, BoolArray, ExecutionSpec, LoadedStudy, MarketData, Study, Window
from foundry.market.panel import Field, Panel
from foundry.market.base import Market, ohlcv_fields, trend_fields

NO_BUY, NO_SELL = 1, 2
TOLERANCE = 1e-3                              # adjusted prices are no longer on the 0.01 tick


class AShare(Market[MarketData | IntradayData, ExecutionSpec]):
    name = 'a_share'
    ppy = 242.0
    geometric = True
    categories = {'pricevolume': 'scale-free daily price/volume primitives',
                  'trend': 'distance to moving average, trailing return'}

    def __init__(self, commission: float, transfer_fee: float, stamp_tax_sell: float, gross_limit: float = 1.,
                 clock: Clock = Clock(), execution: Execution = Execution()):
        self.commission, self.transfer_fee, self.stamp_tax_sell = commission, transfer_fee, stamp_tax_sell
        self.allocation = Allocation(gross_limit, False)
        self.clock, self.execution = clock, execution

    def compile(self, data: MarketData | IntradayData, backend: Backend, study: Study) -> LoadedStudy[AShare, Panel, ExecutionSpec]:
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
        spec = cash_spec(open_, close, fee * open_, cost_sell=(fee + self.stamp_tax_sell) * open_, block=block,
                         schedule=daily_schedule(dates, self.name, clock))
        return LoadedStudy(study, self, Panel(dates, symbols, fields, src, backend), spec,
                           window, data.contract)

    def _intraday(self, data: IntradayData, backend: Backend, study: Study) -> LoadedStudy[AShare, Panel, ExecutionSpec]:
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
        return panel['member'] & panel['trading']
