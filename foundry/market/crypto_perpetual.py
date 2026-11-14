"""USDT linear isolated research market with daily decisions and hourly risk events.

This explicit constant-margin research schedule is not a historical exchange
risk-tier reconstruction. Intrabar liquidation uses adverse mark OHLC touches;
exact tick ordering, order-book fills, ADL and account VIP fees are not asserted.
"""
import numpy as np

from foundry.backtest.perpetual import PerpetualSpec
from foundry.contracts import Allocation, LoadedStudy, Window
from foundry.core.panel import Field, Panel
from foundry.market.base import Market, ohlcv_fields, trend_fields


class CryptoPerpetual(Market):
    name = 'crypto_perpetual'
    decision_clock = 'utc_day_close'
    execution_clock = 'next_utc_day_open_then_hourly_funding_and_mark_risk'
    ppy = 365.
    geometric = True
    categories = {'pricevolume': 'completed UTC daily trade-price/volume primitives',
                  'trend': 'daily trend; execution-only funding and mark risk stay outside features'}

    def __init__(self, leverage, cost_rate, maintenance_rate, liquidation_fee):
        self.allocation = Allocation(leverage, True)
        self.cost_rate, self.maintenance_rate, self.liquidation_fee = cost_rate, maintenance_rate, liquidation_fee

    def compile(self, data, backend, study):
        daily = data.daily
        src = dict(daily.values, member=daily.members)
        fields = [Field(k, 'price', formula=False) for k in src] + ohlcv_fields() + trend_fields()
        shape = data.trade_open.shape
        spec = PerpetualSpec(data.event_times_ms, data.day_index,
            np.r_[True, data.day_index[1:] != data.day_index[:-1]], data.trade_open,
            data.mark_open, data.mark_high, data.mark_low, data.mark_close,
            data.funding_rate, data.funding_mark, np.full(shape, self.maintenance_rate),
            np.full(shape, self.cost_rate), np.full(shape, self.liquidation_fee),
            self.allocation.max_gross, len(daily.dates))
        panel = Panel(daily.dates, daily.assets, fields, src, backend)
        return LoadedStudy(study, self, panel, spec,
                           Window(daily.dates, int(np.searchsorted(daily.dates, np.datetime64(study.start)))), daily.contract)

    def universe(self, panel):
        return panel['member'] & panel.backend.xp.isfinite(panel['close'])
