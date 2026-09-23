"""US equities, daily (S&P 500 constituents; point-in-time membership is applied when the parquet is built, so a
symbol has rows only while it is a member).

Data: data/us_equity_daily/panel.parquet (date, symbol, open, high, low, close, volume, adj_close, adj_factor).
Prices are total-return adjusted (raw x adj_factor). Execution: a symmetric fee of cost_rate x open per share,
borrow cost on short value; geometric Sharpe over 252 days. Universe: the day's members with a price.
"""
from __future__ import annotations

import numpy as np

from foundry.backtest.spec import ExecutionSpec
from foundry.config import UsEquityConfig, path
from foundry.core.panel import Field, Panel
from foundry.market.base import Market, Window, long_table, ohlcv_fields, trend_fields


class UsEquity(Market):
    name = 'us_equity'
    ppy = 252.0
    geometric = True
    categories = {'pricevolume': 'scale-free daily price/volume primitives',
                  'trend': 'distance to moving average, trailing return'}

    def __init__(self):
        self.cfg = UsEquityConfig()

    def load(self, backend):
        dates, symbols, g, start = long_table(path(self.cfg.data_dir) / 'panel.parquet', self.cfg.start, self.cfg.end,
                                              ('open', 'high', 'low', 'close', 'volume', 'adj_factor'))
        src = {k: g[k] * g['adj_factor'] for k in ('open', 'high', 'low', 'close')}
        src['volume'] = g['volume']
        fields = [Field(k, 'price', formula=False) for k in src] + ohlcv_fields() + trend_fields()
        carry = None if self.cfg.daily_borrow == 0 else np.full(src['close'].shape, self.cfg.daily_borrow)
        spec = ExecutionSpec.build(src['open'], src['close'], self.cfg.cost_rate * src['open'], carry=carry)
        return Panel(dates, symbols, fields, src, backend), spec, Window(dates, start)

    def universe(self, panel):
        return panel.backend.xp.isfinite(panel['close'])


MARKET = UsEquity()
