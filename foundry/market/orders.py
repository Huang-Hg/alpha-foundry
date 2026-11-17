"""Compile factual market bars and explicit rules for order execution."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from foundry.config import Execution
from foundry.market.execution import ffill
from foundry.types import BarExecution, ExecutionSchedule, HostArray


def crypto_bars(market: str, schedule: ExecutionSchedule, close_ns: NDArray[np.int64],
                bars: Mapping[str, HostArray], execution: Execution, maker_fee: float, taker_fee: float,
                maintenance: float = 0., liquidation_fee: float = 0.) -> BarExecution:
    """Missing spot quotes carry only valuation; they never create a fill range.

    Zero lot/tick parameters explicitly model continuous research units. These
    settings are not an inferred exchange trading-rule history.
    """
    shape = bars['open'].shape
    prices = ('open', 'high', 'low', 'close')
    valid = np.logical_and.reduce([np.isfinite(bars[key]) & (bars[key] > 0) for key in prices])
    valid &= np.isfinite(bars['volume'])
    full = lambda value: np.full(shape, value, dtype=np.float64)
    if market == 'crypto_spot':
        last = ffill(np.where(valid, bars['close'], np.nan))
        last = np.where(np.isfinite(last), last, 0.)
        marks = [np.where(valid, bars[key], last) for key in ('open', 'high', 'low', 'close')]
        funding_rate, funding_mark = full(0.), last
    else:
        marks = [bars['mark_' + key] for key in ('open', 'high', 'low', 'close')]
        funding_rate, funding_mark = bars['funding_rate'], bars['funding_mark']
    return BarExecution(schedule=schedule, event_close_ns=close_ns,
        trade_open=bars['open'], trade_high=bars['high'], trade_low=bars['low'], trade_close=bars['close'],
        volume=np.where(valid, bars['volume'], 0.),
        mark_open=marks[0], mark_high=marks[1], mark_low=marks[2], mark_close=marks[3],
        fee_buy=full(0.), fee_sell=full(0.), maker_fee=full(maker_fee), taker_fee=full(taker_fee),
        funding_rate=funding_rate, funding_mark=funding_mark,
        maintenance_rate=full(maintenance), liquidation_fee=full(liquidation_fee),
        buy_allowed=valid, sell_allowed=valid, quote_valid=valid,
        qty_step=full(execution.quantity_step), fill_step=full(execution.quantity_step), price_tick=full(execution.price_tick),
        split_factor=full(1.), cash_dividend=full(0.), min_quantity=full(0.),
        limit_up=full(0.), limit_down=full(0.),
        session_start=np.zeros(shape, dtype=bool), session_end=np.zeros(shape, dtype=bool),
        inventory_t1=np.zeros(shape, dtype=bool), cash_t1=np.zeros(shape, dtype=bool),
        odd_lot_sell=np.zeros(shape, dtype=bool), market=market)


def equity_bars(market: str, schedule: ExecutionSchedule, close_ns: NDArray[np.int64],
                bars: Mapping[str, HostArray]) -> BarExecution:
    """Compile explicit cash-equity facts without inferring rules from symbols."""
    valid = np.logical_and.reduce([np.isfinite(bars[k]) & (bars[k] > 0)
                                   for k in ('open', 'high', 'low', 'close')])
    shape = valid.shape
    # Carry only valuation through quote gaps. Corporate actions also transform
    # the carried mark, so suspended holdings retain their correct cash value.
    marks = {k: np.zeros(shape, dtype=np.float64) for k in ('open', 'high', 'low', 'close')}
    last = np.zeros(shape[1], dtype=np.float64)
    for event in range(shape[0]):
        last = last / bars['split_factor'][event]
        for k in marks:
            marks[k][event] = np.where(valid[event], bars[k][event], last)
        last = marks['close'][event]
    zero = np.zeros(shape, dtype=np.float64)
    return BarExecution(schedule=schedule, event_close_ns=close_ns,
        trade_open=bars['open'], trade_high=bars['high'], trade_low=bars['low'], trade_close=bars['close'],
        volume=np.where(valid, bars['volume'], 0.),
        mark_open=marks['open'], mark_high=marks['high'], mark_low=marks['low'], mark_close=marks['close'],
        fee_buy=bars['fee_buy'], fee_sell=bars['fee_sell'],
        maker_fee=bars['maker_fee'], taker_fee=bars['taker_fee'],
        funding_rate=zero, funding_mark=marks['close'], maintenance_rate=zero, liquidation_fee=zero,
        buy_allowed=valid & ~bars['suspended'] & bars['can_buy'],
        sell_allowed=valid & ~bars['suspended'] & bars['can_sell'], quote_valid=valid,
        qty_step=bars['quantity_step'], fill_step=bars['fill_step'], price_tick=bars['price_tick'],
        split_factor=bars['split_factor'], cash_dividend=bars['cash_dividend'],
        min_quantity=bars['min_quantity'], limit_up=bars['limit_up'], limit_down=bars['limit_down'],
        session_start=bars['session_start'], session_end=bars['session_end'],
        inventory_t1=bars['inventory_t1'], cash_t1=bars['cash_t1'], odd_lot_sell=bars['odd_lot_sell'],
        market=market)
