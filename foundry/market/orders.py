"""Compile factual crypto bars and explicit research rules for order execution."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from foundry.config import Execution
from foundry.market.execution import ffill
from foundry.types import BarExecution, ExecutionSchedule, HostArray


def crypto_bars(market: str, schedule: ExecutionSchedule, close_ns: NDArray[np.int64],
                bars: Mapping[str, HostArray], execution: Execution, fee: float,
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
    return BarExecution(schedule, close_ns, bars['open'], bars['high'], bars['low'], bars['close'],
        np.where(valid, bars['volume'], 0.), *marks, full(fee), full(fee), funding_rate, funding_mark,
        full(maintenance), full(liquidation_fee), valid, valid, valid,
        full(execution.quantity_step), full(execution.price_tick), market)
