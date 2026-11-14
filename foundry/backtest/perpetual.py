"""USDT linear isolated-margin research execution, not an exchange replica.

Daily close targets rebalance at the next UTC day's first hourly event. Each
asset receives isolated collateral abs(target)/leverage times post-trade NAV;
unallocated cash remains separate and collateral is redistributed only at the
scheduled daily rebalance. Funding settles before rebalance. Mark OHLC adverse
touches trigger the stated bar liquidation model, with no intrabar replenishment.
Liquidation fees record actual collateral deductions; isolated_loss_cap records
the model's compensation for a negative isolated bucket when losses are capped.
Fee and maintenance schedules, and funding settlement marks, are explicit input
assumptions whose historical limitations belong in the study data contract.
"""
from dataclasses import dataclass, fields, replace
import hashlib

import numpy as np

from foundry.contracts import ExecutionRiskBatch, SimulationResult
from foundry.core import metrics


MODEL = 'linear_usdt_isolated_daily_collateral_hourly_mark_touch_v1'
EVENT_FIELDS = ('event_times_ms', 'day_index', 'rebalance')
PANEL_FIELDS = ('trade_open', 'mark_open', 'mark_high', 'mark_low', 'mark_close',
                'funding_rate', 'funding_mark', 'maintenance_rate', 'fee_rate', 'liquidation_fee')
ARRAY_FIELDS = EVENT_FIELDS + PANEL_FIELDS


@dataclass(frozen=True)
class PerpetualSpec:
    """Owned read-only host arrays for T daily targets and E hourly events.

    event_times_ms/day_index are int64 (E,), UTC bar-start milliseconds and
    indices in [0, days); rebalance is bool (E,). Price and rate arrays are
    float64 (E,S): prices use USDT per asset unit, funding_rate is the signed
    settlement fraction, and maintenance/fee/liquidation_fee are fractions of
    mark/traded notional. The loader validates a complete, ordered event grid.
    Intrabar first-liquidation timestamps identify the bar, not a known tick.
    """
    event_times_ms: np.ndarray
    day_index: np.ndarray
    rebalance: np.ndarray
    trade_open: np.ndarray
    mark_open: np.ndarray
    mark_high: np.ndarray
    mark_low: np.ndarray
    mark_close: np.ndarray
    funding_rate: np.ndarray
    funding_mark: np.ndarray
    maintenance_rate: np.ndarray
    fee_rate: np.ndarray
    liquidation_fee: np.ndarray
    leverage: float
    days: int

    def __post_init__(self):
        for name in ARRAY_FIELDS:
            dtype = np.int64 if name in EVENT_FIELDS[:2] else np.bool_ if name == 'rebalance' else np.float64
            owned = np.array(getattr(self, name), dtype=dtype, order='C', copy=True)
            owned.flags.writeable = False
            object.__setattr__(self, name, owned)
        ratio = self.trade_open / self.mark_open
        assert np.all(self.leverage * (np.abs(1 - ratio).max(1)
                                      + (self.fee_rate * ratio).max(1)) < 1), 'non-monotone rebalance root'

    @property
    def shape(self):
        return self.days, self.trade_open.shape[1]

    @property
    def snapshot(self):
        digest = hashlib.sha256(MODEL.encode())
        for definition in fields(self):
            value = getattr(self, definition.name)
            digest.update(definition.name.encode())
            if definition.name in ARRAY_FIELDS:
                digest.update(str((value.shape, value.dtype.str)).encode())
                digest.update(value.tobytes())
            else:
                digest.update(str(value).encode())
        return digest.hexdigest()

    def rows(self, end):
        stop = int(np.searchsorted(self.day_index, end))
        return replace(self, **{name: getattr(self, name)[:stop] for name in ARRAY_FIELDS}, days=end)


def host_statistics(returns, start, ppy):
    """CPU reference statistics; the CUDA executor owns its device reduction."""
    r = returns[:, start:]
    return np.column_stack((metrics.sharpe(r, ppy, np), r.mean(1) * ppy,
                            r.std(1) * ppy ** .5, metrics.max_drawdown(r, np),
                            np.prod(1 + r, axis=1) - 1))


def simulation_result(arrays, statistics):
    """All risk buffers retain the complete input grid on their execution device."""
    return SimulationResult(arrays[0], statistics, ExecutionRiskBatch(*arrays[1:], MODEL))


def concatenate_results(results, start, xp):
    risk = ExecutionRiskBatch(*(xp.concatenate([getattr(r.risk, f.name) for r in results])
                                for f in fields(ExecutionRiskBatch)[:-1]), MODEL)
    return SimulationResult(xp.concatenate([r.returns[:, start:] for r in results]),
                            xp.concatenate([r.statistics for r in results]), risk)


def book_fingerprints(W, xp, to_host):
    values = xp.ascontiguousarray(xp.where(W == 0, 0., W), dtype=xp.float64).reshape(len(W), -1).view(xp.uint64)
    index = xp.arange(1, values.shape[1] + 1, dtype=xp.uint64)
    x = values ^ (index * xp.uint64(0x9e3779b97f4a7c15))
    x = (x ^ (x >> 30)) * xp.uint64(0xbf58476d1ce4e5b9)
    x = (x ^ (x >> 27)) * xp.uint64(0x94d049bb133111eb)
    return to_host(xp.bitwise_xor.reduce(x ^ (x >> 31), axis=1)).tolist()
