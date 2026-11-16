"""USDT linear isolated-margin research execution, not an exchange replica.

Targets rebalance at their compiled next-open execution event. Each
asset receives isolated collateral abs(target)/leverage times post-trade NAV;
unallocated cash remains separate and collateral is redistributed only at the
scheduled target update. Funding settles before rebalance. Mark OHLC adverse
touches trigger the stated bar liquidation model, with no intrabar replenishment.
Liquidation fees record actual collateral deductions; isolated_loss_cap records
the model's compensation for a negative isolated bucket when losses are capped.
Fee and maintenance schedules, and funding settlement marks, are explicit input
assumptions whose historical limitations belong in the study data contract.
"""

from dataclasses import fields
import numpy as np

from foundry.types import ExecutionRiskBatch, SimulationResult, PERPETUAL_MODEL
from foundry.numeric import metrics






def host_statistics(returns, start, ppy):
    """CPU reference statistics; the CUDA executor owns its device reduction."""
    r = returns[:, start:]
    return np.column_stack((metrics.sharpe(r, ppy, np), r.mean(1) * ppy,
                            r.std(1) * ppy ** .5, metrics.max_drawdown(r, np),
                            np.prod(1 + r, axis=1) - 1))


def simulation_result(arrays, statistics):
    """All risk buffers retain the complete input grid on their execution device."""
    return SimulationResult(arrays[0], statistics, ExecutionRiskBatch(*arrays[1:], PERPETUAL_MODEL))


def concatenate_results(results, start, xp):
    risk = ExecutionRiskBatch(*(xp.concatenate([getattr(r.risk, f.name) for r in results])
                                for f in fields(ExecutionRiskBatch)[:-1]), PERPETUAL_MODEL)
    return SimulationResult(xp.concatenate([r.returns[:, start:] for r in results]),
                            xp.concatenate([r.statistics for r in results]), risk)


def book_fingerprints(W, xp, to_host):
    values = xp.ascontiguousarray(xp.where(W == 0, 0., W), dtype=xp.float64).reshape(len(W), -1).view(xp.uint64)
    index = xp.arange(1, values.shape[1] + 1, dtype=xp.uint64)
    x = values ^ (index * xp.uint64(0x9e3779b97f4a7c15))
    x = (x ^ (x >> 30)) * xp.uint64(0xbf58476d1ce4e5b9)
    x = (x ^ (x >> 27)) * xp.uint64(0x94d049bb133111eb)
    return to_host(xp.bitwise_xor.reduce(x ^ (x >> 31), axis=1)).tolist()
