"""CUDA execution: batched books through the symmetric-spec kernel (kernels/backtest.cu), plus the fused long-only
book. Returns stay on the device; statistics over the research window come back fused with the returns.
"""
from __future__ import annotations

from pathlib import Path

import cupy as cp
import numpy as np

from foundry.backtest.spec import ExecutionSpec

_MOD = cp.RawModule(code=(Path(__file__).parent / 'kernels' / 'backtest.cu').read_text(encoding='utf-8'),
                    options=('--std=c++14',))
_SIMULATE = {'f64': _MOD.get_function('k_simulate_f64'), 'mixed': _MOD.get_function('k_simulate_mixed')}
_BLEND = {'f64': _MOD.get_function('k_blend_f64'), 'mixed': _MOD.get_function('k_blend_mixed')}
MAX_ASSETS = 128                              # 32 lanes x 4 assets in registers

# Long-only book: exposure clip[0,1] in the day's universe (NaN = cash), one equal slot per asset of the universe.
book = cp.ElementwiseKernel(
    'T v, bool universe, T slots', 'T o',
    'o = (universe && isfinite(v)) ? (((v < (T)0) ? (T)0 : ((v > (T)1) ? (T)1 : v)) / slots) : (T)0;', 'fx_book')


class DeviceSpec:
    """An ExecutionSpec resident on the device, in float64 and in the float32 of the mixed-precision path."""

    def __init__(self, spec: ExecutionSpec):
        assert spec.symmetric and spec.open_raw.shape[1] <= MAX_ASSETS, 'the CUDA kernel runs symmetric specs of <= 128 assets'
        names = ('open_raw', 'prev_close', 'open_ff0', 'close_ff0', 'cost_buy')
        self.f64 = [cp.asarray(getattr(spec, k), dtype=cp.float64) for k in names] + [cp.asarray(spec.live, dtype=cp.uint8)]
        self.mixed = [a.astype(cp.float32) for a in self.f64[:5]] + [self.f64[5]]

    def arrays(self, precision):
        return self.f64 if precision == 'f64' else self.mixed


def simulate(W, spec: DeviceSpec, start: int, ppy: float, precision='f64'):
    """Books W (B,T,S) -> (net daily returns (B,T) f64, stats (B,5) = sharpe, mean, vol, max drawdown, total return
    over rows [start:]). One launch walks the days in order, so it has a latency floor independent of B
    (V100S, T=4262 x S=73: ~31 ms for 1..256 books): batch books into as few launches as possible."""
    B, T, S = W.shape
    rr, stats = cp.empty((B, T), dtype=cp.float64), cp.empty((B, 5), dtype=cp.float64)
    _SIMULATE[precision](((B + 3) // 4,), (128,),
                         (W if precision == 'f64' else W.astype(cp.float32), *spec.arrays(precision),
                          np.int32(T), np.int32(S), np.int32(B), np.int64(start), np.float64(ppy), rr, stats))
    return rr, stats


def blend(W, U, owners, weights, spec: DeviceSpec, start: int, ppy: float, precision, batch: int):
    """Trial n = (U + weights[n] * W[owners[n]]) / max(1, exposure), fused into the kernel (U = the portfolio's raw
    additive sum). -> (returns over [start:], stats)."""
    eref = cp.ascontiguousarray(U.sum(1), dtype=cp.float64)
    eown = cp.ascontiguousarray(W.sum(2), dtype=cp.float64)
    if precision == 'mixed':
        W, U = W.astype(cp.float32), U.astype(cp.float32)
    T, S = W.shape[1:]
    N = len(weights)
    rr, stats = cp.empty((N, T), dtype=cp.float64), cp.empty((N, 5), dtype=cp.float64)
    for lo in range(0, N, batch):
        n = min(batch, N - lo)
        _BLEND[precision](((n + 3) // 4,), (128,),
                          (W, U, owners[lo:lo + n], weights[lo:lo + n], eref, eown, *spec.arrays(precision),
                           np.int32(T), np.int32(S), np.int32(n), np.int64(start), np.float64(ppy), rr[lo:lo + n], stats[lo:lo + n]))
    return rr[:, start:], stats
