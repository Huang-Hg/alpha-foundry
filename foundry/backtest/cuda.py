"""CUDA execution: warp-parallel symmetric books and block-parallel general execution rules.
Prices, holdings and trial grids stay on device; CPU execution is an independent reference.
"""
from __future__ import annotations

from functools import cached_property
from pathlib import Path

import cupy as cp
import numpy as np

from foundry.numeric.cuda_source import paired_source
from foundry.types import ExecutionSpec, FloatArray, IntArray, SimulationResult

_KERNELS = Path(__file__).parent / 'kernels'
MAX_ASSETS = 128
_FINGERPRINT_SOURCE = r'''
extern "C" __global__ void fingerprints(const double* w, unsigned long long* out, int n) {
    __shared__ unsigned long long hashes[256];
    unsigned long long h = 0;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        double v = w[(long long)blockIdx.x*n+j];
        unsigned long long x = v == 0 ? 0 : __double_as_longlong(v);
        x ^= (unsigned long long)(j+1)*0x9e3779b97f4a7c15ULL;
        x = (x^(x>>30))*0xbf58476d1ce4e5b9ULL;
        x = (x^(x>>27))*0x94d049bb133111ebULL;
        h ^= x^(x>>31);
    }
    hashes[threadIdx.x] = h;
    __syncthreads();
    for (int d=128; d; d>>=1) {
        if (threadIdx.x < d) hashes[threadIdx.x] ^= hashes[threadIdx.x+d];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = hashes[0];
}
'''

class _DeviceSpec:
    """Resident execution spec: warp kernel for small symmetric books, block kernel otherwise.

    GPU representations are cached lazily; paired values retain ~48 significant bits
    but have FP32's exponent range. Nonfinite recurrence states remain invalid.
    """

    def __init__(self, spec: ExecutionSpec, pack):
        self._pack = pack
        self.host = spec
        self.backend = 'cuda'
        self.general = not spec.symmetric or spec.open_raw.shape[1] > MAX_ASSETS
        self.flags = int(spec.cost_sell is not None) | int(spec.block is not None) << 1 | int(spec.carry is not None) << 2
        names = ('open_raw', 'prev_close', 'open_ff0', 'close_ff0', 'cost_buy')
        self.storage = [cp.asarray(getattr(spec, k), dtype=cp.float64) for k in names] + [cp.asarray(spec.live, dtype=cp.uint8)]
        self.schedule = (cp.asarray(spec.schedule.decision_index), cp.asarray(spec.schedule.measurement_index))

    @cached_property
    def extra_storage(self):
        shape = self.host.live.shape
        return [self.storage[4] if self.host.cost_sell is None else cp.asarray(self.host.cost_sell), cp.zeros(shape, cp.float64) if self.host.carry is None else cp.asarray(self.host.carry), cp.zeros(shape, cp.uint8) if self.host.block is None else cp.asarray(self.host.block)]

    @cached_property
    def extra_mixed(self):
        return [a.astype(cp.float32) for a in self.extra_storage[:2]] + [self.extra_storage[2]]

    @cached_property
    def extra_paired(self):
        return [self._pack(a) for a in self.extra_storage[:2]] + [self.extra_storage[2]]

    @cached_property
    def mixed(self):
        return [a.astype(cp.float32) for a in self.storage[:5]] + [self.storage[5]]

    @cached_property
    def paired(self):
        return [self._pack(a) for a in self.storage[:5]] + [self.storage[5]]

    def arrays(self, precision):
        return getattr(self, precision)

    def general_arrays(self, precision):
        base, extra = (self.arrays(precision), getattr(self, 'extra_' + precision))
        return (*base[:5], *extra[:2], base[5], extra[2], np.int32(self.flags))

class Executor:
    parallel_trials = True
    precisions = frozenset({'mixed', 'paired'})

    def __init__(self, gross_limit: float = 1.):
        self.gross_limit = gross_limit
        self._MOD = cp.RawModule(code=paired_source()+'\n'+'\n'.join(((_KERNELS / f).read_text(encoding='utf-8') for f in ('reductions.cuh', 'backtest.cu', 'statistics.cuh'))), options=('--std=c++14',))
        self._SIMULATE = {p: self._MOD.get_function(f'k_simulate_{p}') for p in ('mixed', 'paired')}
        self._BLEND = {p: self._MOD.get_function(f'k_blend_{p}') for p in ('mixed', 'paired')}
        self._STATS = self._MOD.get_function('k_statistics_paired')
        self._SPLIT = cp.ElementwiseKernel('float64 v', 'float32 hi, float32 lo', 'hi = (float)v; lo = isfinite(v) ? (float)(v - (double)hi) : 0.0f;', 'bt_split')
        self._general_modules = {}
        self._fingerprint = cp.RawKernel(_FINGERPRINT_SOURCE, 'fingerprints')

    def fingerprints(self, W: FloatArray) -> list[int]:
        hashes = cp.empty(len(W), dtype=cp.uint64)
        self._fingerprint((len(W),), (256,), (W, hashes, np.int32(W.shape[1]*W.shape[2])))
        return cp.asnumpy(hashes).tolist()

    def _general_module(self, slots):
        if slots not in self._general_modules:
            source = paired_source()+'\n'+'\n'.join(((_KERNELS / f).read_text(encoding='utf-8') for f in ('reductions.cuh', 'backtest.cu', 'statistics.cuh', 'general.cu')))
            self._general_modules[slots] = cp.RawModule(code=f'#define GEN_SLOTS {slots}\n' + source, options=('--std=c++14',))
        return self._general_modules[slots]

    def paired(self, x):
        """Pack contiguous (hi, lo) FP32 pairs from the interchange array, once outside the serial recurrence."""
        out = cp.empty(x.shape + (2,), dtype=cp.float32)
        self._SPLIT(x, out[..., 0], out[..., 1])
        return out

    def _pack(self, x, precision):
        if precision == 'paired':
            return self.paired(x)
        return x.astype(cp.float32)

    def _statistics(self, rr, stats, start, ppy):
        B, T = rr.shape
        self._STATS((B,), (128,), (rr, np.int32(T), np.int32(start), np.float64(ppy), stats))

    def simulate(self, W: FloatArray, spec: _DeviceSpec, start: int, ppy: float, precision: str) -> SimulationResult:
        """Targets (B,D,S) -> measurement returns (B,V) and statistics over [start:].

        One launch walks E execution events in order and gathers only scheduled
        target rows. Targets are never expanded to the event axis.
        """
        B, D, S = W.shape
        E, V = len(spec.host.schedule.event_ns), spec.host.schedule.measurement_count
        rr, stats = (cp.empty((B, V), dtype=cp.float64), cp.empty((B, 5), dtype=cp.float64))
        if spec.general:
            kernel = self._general_module((S + 127) // 128).get_function(f'k_general_{precision}')
            kernel((B,), (128,), (self._pack(W, precision), *(np.uint64(0),) * 5, *spec.general_arrays(precision), *spec.schedule, np.int32(D), np.int32(E), np.int32(V), np.int32(S), np.int64(start), np.float64(ppy), rr, stats))
            self._statistics(rr, stats, start, ppy)
            return SimulationResult(rr, stats, None)
        self._SIMULATE[precision](((B + 3) // 4,), (128,), (self._pack(W, precision), *spec.arrays(precision), *spec.schedule, np.int32(D), np.int32(E), np.int32(V), np.int32(S), np.int32(B), np.int64(start), np.float64(ppy), rr, stats))
        self._statistics(rr, stats, start, ppy)
        return SimulationResult(rr, stats, None)

    def blend(self, W: FloatArray, U: FloatArray, owners: IntArray, weights: FloatArray,
              spec: _DeviceSpec, start: int, ppy: float, precision: str, batch: int) -> SimulationResult:
        """Trial n = (U + weights[n] * W[owners[n]]) / max(1, exposure/gross_limit), fused into the kernel (U = the portfolio's raw
    additive sum). -> (returns over [start:], stats)."""
        eref = cp.ascontiguousarray(U.sum(1), dtype=cp.float64)
        eown = cp.ascontiguousarray(W.sum(2), dtype=cp.float64)
        if self.gross_limit != 1.:
            eref /= self.gross_limit
            eown /= self.gross_limit
        D, S = W.shape[1:]
        E, V = len(spec.host.schedule.event_ns), spec.host.schedule.measurement_count
        N = len(weights)
        W, U = (self._pack(W, precision), self._pack(U, precision))
        if precision == 'paired':
            weights, eref, eown = (self.paired(weights), self.paired(eref), self.paired(eown))
        rr, stats = (cp.empty((N, V), dtype=cp.float64), cp.empty((N, 5), dtype=cp.float64))
        for lo in range(0, N, batch):
            n = min(batch, N - lo)
            if spec.general:
                kernel = self._general_module((S + 127) // 128).get_function(f'k_general_blend_{precision}')
                kernel((n,), (128,), (W, U, owners[lo:lo + n], weights[lo:lo + n], eref, eown, *spec.general_arrays(precision), *spec.schedule, np.int32(D), np.int32(E), np.int32(V), np.int32(S), np.int64(start), np.float64(ppy), rr[lo:lo + n], stats[lo:lo + n]))
                continue
            self._BLEND[precision](((n + 3) // 4,), (128,), (W, U, owners[lo:lo + n], weights[lo:lo + n], eref, eown, *spec.arrays(precision), *spec.schedule, np.int32(D), np.int32(E), np.int32(V), np.int32(S), np.int32(n), np.int64(start), np.float64(ppy), rr[lo:lo + n], stats[lo:lo + n]))
        self._statistics(rr, stats, start, ppy)
        return SimulationResult(rr[:, start:], stats, None)

    def device_spec(self, host: ExecutionSpec) -> _DeviceSpec:
        return _DeviceSpec(host, self.paired)

    def close(self) -> None:
        self._general_modules.clear()
        self.__dict__.clear()

def prepare(gross_limit: float = 1.) -> Executor:
    return Executor(gross_limit)
