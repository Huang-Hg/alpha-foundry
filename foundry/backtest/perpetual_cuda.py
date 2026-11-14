"""Warp-parallel perpetual execution with explicit numerical refinement.

FP32/paired simulation is available for numeric experiments. Admission retains
FP64 until a signed, leveraged screening error policy is calibrated.
"""
from dataclasses import dataclass, field
from pathlib import Path

import cupy as cp
import numpy as np

from foundry.backtest.perpetual import (ARRAY_FIELDS, PerpetualSpec,
                                      concatenate_results, simulation_result)
from foundry.contracts import Allocation
from foundry.portfolio.allocation import cap_book


@dataclass(frozen=True)
class DeviceSpec:
    host: PerpetualSpec
    arrays: tuple
    offsets: object
    funding_prefix: object
    reduced_range: bool
    packed: dict = field(default_factory=dict)
    backend: str = 'cuda'


class Executor:
    parallel_trials = True
    precisions = frozenset({'f64', 'mixed', 'paired'})
    screening_precisions = frozenset({'f64'})

    def __init__(self, return_atol, metric_atol):
        self._modules = {}
        self.return_atol, self.metric_atol = return_atol, metric_atol
        self._split = cp.ElementwiseKernel('float64 x', 'float32 hi, float32 lo',
            'hi=(float)x; lo=isfinite(x)?(float)(x-(double)hi):0.0f;', 'perp_split')

    def _module(self, assets):
        if assets not in self._modules:
            root = Path(__file__).parent
            source = (root / 'src' / 'perpetual.hpp').read_text(encoding='utf-8')
            source += '\n' + (root / 'kernels' / 'float_pair.cuh').read_text(encoding='utf-8')
            source += '\n' + (root / 'kernels' / 'perpetual.cu').read_text(encoding='utf-8')
            source += '\n' + (root / 'kernels' / 'perpetual_warp.cuh').read_text(encoding='utf-8')
            source += '\n' + (root / 'kernels' / 'funding_prefix.cu').read_text(encoding='utf-8')
            self._modules[assets] = cp.RawModule(code=f'#define ASSETS {assets}\n' + source,
                                                options=('--std=c++17', '--fmad=false'))
        return self._modules[assets]

    def device_spec(self, host):
        offsets = np.searchsorted(host.day_index, np.arange(host.days + 1)).astype(np.int64)
        assert np.all((np.diff(offsets) > 0) & (np.diff(offsets) <= 32)), 'hourly daily event grid'
        self._module(host.shape[1]).get_function('perpetual_warp_f64')
        # Conservative exponent domain, evaluated once per immutable spec.
        # Outside it the numerical policy selects the scalar FP64 reference.
        magnitudes = [np.abs(getattr(host, name)) for name in ARRAY_FIELDS[3:]]
        reduced_range = all(np.all((a == 0) | ((a >= 2.**-64) & (a <= 2.**32))) for a in magnitudes)
        arrays = tuple(cp.asarray(getattr(host, name)) for name in ARRAY_FIELDS)
        offsets = cp.asarray(offsets)
        funding_prefix = cp.empty_like(arrays[8])
        self._modules[host.shape[1]].get_function('funding_prefix')(
            (host.days, (host.shape[1] + 31) // 32), (32,),
            (arrays[9], arrays[8], offsets, funding_prefix, np.int32(host.shape[1])))
        return DeviceSpec(host, arrays, offsets, funding_prefix, reduced_range)

    def _pack(self, value, precision):
        value = cp.ascontiguousarray(value, dtype=cp.float64)
        if precision == 'f64':
            return value
        if precision == 'mixed':
            return value.astype(cp.float32)
        pair = cp.empty(value.shape + (2,), cp.float32)
        self._split(value, pair[..., 0], pair[..., 1])
        return pair

    def simulate(self, W, spec, start, ppy, precision):
        B, T, S = W.shape
        E = len(spec.host.event_times_ms)
        arrays = (*[cp.empty((B, T), cp.float64) for _ in range(5)],
                  cp.empty((B, T), cp.int64), cp.empty((B, E, S), cp.uint8),
                  cp.empty(B, cp.int64), cp.empty(B, cp.bool_),
                  cp.empty(B, cp.float64), cp.empty(B, cp.float64))
        if precision not in spec.packed:
            spec.packed[precision] = tuple(self._pack(a, precision) for a in (*spec.arrays[3:], spec.funding_prefix))
        kernel = self._modules[S].get_function('perpetual_warp_' + precision)
        uncertain = cp.zeros(B, cp.uint8) if precision == 'f64' else (
            cp.any((W != 0) & ((cp.abs(W) < 2.**-32) | (cp.abs(W) > 2.**32)), axis=(1, 2))
            | (not spec.reduced_range)).astype(cp.uint8)
        kernel((B,), (32,),
               (self._pack(W, precision), spec.arrays[0], spec.offsets, spec.arrays[2], *spec.packed[precision],
                np.int64(E), np.int64(T), np.float64(spec.host.leverage), *arrays, uncertain, np.float64(self.return_atol)))
        stats = cp.empty((B, 5), cp.float64)
        statistics = self._modules[S].get_function('perpetual_statistics')
        statistics(((B + 31) // 32,), (32,),
                   (arrays[0], np.int64(T), np.int64(B), np.int64(start), np.float64(ppy), stats))
        self._modules[S].get_function('perpetual')(((B + 31) // 32,), (32,),
            (cp.ascontiguousarray(W, dtype=cp.float64), *spec.arrays, np.int64(E), np.int64(T), np.int64(B),
             np.float64(spec.host.leverage), *arrays, uncertain, stats, np.float64(self.metric_atol)))
        statistics(((B + 31) // 32,), (32,),
                   (arrays[0], np.int64(T), np.int64(B), np.int64(start), np.float64(ppy), stats))
        return simulation_result(arrays, stats)

    def blend(self, W, U, owners, weights, spec, start, ppy, precision, batch):
        results = []
        allocation = Allocation(spec.host.leverage, True)
        for lo in range(0, len(weights), batch):
            books = U[None] + weights[lo:lo + batch, None, None] * W[owners[lo:lo + batch]]
            books = cap_book(books, allocation, cp)
            results.append(self.simulate(books, spec, start, ppy, precision))
        return concatenate_results(results, start, cp)

    def fingerprints(self, W):
        hashes = cp.empty(len(W), cp.uint64)
        fingerprint = self._module(W.shape[2]).get_function('perpetual_fingerprints')
        fingerprint((len(W),), (256,), (cp.ascontiguousarray(W, dtype=cp.float64), hashes,
                                       np.int64(W.shape[1] * W.shape[2])))
        return cp.asnumpy(hashes).tolist()

    def close(self):
        self._modules.clear()


def prepare(return_atol, metric_atol):
    return Executor(return_atol, metric_atol)
