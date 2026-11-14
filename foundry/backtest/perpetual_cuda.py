"""CUDA FP64 isolated perpetual execution; resources start at explicit prepare."""
from dataclasses import dataclass
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
    kernel: object
    backend: str = 'cuda'


class Executor:
    precisions = frozenset({'f64'})

    def __init__(self):
        self._modules = {}

    def _module(self, assets):
        if assets not in self._modules:
            root = Path(__file__).parent
            source = (root / 'src' / 'perpetual.hpp').read_text(encoding='utf-8')
            source += '\n' + (root / 'kernels' / 'perpetual.cu').read_text(encoding='utf-8')
            self._modules[assets] = cp.RawModule(code=f'#define ASSETS {assets}\n' + source,
                                                options=('--std=c++17', '--fmad=false'))
        return self._modules[assets]

    def device_spec(self, host):
        kernel = self._module(host.shape[1]).get_function('perpetual')
        return DeviceSpec(host, tuple(cp.asarray(getattr(host, name)) for name in ARRAY_FIELDS), kernel)

    def simulate(self, W, spec, start, ppy, precision):
        assert precision == 'f64', 'perpetual execution requires the FP64 policy'
        B, T, S = W.shape
        E = len(spec.host.event_times_ms)
        arrays = (*[cp.zeros((B, T), cp.float64) for _ in range(5)],
                  cp.zeros((B, T), cp.int64), cp.zeros((B, E, S), cp.uint8),
                  cp.empty(B, cp.int64), cp.empty(B, cp.bool_),
                  cp.empty(B, cp.float64), cp.empty(B, cp.float64))
        spec.kernel(((B + 31) // 32,), (32,),
                    (cp.ascontiguousarray(W, dtype=cp.float64), *spec.arrays,
                     np.int64(E), np.int64(T), np.int64(B), np.float64(spec.host.leverage), *arrays))
        stats = cp.empty((B, 5), cp.float64)
        statistics = self._modules[S].get_function('perpetual_statistics')
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


def prepare():
    return Executor()
