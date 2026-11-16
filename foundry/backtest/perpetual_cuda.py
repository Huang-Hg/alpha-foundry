"""Warp-parallel perpetual accounts with a fixed study policy and paired statistics."""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import TYPE_CHECKING

import cupy as cp
import numpy as np
from foundry.numeric.allocation import cap_book

from foundry.backtest.perpetual import concatenate_results, simulation_result
from foundry.numeric.cuda_source import paired_source
from foundry.types import (
    PERPETUAL_ARRAY_FIELDS,
    Allocation,
    Array,
    FloatArray,
    IntArray,
    PerpetualSpec,
    SimulationResult,
)

if TYPE_CHECKING:
    from foundry.backtest.perpetual_cpu import Executor as HostExecutor


@dataclass(frozen=True)
class DeviceSpec:
    host: PerpetualSpec
    arrays: tuple[Array, ...]
    offsets: IntArray
    backend: str = 'cuda'


class Executor:
    parallel_trials = True
    precisions = frozenset({'native'})

    def __init__(self, host_executor: HostExecutor, host_cells: int):
        self._modules = {}
        self.host_executor, self.host_cells = host_executor, host_cells
        self.calls = {'cpu': 0, 'cuda': 0}

    def _module(self, assets):
        if assets not in self._modules:
            root = Path(__file__).parent
            source = (root / 'src' / 'perpetual_output.hpp').read_text(encoding='utf-8')
            source += '\n' + paired_source() + '\n' + '\n'.join((root / 'kernels' / name).read_text(encoding='utf-8')
                for name in ('reductions.cuh', 'statistics.cuh',
                             'perpetual_fingerprints.cu', 'perpetual_asset.cu'))
            self._modules[assets] = cp.RawModule(code=f'#define ASSETS {assets}\n' + source,
                                                options=('--std=c++17', '--fmad=false'))
        return self._modules[assets]

    def device_spec(self, host: PerpetualSpec) -> DeviceSpec:
        offsets = np.searchsorted(host.day_index, np.arange(host.days + 1)).astype(np.int64)
        assert np.all(np.diff(offsets) > 0), 'each daily interval has execution events'
        self._module(host.shape[1]).get_function('perpetual_asset_native')
        # Contracts may quote prices in very small or large units. Normalize by
        # each asset's first available mark using an exact power of two; all
        # cashflows and maintenance inequalities are invariant to this unit.
        exponents = np.frexp(host.mark_open[0])[1]
        price_fields = {'trade_open', 'mark_open', 'mark_high', 'mark_low', 'mark_close', 'funding_mark'}
        arrays = tuple(cp.asarray(np.ldexp(getattr(host, name), -exponents)
                       if name in price_fields else getattr(host, name)) for name in PERPETUAL_ARRAY_FIELDS)
        offsets = cp.asarray(offsets)
        return DeviceSpec(host, arrays, offsets)

    def simulate(self, W: FloatArray, spec: DeviceSpec, start: int, ppy: float, precision: str) -> SimulationResult:
        assert precision == 'native'
        if W.size <= self.host_cells:
            self.calls['cpu'] += 1
            result = self.host_executor.simulate(cp.asnumpy(W), self.host_executor.device_spec(spec.host), start, ppy, 'native')
            risk = replace(result.risk, **{f.name: cp.asarray(getattr(result.risk, f.name))
                           for f in fields(result.risk) if f.name != 'model'})
            return replace(result, returns=cp.asarray(result.returns), statistics=cp.asarray(result.statistics), risk=risk)
        self.calls['cuda'] += 1
        return self.simulate_device(W, spec, start, ppy)

    def simulate_device(self, W: FloatArray, spec: DeviceSpec, start: int, ppy: float) -> SimulationResult:
        """One CUDA execution; also the explicit kernel verification boundary."""
        B, T, S = W.shape
        E = len(spec.host.event_times_ms)
        arrays = (*[cp.empty((B, T), cp.float64) for _ in range(5)],
                  cp.empty((B, T), cp.int64), cp.empty((B, E, S), cp.uint8),
                  cp.empty(B, cp.int64), cp.empty(B, cp.bool_),
                  cp.empty(B, cp.float64), cp.empty(B, cp.float64))
        kernel = self._modules[S].get_function('perpetual_asset_native')
        kernel((B,), (32,),
               (cp.ascontiguousarray(W), spec.arrays[0], spec.offsets, spec.arrays[2], *spec.arrays[3:],
                np.int64(E), np.int64(T), np.float64(spec.host.leverage), *arrays))
        stats = cp.empty((B, 5), cp.float64)
        self._modules[S].get_function('k_statistics_paired')((B,), (128,),
            (arrays[0], np.int32(T), np.int32(start), np.float64(ppy), stats))
        return simulation_result(arrays, stats)

    def blend(self, W: FloatArray, U: FloatArray, owners: IntArray, weights: FloatArray,
              spec: DeviceSpec, start: int, ppy: float, precision: str, batch: int) -> SimulationResult:
        results = []
        allocation = Allocation(spec.host.gross_limit, True)
        for lo in range(0, len(weights), batch):
            books = U[None] + weights[lo:lo + batch, None, None] * W[owners[lo:lo + batch]]
            books = cap_book(books, allocation, cp)
            results.append(self.simulate(books, spec, start, ppy, precision))
        return concatenate_results(results, start, cp)

    def fingerprints(self, W: FloatArray) -> list[int]:
        hashes = cp.empty(len(W), cp.uint64)
        fingerprint = self._module(W.shape[2]).get_function('perpetual_fingerprints')
        fingerprint((len(W),), (256,), (cp.ascontiguousarray(W, dtype=cp.float64), hashes,
                                       np.int64(W.shape[1] * W.shape[2])))
        return cp.asnumpy(hashes).tolist()

    def close(self) -> None:
        self._modules.clear()
        self.host_executor.close()


def prepare(host_executor: HostExecutor, host_cells: int) -> Executor:
    return Executor(host_executor, host_cells)
