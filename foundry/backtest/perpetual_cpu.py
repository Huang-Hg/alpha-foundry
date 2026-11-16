"""Native CPU reference for the explicit isolated linear-perpetual model."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from foundry.numeric.allocation import cap_book
from foundry.numeric.parallel import HostWorkers

from foundry.backtest import _bt
from foundry.backtest.perpetual import (
    book_fingerprints,
    concatenate_results,
    host_statistics,
    simulation_result,
)
from foundry.types import (
    PERPETUAL_ARRAY_FIELDS,
    Allocation,
    FloatArray,
    IntArray,
    PerpetualSpec,
    SimulationResult,
)


@dataclass(frozen=True)
class HostSpec:
    host: PerpetualSpec
    backend: str = 'cpu'


class Executor:
    parallel_trials = True
    precisions = frozenset({'native'})

    def __init__(self, workers: HostWorkers | None = None):
        self.map = map if workers is None else workers.map

    def device_spec(self, host: PerpetualSpec) -> HostSpec:
        return HostSpec(host)

    def simulate(self, W: FloatArray, spec: HostSpec, start: int, ppy: float, precision: str) -> SimulationResult:
        assert precision == 'native'
        inputs = tuple(getattr(spec.host, name) for name in PERPETUAL_ARRAY_FIELDS)
        def account(w: FloatArray) -> tuple[FloatArray, ...]:
            return _bt.perpetual(np.ascontiguousarray(w[None], dtype=np.float64), inputs, spec.host.leverage)
        arrays = tuple(np.concatenate(parts) for parts in zip(*self.map(account, W)))
        return simulation_result(arrays, host_statistics(arrays[0], start, ppy))

    def blend(self, W: FloatArray, U: FloatArray, owners: IntArray, weights: FloatArray,
              spec: HostSpec, start: int, ppy: float, precision: str, batch: int) -> SimulationResult:
        allocation = Allocation(spec.host.gross_limit, True)
        inputs = tuple(getattr(spec.host, name) for name in PERPETUAL_ARRAY_FIELDS)
        def trial(i: int) -> SimulationResult:
            books = U[None] + weights[i] * W[owners[i]][None]
            books = cap_book(books, allocation, np)
            arrays = _bt.perpetual(books, inputs, spec.host.leverage)
            return simulation_result(arrays, host_statistics(arrays[0], start, ppy))
        return concatenate_results(list(self.map(trial, range(len(weights)))), start, np)

    def fingerprints(self, W: FloatArray) -> list[int]:
        return book_fingerprints(W, np, np.asarray)

    def close(self) -> None:
        pass


def prepare(workers: HostWorkers | None = None) -> Executor:
    return Executor(workers)
