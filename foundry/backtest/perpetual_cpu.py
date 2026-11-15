"""Native CPU reference for the explicit isolated linear-perpetual model."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from foundry.numeric.allocation import cap_book

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
    parallel_trials = False
    precisions = frozenset({'native'})

    def device_spec(self, host: PerpetualSpec) -> HostSpec:
        return HostSpec(host)

    def simulate(self, W: FloatArray, spec: HostSpec, start: int, ppy: float, precision: str) -> SimulationResult:
        assert precision == 'native'
        arrays = _bt.perpetual(np.ascontiguousarray(W, dtype=np.float64),
                               tuple(getattr(spec.host, name) for name in PERPETUAL_ARRAY_FIELDS),
                               spec.host.leverage)
        return simulation_result(arrays, host_statistics(arrays[0], start, ppy))

    def blend(self, W: FloatArray, U: FloatArray, owners: IntArray, weights: FloatArray,
              spec: HostSpec, start: int, ppy: float, precision: str, batch: int) -> SimulationResult:
        results = []
        allocation = Allocation(spec.host.leverage, True)
        for lo in range(0, len(weights), batch):
            books = U[None] + weights[lo:lo + batch, None, None] * W[owners[lo:lo + batch]]
            books = cap_book(books, allocation, np)
            results.append(self.simulate(books, spec, start, ppy, precision))
        return concatenate_results(results, start, np)

    def fingerprints(self, W: FloatArray) -> list[int]:
        return book_fingerprints(W, np, np.asarray)

    def close(self) -> None:
        pass


def prepare() -> Executor:
    return Executor()
