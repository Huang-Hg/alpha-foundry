"""Native CPU reference for the explicit isolated linear-perpetual model."""
from dataclasses import dataclass

import numpy as np

from foundry.backtest import _bt
from foundry.backtest.perpetual import (ARRAY_FIELDS, PerpetualSpec, book_fingerprints, host_statistics,
                                      concatenate_results, simulation_result)
from foundry.contracts import Allocation
from foundry.portfolio.allocation import cap_book


@dataclass(frozen=True)
class HostSpec:
    host: PerpetualSpec
    backend: str = 'cpu'


class Executor:
    precisions = frozenset({'f64'})

    def device_spec(self, host):
        return HostSpec(host)

    def simulate(self, W, spec, start, ppy, precision):
        assert precision == 'f64', 'perpetual execution requires the FP64 policy'
        arrays = _bt.perpetual(np.ascontiguousarray(W, dtype=np.float64),
                               tuple(getattr(spec.host, name) for name in ARRAY_FIELDS),
                               spec.host.leverage)
        return simulation_result(arrays, host_statistics(arrays[0], start, ppy))

    def blend(self, W, U, owners, weights, spec, start, ppy, precision, batch):
        results = []
        allocation = Allocation(spec.host.leverage, True)
        for lo in range(0, len(weights), batch):
            books = U[None] + weights[lo:lo + batch, None, None] * W[owners[lo:lo + batch]]
            books = cap_book(books, allocation, np)
            results.append(self.simulate(books, spec, start, ppy, precision))
        return concatenate_results(results, start, np)

    def fingerprints(self, W):
        return book_fingerprints(W, np, np.asarray)

    def close(self):
        pass


def prepare():
    return Executor()
