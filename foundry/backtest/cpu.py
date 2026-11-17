"""CPU batch execution of the same market rules and statistics contract as CUDA."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from foundry.backtest import _bt
from foundry.numeric import metrics
from foundry.numeric.parallel import HostWorkers
from foundry.types import ExecutionSpec, FloatArray, IntArray, SimulationResult


@dataclass(frozen=True)
class HostSpec:
    host: ExecutionSpec
    backend: str = 'cpu'


class Executor:
    parallel_trials = True
    precisions = frozenset({'native'})

    def __init__(self, workers: HostWorkers | None = None, gross_limit: float = 1.):
        self.map = map if workers is None else workers.map
        self.gross_limit = gross_limit

    def device_spec(self,host: ExecutionSpec) -> HostSpec:
        return HostSpec(host)

    def simulate(self,W: FloatArray,spec: HostSpec,start: int,ppy: float,precision: str) -> SimulationResult:
        assert precision == 'native'
        s=spec.host
        def account(w: FloatArray) -> FloatArray:
            return _bt.simulate(np.ascontiguousarray(w,dtype=np.float64),s.open_raw,s.prev_close,s.open_ff0,
                                s.close_ff0,s.cost_buy,s.cost_sell,s.live,s.block,s.carry,
                                s.schedule.decision_index,s.schedule.measurement_index,
                                s.schedule.measurement_count)
        return self._result(np.stack(list(self.map(account, W))), start, ppy)

    @staticmethod
    def _result(rr: FloatArray, start: int, ppy: float) -> SimulationResult:
        r=rr[:,start:]
        stats=np.column_stack((metrics.sharpe(r,ppy,np),r.mean(1)*ppy,r.std(1)*ppy**.5,
                               metrics.max_drawdown(r,np),np.prod(1+r,axis=1)-1))
        return SimulationResult(rr, stats, None)

    def blend(self,W: FloatArray,U: FloatArray,owners: IntArray,weights: FloatArray,
              spec: HostSpec,start: int,ppy: float,precision: str,batch: int) -> SimulationResult:
        s = spec.host
        def trial(i: int) -> FloatArray:
            book = U + weights[i] * W[owners[i]]
            gross = book.sum(1, keepdims=True)
            if self.gross_limit != 1.:
                gross /= self.gross_limit
            book /= np.maximum(1., gross)
            return _bt.simulate(book, s.open_raw, s.prev_close, s.open_ff0, s.close_ff0,
                                s.cost_buy, s.cost_sell, s.live, s.block, s.carry,
                                s.schedule.decision_index, s.schedule.measurement_index,
                                s.schedule.measurement_count)
        result = self._result(np.stack(list(self.map(trial, range(len(weights))))), start, ppy)
        return SimulationResult(result.returns[:, start:], result.statistics, None)

    def fingerprints(self,W: FloatArray) -> list[int]:
        values=np.ascontiguousarray(np.where(W==0,0.,W),dtype=np.float64).reshape(len(W),-1).view(np.uint64)
        index=np.arange(1,values.shape[1]+1,dtype=np.uint64)
        x=values^(index*np.uint64(0x9e3779b97f4a7c15))
        x=(x^(x>>30))*np.uint64(0xbf58476d1ce4e5b9)
        x=(x^(x>>27))*np.uint64(0x94d049bb133111eb)
        return np.bitwise_xor.reduce(x^(x>>31),axis=1).tolist()

    def close(self) -> None:
        pass


def prepare(workers: HostWorkers | None = None, gross_limit: float = 1.) -> Executor:
    return Executor(workers, gross_limit)
