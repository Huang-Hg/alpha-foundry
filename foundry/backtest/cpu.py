"""CPU batch execution of the same market rules and statistics contract as CUDA."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from foundry.backtest import _bt
from foundry.backtest.spec import ExecutionSpec
from foundry.core import metrics
from foundry.contracts import SimulationResult


@dataclass(frozen=True)
class HostSpec:
    host: ExecutionSpec
    backend: str = 'cpu'


class Executor:
    parallel_trials = False
    screening_precisions = frozenset({'f64'})
    precisions = frozenset({'f64'})
    def device_spec(self,host):
        return HostSpec(host)

    def simulate(self,W,spec,start,ppy,precision):
        assert precision=='f64','CPU executor uses the FP64 reference policy'
        s=spec.host
        rr=np.stack([_bt.simulate(np.ascontiguousarray(w,dtype=np.float64),s.open_raw,s.prev_close,s.open_ff0,
                                 s.close_ff0,s.cost_buy,s.cost_sell,s.live,s.block,s.carry) for w in W])
        r=rr[:,start:]
        stats=np.column_stack((metrics.sharpe(r,ppy,np),r.mean(1)*ppy,r.std(1)*ppy**.5,
                               metrics.max_drawdown(r,np),np.prod(1+r,axis=1)-1))
        return SimulationResult(rr, stats, None)

    def blend(self,W,U,owners,weights,spec,start,ppy,precision,batch):
        returns=[];statistics=[]
        for lo in range(0,len(weights),batch):
            books=U[None]+weights[lo:lo+batch,None,None]*W[owners[lo:lo+batch]]
            books/=np.maximum(1.,books.sum(2,keepdims=True))
            result=self.simulate(books,spec,start,ppy,precision)
            returns.append(result.returns[:,start:]);statistics.append(result.statistics)
        return SimulationResult(np.concatenate(returns), np.concatenate(statistics), None)

    def fingerprints(self,W):
        values=np.ascontiguousarray(np.where(W==0,0.,W),dtype=np.float64).reshape(len(W),-1).view(np.uint64)
        index=np.arange(1,values.shape[1]+1,dtype=np.uint64)
        x=values^(index*np.uint64(0x9e3779b97f4a7c15))
        x=(x^(x>>30))*np.uint64(0xbf58476d1ce4e5b9)
        x=(x^(x>>27))*np.uint64(0x94d049bb133111eb)
        return np.bitwise_xor.reduce(x^(x>>31),axis=1).tolist()

    def close(self):
        pass


def prepare():
    return Executor()
