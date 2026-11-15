"""Isolated candidate arithmetic; only selected rolling reductions are paired."""
from pathlib import Path
import numpy as np
import cupy as cp
from foundry.ops.cuda import Ops
from foundry.numeric.cuda_source import paired_source

class PairedOps(Ops):
    def __init__(self):
        super().__init__()
        self.pair_module=cp.RawModule(code=paired_source()+'\n'+Path(__file__).with_name('kernels').joinpath('pair_windows.cu').read_text(),options=('--std=c++17','--fmad=false'))
        self.pair_kernel=self.pair_module.get_function('pair_window')
        def apply(mode):return lambda x,w:self.window(x,w,mode)
        for name,mode in [('ts_sum',0),('ts_mean',1),('ts_mad',2),('ts_wma',3)]:self.IMPL[name]=apply(mode)
        self.IMPL['ts_decay_exp_window']=lambda x,w,f:self.window(x,w,4,f)
    def window(self,x,w,mode,factor=0.):
        t,s=x.shape;out=cp.empty_like(x)
        self.pair_kernel(((t*s+127)//128,),(128,),(x,out,np.int32(t),np.int32(s),np.int32(w),np.int32(mode),np.float64(factor)))
        return out
    def close(self):
        del self.pair_kernel,self.pair_module
        super().close()

def prepare():return PairedOps()
