"""GPU-only paired statistics: identical input returns isolate statistics error."""
import unittest
from pathlib import Path
import numpy as np
import cupy as cp

from foundry.backtest import cuda


def statistics_kernel(group):
    code='\n'.join((Path(cuda.__file__).with_name('kernels')/f).read_text() for f in
                   ('float_pair.cuh','backtest.cu','statistics.cuh'))
    code+=f'''
extern "C" __global__ void test_stats(const double* r,int T,int start,double ppy,double* s) {{
    paired_statistics<{group}>(r+((long long)blockIdx.x)*T,T,start,ppy,s+5*blockIdx.x);
}}
'''
    return cp.RawModule(code=code,options=('--std=c++14',)).get_function('test_stats')


def reference(r,start,ppy):
    x=r[:,start:]
    mean=x.mean(1)*ppy;vol=x.std(1)*np.sqrt(ppy)
    with np.errstate(divide='ignore',invalid='ignore'):
        sh=mean/vol
    cum=np.cumprod(1+x,axis=1)
    peak=np.maximum.accumulate(np.maximum(cum,1),axis=1)
    dd=np.minimum((cum/peak-1).min(1),0)
    return np.stack((sh,mean,vol,dd,cum[:,-1]-1),axis=1)


class StatisticsTests(unittest.TestCase):
    def test_range_cancellation_and_peak_boundaries(self):
        rng=np.random.default_rng(25)
        r=rng.normal(.0003,.015,(10,1024))
        r[0]=0
        r[1]=.125                 # exact constant, Sharpe +inf
        r[2]=-.125                # exact constant, Sharpe -inf
        r[3]=.9                   # terminal wealth ~1e285, beyond FP32
        r[4]=-.5                  # approaches FP64 underflow
        r[5,:512]=.8;r[5,512:]=-.9
        r[6,:512]=-.9;r[6,512:]=.8
        r[7]=np.tile([.5,-1/3],512)
        r[8]=1e-10*np.tile([1.,-1.],512)
        r[9]=-.0001               # initial wealth must count as peak
        d=cp.asarray(r)
        for group in (32,128):
            fn=statistics_kernel(group)
            for start in (0,17):
                with self.subTest(group=group,start=start):
                    actual=cp.empty((len(r),5),cp.float64)
                    fn((len(r),),(group,),(d,np.int32(r.shape[1]),np.int32(start),np.float64(242.),actual))
                    expect=reference(r,start,242.)
                    # Constant non-dyadic rows have an implementation-dependent
                    # tiny computed variance; use the exact mathematical limit.
                    expect[[3,4,9],0]=[np.inf,-np.inf,-np.inf]
                    expect[[3,4,9],2]=0
                    np.testing.assert_allclose(cp.asnumpy(actual),expect,rtol=1e-9,atol=1e-10)

    def test_small_nonzero_variance(self):
        x=.001+np.tile([1e-9,-1e-9],512)
        r=cp.asarray(x[None]);st=cp.empty((1,5),cp.float64)
        statistics_kernel(128)((1,),(128,),(r,np.int32(1024),np.int32(0),np.float64(242.),st))
        np.testing.assert_allclose(cp.asnumpy(st),reference(x[None],0,242.),rtol=1e-8,atol=1e-10)

    def test_segment_tails_and_signed_ruin(self):
        rng=np.random.default_rng(25)
        for days in (1,7,31,33,127,129,601):
            r=rng.normal(0,.03,(5,days))
            r[0,days//2]=-1.
            r[1,days//2]=-1.2
            r[2,::3]=-.08;r[2,1::3]=.09
            for group in (32,128):
                with self.subTest(days=days,group=group):
                    st=cp.empty((len(r),5),cp.float64)
                    statistics_kernel(group)((len(r),),(group,),
                        (cp.asarray(r),np.int32(days),np.int32(0),np.float64(242.),st))
                    np.testing.assert_allclose(cp.asnumpy(st),reference(r,0,242.),rtol=1e-9,atol=1e-10)


if __name__=='__main__':
    unittest.main()
