"""Independent CPU moments and GPU paired moments on identical input arrays."""
import unittest

import cupy as cp
import numpy as np

from foundry.numeric.cuda import Moments
from foundry.numeric.host import measure
from foundry.numeric.cuda_source import paired_source


class ReturnMomentTests(unittest.TestCase):
    def test_pair_normalization_preserves_cancellation_and_products(self):
        rng=np.random.default_rng(31)
        x=rng.uniform(-1.,1.,1024)*10.**rng.uniform(-10.,10.,1024)
        y=rng.uniform(-1.,1.,1024)*10.**rng.uniform(-10.,10.,1024)
        y[:512]=-x[:512]*(1.+rng.choice([-1.,0.,1.],512)*2.**-40)
        kernel=cp.RawKernel(paired_source()+r'''
extern "C" __global__ void arithmetic(const double* x,const double* y,double* out) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    FloatPair a(x[i]),b(y[i]);
    out[i*3]=double(a+b);out[i*3+1]=double(a*b);out[i*3+2]=double(a/b);
}''','arithmetic',options=('--std=c++17','--fmad=false'))
        output=cp.empty((len(x),3),cp.float64)
        kernel((len(x)//128,),(128,),(cp.asarray(x),cp.asarray(y),output))
        a,b=x.astype(np.longdouble),y.astype(np.longdouble)
        reference=np.column_stack((a+b,a*b,a/b))
        scale=np.column_stack((np.abs(a)+np.abs(b),np.abs(a*b),np.abs(a/b)))
        error=np.abs(cp.asnumpy(output).astype(np.longdouble)-reference)
        self.assertTrue(np.all(error <= scale*2.**-44))

    def test_weighted_moments_and_growth_preserve_reference(self):
        implementation=Moments();self.addCleanup(implementation.close)
        r=np.random.default_rng(25).normal(.0003,.02,(129,601))
        r[0]=0.;r[1]=.125;r[2]=-.125
        p=np.geomspace(.1,1.,r.shape[1]);p/=p.sum()
        expected=measure(r,p,365.)
        actual=implementation.measure(cp.asarray(r),cp.asarray(p),365.)
        for name in ('sharpe','cagr','weighted_sharpe','recent_growth'):
            first,second=getattr(expected,name),cp.asnumpy(getattr(actual,name))
            np.testing.assert_allclose(second,first,rtol=1e-10,atol=1e-10)

    def test_nonfinite_and_ruined_growth_remains_nonfinite(self):
        implementation=Moments();self.addCleanup(implementation.close)
        r=np.array([[0.,-1.,.02],[0.,-1.01,.02],[np.nan,0.,.02]])
        p=np.full(3,1/3)
        actual=implementation.measure(cp.asarray(r),cp.asarray(p),365.)
        self.assertEqual(float(actual.cagr[0]),-1.)
        self.assertTrue(bool(cp.isnan(actual.cagr[1:]).all()))
        self.assertTrue(bool((~cp.isfinite(actual.recent_growth)).all()))


if __name__=='__main__':unittest.main()
