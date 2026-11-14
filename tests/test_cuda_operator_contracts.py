"""Numerical operations that can change formula validity or trading decisions."""
from dataclasses import replace
import unittest
from unittest.mock import patch
import numpy as np
import cupy as cp
from foundry.runtime import prepare_backend
from foundry.portfolio import admission
from fixtures import runtime_for, research_for


class RollingContractTests(unittest.TestCase):
    def setUp(self):
        self.cpu=prepare_backend('cpu');self.gpu=prepare_backend('cuda')
        self.addCleanup(self.cpu.close);self.addCleanup(self.gpu.close)

    def test_recurring_slope_and_pair_history_match_after_extreme_values_leave(self):
        rng=np.random.default_rng(25)
        x=rng.normal(size=(380,7));y=rng.normal(size=x.shape)
        y[13,0]=1e12;y[30:33,1]=1e10
        x[:9,2]=np.nan;y[:17,2]=np.nan
        x[110:112,3]=np.nan;y[205,4]=np.nan;x[:,5]=np.nan;y[-10:,6]=np.nan
        for window in (3,30,60,400):
            for name,args in [('ts_std',(y,)),('ts_slope',(y,)),('ts_corr',(x,y)),('ts_cov',(x,y))]:
                with self.subTest(name=name,window=window):
                    expected=self.cpu.impl[name](*args,window)
                    actual=cp.asnumpy(self.gpu.impl[name](*(cp.asarray(a) for a in args),window))
                    np.testing.assert_array_equal(actual,expected)
                    prefix=cp.asnumpy(self.gpu.impl[name](*(cp.asarray(a[:220]) for a in args),window))
                    np.testing.assert_array_equal(prefix,actual[:220])

    def test_constant_and_near_constant_population_standardization(self):
        x=np.full((100,4),.1)
        x[:,1]+=np.arange(100)%2*1e-13
        x[:,2]+=np.sin(np.arange(100))*1e-11
        x[:8,3]=np.nan;x[70,3]=np.nan
        for name in ('ts_std','ts_zscore','ts_snr'):
            expected=self.cpu.impl[name](x,60)
            actual=cp.asnumpy(self.gpu.impl[name](cp.asarray(x),60))
            np.testing.assert_array_equal(actual,expected)
            if name != 'ts_std':
                self.assertTrue(np.isnan(actual[:,0]).all())

    def test_slope_resynchronization_uses_each_columns_start(self):
        x=np.random.default_rng(7).normal(size=(65560,2))
        x[:11,1]=np.nan;x[300,0]=1e9
        expected=self.cpu.impl['ts_slope'](x,3)
        actual=cp.asnumpy(self.gpu.impl['ts_slope'](cp.asarray(x),3))
        np.testing.assert_array_equal(actual,expected)

    def test_tiny_tanh_is_the_correctly_rounded_identity(self):
        magnitudes=np.r_[np.nextafter(0.,1.),np.logspace(-300,-9,80)]
        x=np.column_stack((magnitudes,-magnitudes))
        np.testing.assert_array_equal(self.cpu.impl['tanh'](x),x)
        portable=prepare_backend('portable');self.addCleanup(portable.close)
        np.testing.assert_array_equal(portable.impl['tanh'](x),x)
        np.testing.assert_array_equal(cp.asnumpy(self.gpu.impl['tanh'](cp.asarray(x))),x)

    def test_constant_covariance_cannot_create_a_signal(self):
        portable=prepare_backend('portable');self.addCleanup(portable.close)
        varied=np.random.default_rng(25).normal(size=(320,3))
        constant=np.broadcast_to([1.,.1,1e10],varied.shape).copy()
        constant[100:140,1]=.2
        for backend in (self.cpu,self.gpu,portable):
            xp=backend.xp
            for left,right in ((varied,constant),(constant,varied)):
                cov=backend.to_host(backend.impl['ts_cov'](xp.asarray(left),xp.asarray(right),30))
                corr=backend.to_host(backend.impl['ts_corr'](xp.asarray(left),xp.asarray(right),30))
                np.testing.assert_array_equal(cov[29:100],0.)
                np.testing.assert_array_equal(cov[169:],0.)
                self.assertTrue(np.isnan(corr[29:100]).all())
                self.assertTrue(np.isnan(corr[169:]).all())
                self.assertTrue(np.isfinite(corr[100:129,1]).all())


class FeasibleWeightTieTests(unittest.TestCase):
    def test_tiny_utility_rounding_cannot_choose_a_different_weight(self):
        original=admission.weight_metrics
        for backend in ('cpu','cuda'):
            with runtime_for('crypto_perpetual',backend) as runtime:
                engine=runtime.engine;xp=engine.xp
                research=research_for(runtime)
                rows=engine.evaluate(('mul_const(gt(ma_dist_20,0),-1)',))
                for epsilon,chosen in ((1e-13,0),(1e-5,1)):
                    def measured(rr,dd,ref,ppy,shape):
                        value=original(rr,dd,ref,ppy,shape)
                        utility=xp.full(shape,.5);utility[:,1]+=epsilon
                        return replace(value,valid=xp.ones(shape,dtype=bool),
                            checks=xp.ones(shape+(4,),dtype=bool),utility=utility,cagr=xp.ones(shape))
                    with self.subTest(backend=backend,epsilon=epsilon), patch.object(admission,'weight_metrics',side_effect=measured), patch.object(engine,'blend',wraps=engine.blend) as blend:
                        proposal=admission.proposals(engine,rows,research.portfolio.reference)[0]
                        weights=engine.to_host(blend.call_args.args[3])
                        self.assertNotEqual(weights[0],weights[1])
                        self.assertTrue(proposal.passed)
                        self.assertEqual(proposal.weight,float(weights[chosen]))
