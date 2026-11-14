"""GPU-only execution accuracy tests; no market data or training required."""
import unittest
from dataclasses import replace
from types import SimpleNamespace

import cupy as cp
import numpy as np

from foundry.backtest import cpu, cuda
from foundry.backtest.spec import ExecutionSpec
from foundry.config import PortfolioConfig
from foundry.portfolio.admission import return_requirements


def fixture(executor, days=257, assets=73):
    rng = np.random.default_rng(25)
    close = np.exp(np.cumsum(rng.normal(0, .02, (days, assets)), axis=0))
    close *= np.geomspace(1e-4, 1e5, assets)
    open_ = close * np.exp(rng.normal(0, .005, close.shape))
    open_[10:14, 0] = np.nan
    close[10:14, 0] = np.nan
    open_[:8, -1] = close[:8, -1] = np.nan
    spec = ExecutionSpec.build(open_, close, open_ * .001)
    W = rng.uniform(0, 1, (5, days, assets)) / assets
    W[0] = 0
    W[1] *= 3                         # exposure cap
    W[2, :, ::2] *= -1               # short and long positions
    W[:, :3] = 0
    W[3, 12, 0] = np.nan
    return cp.asarray(W), executor.device_spec(spec)


class CudaBacktestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cuda = cuda.prepare()
        cls.cpu = cpu.prepare()

    @classmethod
    def tearDownClass(cls):
        cls.cuda.close()
        cls.cpu.close()

    def test_zero_drawdown_growth_limit_matches_cpu(self):
        cfg = PortfolioConfig()
        reference = SimpleNamespace(xp=np, cfg=cfg, maxdd=0., cagr=0., recent_growth=0.)
        drawdowns = np.array([0., -1e-15, -.1])
        expected = return_requirements(drawdowns, reference)
        reference.xp = cp
        actual = return_requirements(cp.asarray(drawdowns), reference)
        np.testing.assert_array_equal(cp.asnumpy(actual), expected)

    def test_gpu_preserves_general_rules_and_large_universes(self):
        for assets in (7, 129):
            W, base = fixture(self.cuda, days=31, assets=assets)
            host = base.host
            block = np.zeros(host.live.shape, dtype=np.uint8)
            block[12:18, 1] = 3
            general = replace(host, cost_sell=host.cost_buy * 3, block=block,
                              carry=np.full(host.live.shape, .0003))
            for source in (host, general):
                spec = self.cuda.device_spec(source)
                if assets > 128 or source is general:
                    self.assertEqual(spec.backend, 'cuda')
                    self.assertTrue(spec.general)
                    expected = self.cpu.simulate(cp.asnumpy(W), self.cpu.device_spec(source), 3, 252., 'f64').returns
                    actual = self.cuda.simulate(W, spec, 3, 252., 'paired')
                    np.testing.assert_allclose(cp.asnumpy(actual.returns), expected, atol=1e-10, rtol=0)

    def test_cascading_freeze_requires_reallocation(self):
        o = np.full((3, 3), 100.)
        c = o.copy(); c[2] = [100., 120., 80.]
        block = np.zeros((3, 3), np.uint8); block[2] = [2, 2, 0]
        spec = ExecutionSpec.build(o, c, o*0, block=block)
        W = np.array([[.8,.1,0], [.2,.2,.6], [0,0,0]])
        expected = self.cpu.simulate(W[None], self.cpu.device_spec(spec), 0, 242., 'f64').returns[0]
        np.testing.assert_allclose(expected, 0, atol=1e-15)
        for precision in ('f64', 'paired'):
            r = self.cuda.simulate(cp.asarray(W[None]), self.cuda.device_spec(spec), 0, 242., precision)
            cp.testing.assert_allclose(r.returns[0], cp.asarray(expected), atol=1e-12, rtol=0)

    def test_gpu_all_rule_combinations(self):
        rng = np.random.default_rng(31)
        for assets in (7, 129, 299, 500):
            W, base = fixture(self.cuda, days=31, assets=assets)
            host = base.host
            block = rng.integers(0, 4, host.live.shape, dtype=np.uint8)
            for flags in range(8):
                with self.subTest(assets=assets, flags=flags):
                    source = replace(host, cost_sell=host.cost_buy*3 if flags&1 else None,
                                     block=block if flags&2 else None,
                                     carry=np.full(host.live.shape,.0003) if flags&4 else None)
                    expected = self.cpu.simulate(cp.asnumpy(W), self.cpu.device_spec(source), 3, 242., 'f64').returns
                    for precision in ('f64','paired'):
                        actual = self.cuda.simulate(W,self.cuda.device_spec(source),3,242.,precision)
                        np.testing.assert_allclose(cp.asnumpy(actual.returns),expected,atol=1e-10,rtol=0)

    def test_pair_rebases_large_wealth_without_changing_returns(self):
        close = np.tile([[1., 2.], [2., 1.]], (200, 1))
        open_ = np.vstack([close[:1], close[:-1]])
        spec = self.cuda.device_spec(ExecutionSpec.build(open_, close, open_ * 0))
        W = cp.asarray((close == 1.)[None], dtype=cp.float64)
        exact = self.cuda.simulate(W, spec, 0, 365., 'f64')
        actual = self.cuda.simulate(W, spec, 0, 365., 'paired')
        self.assertTrue(bool(cp.isfinite(exact.returns).all()))
        self.assertGreater(float(cp.prod(1 + exact.returns)), 1e100)
        cp.testing.assert_allclose(actual.returns, exact.returns, atol=1e-10, rtol=0)

    def test_pair_retains_low_bits(self):
        x = cp.asarray([1. + 2.**-30, 1e-6, 123456.7890123, -1e10], dtype=cp.float64)
        p = self.cuda.paired(x)
        cp.testing.assert_allclose(p[..., 0].astype(cp.float64) + p[..., 1], x, rtol=4e-15, atol=0)

    def test_behavior_fingerprints_match_cpu_and_preserve_duplicates(self):
        weights = np.random.default_rng(11).normal(size=(5, 37, 129))
        weights[1] = weights[0]
        weights[2] = 0.
        weights[3] = -0.
        expected = self.cpu.fingerprints(weights)
        actual = self.cuda.fingerprints(cp.asarray(weights))
        self.assertEqual(actual, expected)
        self.assertEqual(actual[0], actual[1])
        self.assertEqual(actual[2], actual[3])
        self.assertNotEqual(actual[0], actual[4])

    def test_paired_recurrence_matches_f64(self):
        for assets in (1, 31, 32, 33, 73, 128):
            with self.subTest(assets=assets):
                W, spec = fixture(self.cuda, assets=assets)
                exact = self.cuda.simulate(W, spec, 3, 365., 'f64')
                actual = self.cuda.simulate(W, spec, 3, 365., 'paired')
                cp.testing.assert_allclose(actual.returns, exact.returns, rtol=0, atol=1e-10)
                cp.testing.assert_allclose(actual.statistics[:, 1:], exact.statistics[:, 1:], rtol=1e-8, atol=1e-10)
                cp.testing.assert_allclose(actual.statistics[1:, 0], exact.statistics[1:, 0], rtol=1e-8, atol=1e-8)

    def test_fused_pair_blends_and_batch_tail(self):
        W, spec = fixture(self.cuda)
        W = cp.nan_to_num(W)
        U = W[1] * .8
        owners = cp.asarray([1, 2, 4, 1, 3], dtype=cp.int32)
        weights = cp.asarray([.01, .3, .5, 1., 1e-8], dtype=cp.float64)
        expected = self.cuda.blend(W, U, owners, weights, spec, 3, 365., 'f64', 3)
        actual = self.cuda.blend(W, U, owners, weights, spec, 3, 365., 'paired', 3)
        cp.testing.assert_allclose(actual.returns, expected.returns, atol=1e-10, rtol=0)
        cp.testing.assert_allclose(actual.statistics, expected.statistics, atol=1e-8, rtol=1e-8)


if __name__ == '__main__':
    unittest.main()
