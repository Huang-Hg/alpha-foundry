"""Native extensions: ABI and invariants of the operator library and the execution kernel. No market data, no GPU;
random-walk prices only exercise the mechanics (CI, also under ASan/UBSan)."""
import unittest
from types import SimpleNamespace

import numpy as np

from foundry.backtest import cpu
from foundry.backtest.spec import ExecutionSpec
from foundry.ops import _ops
from foundry.config import PortfolioConfig
from foundry.portfolio.admission import return_requirements

T, S = 300, 8


def prices(seed=0):
    rng = np.random.default_rng(seed)
    close = np.cumprod(1 + 0.01 * rng.standard_normal((T, S)), axis=0) * 100
    return close * (1 + 0.002 * rng.standard_normal((T, S))), close


class OperatorTests(unittest.TestCase):
    def test_zero_drawdown_reference_has_an_explicit_growth_limit(self):
        cfg = PortfolioConfig()
        reference = SimpleNamespace(xp=np, cfg=cfg, maxdd=0., cagr=0., recent_growth=0.)
        actual = return_requirements(np.array([0., -1e-15, -.1]), reference)
        np.testing.assert_array_equal(actual[:, :2], [[cfg.min_sharpe_gain, 0.]]*3)
        np.testing.assert_array_equal(actual[0, 2:], [0., 0.])
        self.assertTrue(np.isposinf(actual[1:, 2:]).all())

    def test_truncated_history_shorter_than_window_is_only_warmup(self):
        x=np.arange(117*8,dtype=float).reshape(117,8)
        for name in ('ts_sum','ts_mean','ts_wma','ts_rank'):
            with self.subTest(op=name):
                self.assertTrue(np.isnan(getattr(_ops,name)(x,120)).all())

    def test_rolling_windows_warm_up_and_propagate_nan(self):
        x = np.random.default_rng(1).standard_normal((T, S))
        x[100, 3] = np.nan
        m = _ops.ts_mean(x, 10)
        self.assertTrue(np.isnan(m[:9]).all() and np.isfinite(m[9:100]).all())
        self.assertTrue(np.isnan(m[100:110, 3]).all() and np.isfinite(m[110:, 3]).all())
        np.testing.assert_allclose(m[50], x[41:51].mean(0), rtol=1e-9, atol=1e-15)     # a running sum, not a fresh one
        self.assertTrue(np.array_equal(_ops.ts_ref(x, 3)[3:], x[:-3], equal_nan=True))
        r = _ops.cs_rank(x)
        self.assertTrue(((r[np.isfinite(r)] >= 0) & (r[np.isfinite(r)] <= 1)).all())


class KernelTests(unittest.TestCase):
    def setUp(self):
        self.executor = cpu.prepare()
        self.addCleanup(self.executor.close)
        self.open, self.close = prices()
        self.cost = 0.001 * self.open
        self.spec = ExecutionSpec.build(self.open, self.close, self.cost)
        self.W = np.full((T, S), 1.0 / S)

    def returns(self, weights, spec):
        result = self.executor.simulate(weights[None], self.executor.device_spec(spec), 0, 252., 'f64')
        self.assertIsNone(result.risk)
        return result.returns[0]

    def test_cash_earns_nothing_and_costs_lower_returns(self):
        self.assertEqual(np.abs(self.returns(np.zeros((T, S)), self.spec)).max(), 0)
        free = self.returns(self.W, ExecutionSpec.build(self.open, self.close, 0 * self.open))
        paid = self.returns(self.W, self.spec)
        self.assertLess(paid.sum(), free.sum())

    def test_general_rules_reduce_to_the_symmetric_case_bitwise(self):
        general = ExecutionSpec.build(self.open, self.close, self.cost, cost_sell=self.cost.copy(),
                                      block=np.zeros((T, S), np.uint8), carry=np.zeros((T, S)))
        self.assertTrue(np.array_equal(self.returns(self.W, general), self.returns(self.W, self.spec)))

    def test_a_blocked_direction_holds_the_position(self):
        rising = np.linspace(0, 1, T)[:, None] * np.ones((1, S)) / S        # the target keeps growing
        block = np.zeros((T, S), np.uint8)
        block[150:, 0] = 1                                                 # asset 0 cannot be bought from day 150
        blocked = ExecutionSpec.build(self.open, self.close, self.cost, block=block)
        a, b = self.returns(rising, self.spec), self.returns(rising, blocked)
        self.assertTrue(np.array_equal(a[:150], b[:150]))                   # day 150 trades at its open
        self.assertFalse(np.array_equal(a[150:], b[150:]))

    def test_sell_side_cost_and_carry_only_cost(self):
        turn = np.where(np.arange(T)[:, None] % 2 == 0, 1.0 / S, 0.0) * np.ones((1, S))
        taxed = ExecutionSpec.build(self.open, self.close, self.cost, cost_sell=2 * self.cost)
        self.assertLess(self.returns(turn, taxed).sum(), self.returns(turn, self.spec).sum())
        short = -self.W
        carried = ExecutionSpec.build(self.open, self.close, self.cost, carry=np.full((T, S), 1e-4))
        self.assertLess(self.returns(short, carried).sum(), self.returns(short, self.spec).sum())
        self.assertTrue(np.array_equal(self.returns(self.W, carried), self.returns(self.W, self.spec)))   # long only

    def test_no_price_closes_the_position(self):
        o, c = self.open.copy(), self.close.copy()
        o[200:, 2] = c[200:, 2] = np.nan                                    # asset 2 stops trading
        spec = ExecutionSpec.build(o, c, 0.001 * o)
        self.assertFalse(spec.live[200:, 2].any())
        self.assertTrue(np.isfinite(self.returns(self.W, spec)).all())


if __name__ == '__main__':
    unittest.main()
