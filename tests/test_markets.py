"""Every market end to end on its real data with the one execution kernel (local: needs data/, the crypto part the
GPU record in production/)."""
import json
import unittest

import numpy as np

from foundry.backtest import cpu
from foundry.config import ROOT
from foundry.core.metrics import summary
from foundry.expr.evaluate import backend, evaluate
from foundry.expr.parse import parse
from foundry.market.base import market


def run(m, panel, spec, window, formula):
    W = cpu.book(evaluate(parse(formula, panel.vocabulary).root, panel.values(panel.vocabulary), backend('cpu'), {}),
                 m.universe(panel))
    W[:window.start] = 0
    return W, cpu.simulate(W, spec)[window.start:]


class MarketTests(unittest.TestCase):
    def test_equities_run_end_to_end(self):
        for name in ('us_equity', 'a_share'):
            with self.subTest(market=name):
                m = market(name)
                panel, spec, window = m.load(backend('cpu'))
                self.assertGreater(panel.shape[1], 128)                   # beyond the CUDA kernel: CPU only
                for formula in ('gt(ma_dist_20, 0)', 'ts_zscore(dvol, 20)'):
                    _, rr = run(m, panel, spec, window, formula)
                    s = summary(rr, m.ppy, m.geometric)
                    self.assertTrue(np.isfinite(rr).all() and all(np.isfinite(v) for v in s.values()), (formula, s))

    def test_a_share_limits_and_suspensions_bind(self):
        m = market('a_share')
        panel, spec, window = m.load(backend('cpu'))
        b = spec.block
        self.assertTrue(((b == 1).any(), (b == 2).any(), (b == 3).any()) == (True, True, True))
        self.assertTrue((spec.cost_sell > spec.cost_buy)[spec.live > 0].all())          # stamp tax on sells
        W, _ = run(m, panel, spec, window, 'gt(ma_dist_20, 0)')
        free = type(spec)(**{**spec.__dict__, 'block': None})
        self.assertFalse(np.array_equal(cpu.simulate(W, spec), cpu.simulate(W, free)))

    def test_crypto_cpu_pipeline_reproduces_the_gpu_record(self):
        """Books identical and returns within rounding: the CPU kernel and the CUDA kernel are one recursion."""
        m = market('crypto')
        panel, spec, window = m.load(backend('cpu'))
        record = np.load(ROOT / 'production' / 'regression.npz')
        for i, member in enumerate(json.loads(str(record['members']))):
            W, rr = run(m, panel, spec, window, member['expr'])
            self.assertTrue(np.array_equal(W, record[f'member{i}_W']), member['expr'])
            self.assertLess(np.abs(rr - record[f'member{i}_rr']).max(), 1e-15, member['expr'])


if __name__ == '__main__':
    unittest.main()
