"""Full equity universes on the CPU reference, executed on the authorized cloud worker."""
from dataclasses import replace
import unittest

import numpy as np

from foundry.core.metrics import summary
from fixtures import runtime_for


class MarketTests(unittest.TestCase):
    def test_equities_run_end_to_end(self):
        for name in ('us_equity', 'a_share'):
            with self.subTest(market=name):
                runtime = runtime_for(name, 'cpu')
                self.addCleanup(runtime.close)
                self.assertGreater(runtime.panel.shape[1], 128)
                for row in runtime.engine.evaluate(('gt(ma_dist_20,0)', 'ts_zscore(dvol,20)')):
                    self.assertTrue(row.ok, row.error)
                    metrics = summary(row.returns, runtime.market.ppy, runtime.market.geometric)
                    self.assertTrue(np.isfinite(row.returns).all())
                    self.assertTrue(all(np.isfinite(v) for v in metrics.values()), metrics)

    def test_a_share_limits_suspensions_and_sell_tax_bind(self):
        runtime = runtime_for('a_share', 'cpu')
        self.addCleanup(runtime.close)
        spec = runtime.host_spec
        for code in (1, 2, 3):
            self.assertTrue((spec.block == code).any(), code)
        self.assertTrue((spec.cost_sell > spec.cost_buy)[spec.live > 0].all())
        row = runtime.engine.evaluate(('gt(ma_dist_20,0)',))[0]
        free = replace(spec, block=None)
        self.assertFalse(np.array_equal(runtime.executor.simulate(row.book[None], runtime.executor.device_spec(spec), 0, runtime.market.ppy, 'native').returns, runtime.executor.simulate(row.book[None], runtime.executor.device_spec(free), 0, runtime.market.ppy, 'native').returns))


if __name__ == '__main__':
    unittest.main()
