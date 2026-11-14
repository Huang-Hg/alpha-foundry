"""Equities and crypto spot share execution, metrics and admission semantics."""
from dataclasses import replace
import unittest

import cupy as cp
import numpy as np

from foundry.backtest import cpu, cuda
from foundry.portfolio import admission
from foundry.candidates import Formula
from foundry.research_types import Candidate
from fixtures import EXPRESSIONS, research_for, runtime_for


class MarketDispatchTests(unittest.TestCase):
    def test_cpu_and_cuda_formula_books_and_research_decisions_match(self):
        expressions = ('gt(ma_dist_20,0)', 'gt(ts_rank(ma_dist_20,120),0.7)',
                       'gt(cs_rank(is_nan(ma_dist_60)),0.5)',
                       'gt(cs_rank(ts_mean(rangepos,20)),0.7)')
        candidates = tuple(Candidate(str(i), Formula(expression)) for i, expression in enumerate(expressions))
        for name in ('us_equity', 'a_share', 'crypto_spot'):
            with self.subTest(market=name), runtime_for(name, 'cpu') as host, runtime_for(name) as device:
                expected = host.engine.evaluate(expressions)
                actual = device.engine.evaluate(expressions)
                for first, second in zip(expected, actual):
                    self.assertEqual((first.ok, first.valid_returns), (second.ok, second.valid_returns))
                    np.testing.assert_array_equal(first.book > 0, cp.asnumpy(second.book) > 0)
                    np.testing.assert_allclose(first.book, cp.asnumpy(second.book), rtol=0, atol=1e-12)
                    np.testing.assert_allclose(first.returns, cp.asnumpy(second.returns), rtol=0, atol=1e-10)
                    for key in ('sharpe', 'ann_ret', 'cagr', 'maxdd', 'total_ret', 'turnover', 'exposure'):
                        self.assertAlmostEqual(getattr(first.metrics, key), getattr(second.metrics, key), delta=1e-8)
                host_research, device_research = research_for(host), research_for(device)
                host_result = host_research.evaluate(candidates, host_research.context.id)
                device_result = device_research.evaluate(candidates, device_research.context.id)
                for first, second in zip(host_result.rows, device_result.rows):
                    a, b = first.verdict, second.verdict
                    self.assertEqual((a.passed, a.stopped_at, a.reason), (b.passed, b.stopped_at, b.reason))
                    self.assertEqual(a.proposal.weight, b.proposal.weight)
                    self.assertEqual([(c.name, c.passed) for c in a.proposal.checks],
                                     [(c.name, c.passed) for c in b.proposal.checks])

    def test_decisions_weights_and_threshold_sides_match_f64(self):
        for name in ('us_equity', 'a_share', 'crypto_spot'):
            with self.subTest(market=name), runtime_for(name) as runtime:
                engine = runtime.engine
                rows = engine.evaluate(EXPRESSIONS)
                paired_rows = engine.evaluate(EXPRESSIONS, precision='paired')
                self.assertTrue(all(r.ok for r in rows))
                for expected, actual in zip(rows, paired_rows):
                    cp.testing.assert_array_equal(actual.book > 0, expected.book > 0)
                    cp.testing.assert_allclose(actual.returns, expected.returns, rtol=0, atol=1e-10)
                    for key in ('sharpe', 'ann_ret', 'cagr', 'maxdd', 'total_ret'):
                        self.assertAlmostEqual(getattr(actual.metrics, key), getattr(expected.metrics, key), delta=1e-8)
                ref = research_for(runtime).portfolio.reference
                exact = admission.proposals(engine, rows, ref, 'f64')
                paired = admission.proposals(engine, rows, ref, 'paired')
                for expected, actual in zip(exact, paired):
                    self.assertEqual((actual.passed, actual.weight), (expected.passed, expected.weight))
                    self.assertEqual([(c.name, c.passed) for c in actual.checks],
                                     [(c.name, c.passed) for c in expected.checks])
                    for key in ('wsharpe', 'sharpe', 'cagr', 'maxdd', 'recent_growth'):
                        self.assertAlmostEqual(getattr(actual, key), getattr(expected, key), delta=1e-8)
                threshold = exact[0].wsharpe - ref.wsharpe
                for offset in (-1e-7, 1e-7):
                    boundary = replace(ref, cfg=replace(ref.cfg, min_sharpe_gain=threshold + offset),
                                       precision_errors={})
                    a = admission.proposals(engine, rows[:1], boundary, 'f64')[0]
                    b = admission.proposals(engine, rows[:1], boundary, 'paired')[0]
                    self.assertEqual((a.passed, a.weight), (b.passed, b.weight))

    def test_full_universes_keep_all_execution_rules(self):
        executor = cuda.prepare()
        self.addCleanup(executor.close)
        for name in ('us_equity', 'a_share'):
            with self.subTest(market=name), runtime_for(name, 'cpu') as runtime:
                panel, spec, window, market = runtime.panel, runtime.host_spec, runtime.window, runtime.market
                self.assertGreater(panel.shape[1], 128)
                weights = np.stack([runtime.engine.signal_to_book(signal) for signal in
                                    (np.ones(panel.shape), (panel['close'] > panel['open']).astype(float))])
                weights[:, :window.start] = 0
                reference = runtime.executor.simulate(weights, runtime.executor.device_spec(spec), window.start, market.ppy, 'f64').returns
                device = executor.device_spec(spec)
                actual = executor.simulate(cp.asarray(weights), device, window.start, market.ppy, 'paired')
                self.assertEqual(device.backend, 'cuda')
                np.testing.assert_allclose(cp.asnumpy(actual.returns), reference, atol=1e-10, rtol=0)
                owners = cp.asarray([0, 1, 0], dtype=cp.int32)
                scales = cp.asarray([.25, .5, 1.])
                base = cp.asarray(weights[0] * .8)
                trial = base[None] + scales[:, None, None] * cp.asarray(weights)[owners]
                trial /= cp.maximum(1., trial.sum(2, keepdims=True))
                expected = runtime.executor.simulate(cp.asnumpy(trial), runtime.executor.device_spec(spec), window.start, market.ppy, 'f64').returns[:, window.start:]
                actual = executor.blend(cp.asarray(weights), base, owners, scales, device, window.start,
                                            market.ppy, 'paired', 2)
                np.testing.assert_allclose(cp.asnumpy(actual.returns), expected, atol=1e-10, rtol=0)


if __name__ == '__main__':
    unittest.main()
