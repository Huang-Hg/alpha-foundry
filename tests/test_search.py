"""Search layer on the GPU and the real panel: the portfolio objective, the single-alpha gate, the member cache, the
noise control and the joint reweighting. production/library.json is only read (and must be byte-identical after every
test); every library here lives in memory or on a temporary copy."""
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cupy as cp
import numpy as np

from foundry.config import ExprConfig
from foundry.engine import Engine, null_lag
from foundry.expr.parse import validate
from foundry.ops.cuda import MAX_PAIR_WINDOW, MAX_WINDOW, unsupported
from foundry.portfolio import admission
from foundry.portfolio.library import SEED, STORE, Library
from foundry.portfolio.portfolio import Portfolio
from foundry.portfolio.reference import CFG, Reference, capped
from foundry.search import single
from foundry.search.evaluator import Archive, Evaluator, weight_metrics
from foundry.search.policy import Policy


class CudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = Engine()
        cls.stored = STORE.read_bytes()
        cls.members = json.loads(cls.stored)['members']
        cls.folder = tempfile.TemporaryDirectory()
        cls.copy = Path(cls.folder.name) / 'library.json'
        cls.copy.write_bytes(cls.stored)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def tearDown(self):
        self.assertEqual(STORE.read_bytes(), self.stored)

    def portfolio(self, members=None):
        """The production portfolio (None) or the given members, backed by a temporary copy of the library file."""
        return Portfolio(self.engine, Library.load(self.copy) if members is None else Library(members, self.copy))

    def test_return_trades_against_drawdown_but_sharpe_never(self):
        need = cp.asnumpy(admission.return_requirements(cp.asarray([-.3, -.33, -.27]),
                                                        SimpleNamespace(maxdd=-.3, cagr=1.4, recent_growth=-.2)))
        self.assertTrue((need[:, 0] == CFG.min_sharpe_gain).all() and (need[:, 1] == 0).all())
        self.assertEqual(need[0, 2:].tolist(), [0., 0.])                          # same drawdown: return must not fall
        self.assertAlmostEqual(need[1, 2], .14); self.assertAlmostEqual(need[1, 3], .02)     # 10% deeper: 10% more return
        self.assertAlmostEqual(need[2, 2], -.14); self.assertAlmostEqual(need[2, 3], -.02)   # 10% shallower: 10% may go

    def test_weight_metrics_is_admission_on_the_blended_book(self):
        """Reweighting and admission judge with one set of statistics: on the book admission chose, weight_metrics
        reproduces admission's deltas, verdict and utility; against the reference itself every delta is zero."""
        ref = self.portfolio([]).reference
        rows = self.engine.formulas([m['expr'] for m in self.members] + ['gt(ma_dist_20,0)'], report=False)
        proposals = admission.proposals(self.engine, rows, ref, precision='f64')
        weights = cp.asarray([p['weight'] for p in proposals])
        _, rr, _, dd = self.engine.returns(capped(ref.U[None] + weights[:, None, None] * cp.stack([r['_W'] for r in rows])))
        feasible, utility, delta = weight_metrics(rr, dd, ref, self.engine.ppy)
        for p, f, u, d in zip(proposals, cp.asnumpy(feasible), cp.asnumpy(utility), cp.asnumpy(delta)):
            self.assertEqual(bool(f), p['passed'], p['vetoes'])
            self.assertAlmostEqual(u, p['utility'], delta=1e-9)
            np.testing.assert_allclose(d, [p['checks'][k][1] for k in admission.KEYS], atol=1e-9)
        _, _, itself = weight_metrics(ref.rr[None], cp.asarray([ref.maxdd]), ref, self.engine.ppy)
        self.assertLess(float(cp.abs(itself).max()), 1e-12)

    def test_cash_correlation_freeze_and_behavior_dedup(self):
        archive = Archive()
        portfolio = self.portfolio()
        evaluator = Evaluator(portfolio, archive)
        results = evaluator.evaluate(['sub(ma_dist_20,ma_dist_20)', 'gt(ma_dist_20,0)'])
        self.assertTrue(results[0]['g1'])
        self.assertFalse(results[0]['passed'])
        archive.update(evaluator.pending)
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            evaluator.evaluate(['gt(ma_dist_20,0)'])
        second = Evaluator(portfolio, archive)
        duplicate = second.evaluate(['gt(ma_dist_20,0)'])[0]
        self.assertTrue(duplicate['duplicate'])
        self.assertFalse(duplicate['passed'])
        self.assertEqual(len(second.context()), 16)

    def test_real_pass_accept_and_reference_change(self):
        expr = self.members[0]['expr']
        portfolio = self.portfolio([])
        evaluator = Evaluator(portfolio, Archive(), rho_max=1.)
        result = evaluator.evaluate([expr])[0]
        self.assertTrue(result['g2'], result)
        self.assertTrue(result['passed'], result)
        self.assertEqual(result['backend'], 'cuda-f64')
        before = evaluator.ref
        accepted = evaluator.accept()
        self.assertIsNotNone(accepted)
        # The candidate was judged as an addition to the equal-weight seed, so the seed stays a member.
        self.assertEqual(portfolio.library.members, [SEED, dict(expr=expr, weight=accepted['weight'])])
        after = portfolio.reference
        ok, _, _ = weight_metrics(after.rr[None], cp.asarray([after.maxdd]), before, self.engine.ppy)
        self.assertTrue(bool(ok[0]))
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            evaluator.accept()

    def test_real_reweight(self):
        portfolio = self.portfolio()
        evaluator = Evaluator(portfolio, Archive())
        before = evaluator.ref
        result = evaluator.reweight(iterations=1, count=8, seed=25)
        self.assertGreater(result['evaluations'], 8)
        if result['changed']:
            stored = portfolio.reference
            ok, _, _ = weight_metrics(stored.rr[None], cp.asarray([stored.maxdd]), before, self.engine.ppy)
            self.assertTrue(bool(ok[0]))

    def test_member_cache_and_mining_path_are_bitwise_the_report_path(self):
        exprs = [m['expr'] for m in self.members]
        fresh = self.engine.formulas(exprs)
        lean = self.engine.formulas(exprs, report=False)
        for expr, (book, rr), a, b in zip(exprs, self.engine.members(exprs), fresh, lean):
            self.assertTrue(bool(cp.array_equal(book, a['_W'])) and bool(cp.array_equal(rr, a['_rr'])), expr)
            self.assertTrue(bool(cp.array_equal(b['_W'], a['_W'])) and bool(cp.array_equal(b['_rr'], a['_rr'])), expr)
            self.assertEqual(b['_valid_rr'], a['_valid_rr'])
            self.assertIs(self.engine.members([expr])[0][0], book)          # second request: the same device array

    def test_null_shift_delays_the_signal_and_keeps_it_causal(self):
        expr = 'gt(ts_delta(btc_fees,7),0)'                                   # market-wide: in or out on the same days for every asset
        lag = null_lag(expr)
        real = self.engine.formulas([expr], report=False)[0]['_W'].sum(1) > 0
        first = self.engine.formulas([expr], report=False, null_shift=True)[0]['_W']
        again = self.engine.formulas([expr], report=False, null_shift=True)[0]['_W']
        self.assertTrue(bool(cp.array_equal(first, again)))
        row0 = self.engine.start
        delayed = first.sum(1) > 0
        self.assertTrue(bool(real[row0:].any()) and not bool(real[row0:].all()))
        self.assertTrue(bool((delayed[row0 + lag:] == real[row0:-lag]).all()))   # day t holds what day t-lag decided, never later data

    def test_single_gate_is_the_treynor_black_frontier(self):
        reference = Reference.build(self.engine, self.members[:1])
        rows = self.engine.formulas([m['expr'] for m in self.members[1:]] + ['gt(ts_delta(btc_fees,7),0)'], report=False)
        R = cp.stack([r['_rr'] for r in rows])
        sharpe, _, years, rho, appraisal = single.measures(self.engine, R, reference)
        best = cp.zeros(len(R))
        for w in cp.linspace(0, 40, 4001):                                   # frictionless mixes P + w*A: the best one is the frontier
            mix = reference.rr[None] + w * R
            best = cp.maximum(best, mix.mean(1) / mix.std(1) * self.engine.ppy ** .5)
        frontier = cp.sqrt(reference.sharpe ** 2 + cp.maximum(appraisal, 0) ** 2)
        self.assertLess(float(cp.abs(best - frontier).max()), 2e-3, (best, frontier))
        self.assertEqual(years.shape, (len(R), single.CFG.recent_years))
        kappa = single.required_appraisal(reference, -.4, .5)
        self.assertAlmostEqual(kappa, -.4 + (((reference.sharpe + CFG.min_sharpe_gain) ** 2 - reference.sharpe ** 2) ** .5 + .4) / .5)
        self.assertEqual(single.required_appraisal(reference, -.4, 0.), float('inf'))    # no spread beyond noise: closed
        needed = rho * reference.sharpe + kappa * cp.sqrt(1 - rho * rho)
        self.assertTrue(bool(((sharpe >= needed) == (appraisal >= kappa)).all()))

    def test_single_gate_never_falls_as_correlation_rises(self):
        n = CFG.max_members
        grid = cp.linspace(-.5, 1, 301)
        for sharpe, kappa in ((1.049, .692), (2.264, 3.37), (1.5, .1)):
            reference = SimpleNamespace(sharpe=sharpe)
            with self.subTest(sharpe=sharpe, kappa=kappa):
                both = single.required_sharpe(grid, grid, reference, kappa)                    # equal-weight start: one correlation
                known = single.required_sharpe(grid, cp.full_like(grid, -1), reference, kappa)  # the portfolio part alone
                for curve in (both, known):
                    self.assertTrue(bool((cp.diff(curve) >= -1e-12).all()), curve)
                # no shorting the portfolio: past rho* the requirement stays at the peak, the best mix is the candidate alone
                self.assertAlmostEqual(float(known[-1]), (sharpe ** 2 + kappa ** 2) ** .5)
        # the anchor: exchangeable alphas at correlation 0.7 need Sharpe 1.5, together they build the target
        c, a = single.CFG.anchor_correlation, single.CFG.anchor_sharpe
        at = single.required_sharpe(cp.asarray([c]), cp.asarray([c]), SimpleNamespace(sharpe=1.049), .692)
        self.assertAlmostEqual(float(at[0]), a)
        self.assertAlmostEqual(a * (n / (1 + (n - 1) * c)) ** .5, single.pool_target())
        self.assertTrue(bool(cp.isinf(single.required_sharpe(grid, grid, SimpleNamespace(sharpe=1.049), float('inf'))).all()))
        # mean correlation with the members: a member against itself and an all-cash book
        rng = cp.random.default_rng(3)
        members = rng.standard_normal((2, 500))
        rho = single.mean_correlation(cp.stack((members[0], cp.zeros(500))), members)
        self.assertAlmostEqual(float(rho[0]), float((1 + cp.corrcoef(members)[0, 1]) / 2))
        self.assertEqual(float(rho[1]), 0.)

    def test_single_objective_rewards_and_admission(self):
        portfolio = self.portfolio([])
        with patch.object(single, 'CFG', replace(single.CFG, calibration_sample=512)):     # a small population keeps the test short
            evaluator = single.SingleEvaluator(portfolio, Archive(), rho_max=1.)
        self.addCleanup(single._POPULATION.clear)
        exprs = [m['expr'] for m in self.members] + ['sub(ma_dist_20,ma_dist_20)', 'gt(ma_dist_20,0)']
        reports = evaluator.evaluate(exprs)
        kappa = evaluator.kappa
        self.assertTrue(0 < evaluator.gate['shrinkage'] < 1 and evaluator.gate['prior_mean'] < 0, evaluator.gate)
        json.dumps(evaluator.gate, allow_nan=False)
        for r in reports:
            self.assertTrue(r['g1'], r)
            worth = r['sharpe'] >= r['required_sharpe'] and r['sharpe'] != 0
            self.assertEqual(r['g2'], worth, r)                               # G2: Sharpe at the requirement of its correlation
            sp, rho_p = evaluator.ref.sharpe, r['rho_portfolio']
            peak = (sp * sp + kappa * kappa) ** .5
            known = rho_p * sp + kappa * (1 - rho_p ** 2) ** .5 if rho_p * peak <= sp else peak
            self.assertGreaterEqual(r['required_sharpe'], known - 1e-9, r)      # never looser than the portfolio part
            self.assertEqual(r['g3'], worth and all(y > single.CFG.recent_year_sharpe for y in r['recent_years']), r)   # G3: and every recent year
            self.assertTrue(1 <= r['log_reward'] <= 4 if r['passed'] else -10 <= r['log_reward'] <= -6, r)
            self.assertEqual(len(r['recent_years']), single.CFG.recent_years)
            json.dumps(r, allow_nan=False)
        # The dashboard funnel is nested: sound books > appraisal ratio at the gate > every recent year as well.
        counters = evaluator.counters
        self.assertEqual((counters['g2'], counters['g3']), (sum(r['g2'] for r in reports), sum(r['g3'] for r in reports)))
        self.assertTrue(counters['g1'] >= counters['g2'] >= counters['g3'] >= counters['low_corr'], counters)
        self.assertEqual((evaluator.gate['recent_years'], evaluator.gate['recent_year_sharpe']),
                         (single.CFG.recent_years, single.CFG.recent_year_sharpe))
        self.assertEqual(reports[len(self.members)]['utility'], 0.)          # an all-cash book scores nothing on any objective
        accepted = evaluator.accept()
        if any(r['passed'] for r in reports):
            self.assertEqual(portfolio.library.members, [SEED, dict(expr=accepted['expr'], weight=1.)])
            self.assertEqual(accepted['utility'], max(r['utility'] for r in reports if r['passed']))
        else:
            self.assertIsNone(accepted)

    def test_generated_operators_fit_device(self):
        policy = Policy(self.engine.vocabulary, 1, 13, ExprConfig().max_cost, 2, (MAX_WINDOW, MAX_PAIR_WINDOW))
        for expr in policy.sample(256, 4, [0.] * 16)['expressions']:
            parsed = validate(expr, self.engine.vocabulary)
            self.assertTrue(parsed['ok'], parsed)
            self.assertEqual(unsupported(parsed['tree']), [], expr)

    def test_store_compare_and_swap(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'library.json'
            path.write_bytes(self.stored)
            first, second = Library.load(path), Library.load(path)
            first.members = first.members[:1]
            first.save()
            with self.assertRaisesRegex(RuntimeError, 'changed on disk'):
                second.save()
            self.assertFalse(path.with_suffix('.lock').exists())


if __name__ == '__main__':
    unittest.main()
