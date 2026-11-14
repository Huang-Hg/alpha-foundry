"""Mining callers share research admission and keep study/version state isolated."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cupy as cp
import numpy as np

from foundry.engine import null_lag
from foundry.contracts import Study
from foundry.expr.parse import validate
from foundry.portfolio import admission
from foundry.portfolio.library import Member
from foundry.candidates import Formula
from foundry.research_types import ContextConflict
from foundry.search import single
from foundry.search.evaluator import Archive, Evaluator
from foundry.search.policy import Policy
from foundry.runtime import open_runtime
from fixtures import EXPRESSIONS, research_for, runtime_for, settings_for


class SearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = runtime_for('us_equity')
        cls.engine = cls.runtime.engine

    @classmethod
    def tearDownClass(cls):
        cls.runtime.close()

    def test_return_trades_against_drawdown_but_sharpe_never(self):
        cfg = self.engine.portfolio_cfg
        reference = SimpleNamespace(maxdd=-.3, cagr=1.4, recent_growth=-.2, xp=cp, cfg=cfg)
        need = cp.asnumpy(admission.return_requirements(cp.asarray([-.3, -.33, -.27]), reference))
        self.assertTrue((need[:, 0] == cfg.min_sharpe_gain).all() and (need[:, 1] == 0).all())
        np.testing.assert_allclose(need[:, 2:], [[0, 0], [.14, .02], [-.14, -.02]], atol=1e-15)

    def test_weight_trials_use_the_same_admission_metrics(self):
        research = research_for(self.runtime)
        ref = research.portfolio.reference
        rows = self.engine.evaluate(EXPRESSIONS, report=False)
        proposals = admission.proposals(self.engine, rows, ref, 'f64')
        for row, proposal in zip(rows, proposals):
            trial = research.trial_weights(cp.asarray([[1., proposal.weight]]),
                                                              cp.stack([ref.U, row.book]), research.context.id)
            self.assertEqual(bool(trial.feasible[0]), proposal.passed)
            self.assertAlmostEqual(float(trial.utility[0]), proposal.utility, delta=1e-9)
            np.testing.assert_allclose(cp.asnumpy(trial.delta[0]), [c.delta for c in proposal.checks], atol=1e-9)

    def test_cash_archive_freeze_and_behavior_dedup(self):
        research, archive = research_for(self.runtime), Archive()
        evaluator = Evaluator(research, archive)
        result = evaluator.evaluate(('sub(ma_dist_20,ma_dist_20)', EXPRESSIONS[0]))
        self.assertTrue(result[0].g1)
        self.assertFalse(result[0].passed)
        archive.update(evaluator.pending)
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            evaluator.evaluate(EXPRESSIONS[:1])
        second = Evaluator(research, archive)
        duplicate = second.evaluate(EXPRESSIONS[:1])[0]
        self.assertTrue(duplicate.duplicate)
        self.assertFalse(duplicate.passed)
        self.assertEqual(len(second.context()), 16)

    def test_reweight_commits_only_an_admissible_successor(self):
        research = research_for(self.runtime, (Member(Formula(EXPRESSIONS[0]), .5), Member(Formula(EXPRESSIONS[1]), .5)))
        before = research.context
        evaluator = Evaluator(research, Archive())
        result = evaluator.reweight(iterations=1, count=8, seed=25)
        self.assertGreater(result.evaluations, 8)
        self.assertEqual(research.portfolio.library, research.store.read(self.engine.study))
        if result.changed:
            self.assertNotEqual(research.context, before)
            with self.assertRaises(ContextConflict):
                research.trial_weights(cp.asarray([[.5, .5]]), cp.stack(evaluator.books), before.id)
        else:
            self.assertEqual(research.context, before)

    def test_member_cache_and_report_path_are_bitwise_equal(self):
        fresh = self.engine.evaluate(EXPRESSIONS)
        lean = self.engine.evaluate(EXPRESSIONS, report=False)
        for expr, (book, returns), first, second in zip(EXPRESSIONS, self.engine.members(tuple(Formula(e) for e in EXPRESSIONS)), fresh, lean):
            cp.testing.assert_array_equal(book, first.book)
            cp.testing.assert_array_equal(returns, first.returns)
            cp.testing.assert_array_equal(second.book, first.book)
            cp.testing.assert_array_equal(second.returns, first.returns)
            self.assertEqual(second.valid_returns, first.valid_returns)
            self.assertIs(self.engine.members([Formula(expr)])[0][0], book)

    def test_noise_control_is_forward_only_and_separate_from_real_cache(self):
        expression = EXPRESSIONS[0]
        real = self.engine.evaluate([expression], report=False)[0].book
        delayed = self.engine.evaluate([expression], report=False, null_shift=True)[0].book
        again = self.engine.evaluate([expression], report=False, null_shift=True)[0].book
        lag = null_lag(expression)
        cp.testing.assert_array_equal(again, delayed)
        self.assertFalse(bool(delayed[:lag].any()))
        start = self.engine.start + lag
        stop = len(real)
        if start < stop:
            cp.testing.assert_array_equal(delayed[start:], real[self.engine.start:stop-lag])
        self.assertFalse(bool(cp.array_equal(real, delayed)))

    def test_member_cache_cannot_cross_study_snapshots(self):
        expression = EXPRESSIONS[0]
        full_book, _ = self.engine.members([Formula(expression)])[0]
        cutoff = str(self.engine.dates[(self.engine.start+len(self.engine.dates))//2])
        study = Study('us_equity', self.engine.study.start, cutoff)
        with open_runtime(self.runtime.settings, study) as shorter:
            short_book, _ = shorter.engine.members([Formula(expression)])[0]
            self.assertIsNot(full_book, short_book)
            self.assertLess(len(short_book), len(full_book))
            self.assertNotEqual(shorter.engine.context.id, self.engine.context.id)
            other = research_for(shorter)
            with self.assertRaises(ContextConflict):
                other.evaluate_device([Formula(expression)], research_for(self.runtime).context.id)

    def test_single_reward_cannot_override_shared_admission(self):
        research = research_for(self.runtime)
        with patch.object(single, 'prior', return_value=(-.4, .5)):
            evaluator = single.SingleEvaluator(research, Archive(), rho_max=1.)
        reports = evaluator.evaluate(EXPRESSIONS)
        shared = research.evaluate_device(tuple(Formula(e) for e in EXPRESSIONS), research.context.id)
        for report, verdict in zip(reports, shared.verdicts):
            self.assertFalse(report.passed and not verdict.passed)
            self.assertEqual(report.weight, verdict.proposal.weight)
        before = research.context
        accepted = evaluator.accept()
        if accepted is not None:
            self.assertNotEqual(research.context, before)
            member = research.portfolio.library.members[-1]
            self.assertEqual((member.source.expression, member.weight), (accepted.expr, accepted.weight))

    def test_single_gate_is_monotone_and_cash_has_zero_correlation(self):
        cfg, pcfg = self.engine.settings.single, self.engine.portfolio_cfg
        grid = cp.linspace(-.5, 1, 301)
        for sharpe, kappa in ((1.049, .692), (2.264, 3.37), (1.5, .1)):
            ref = SimpleNamespace(sharpe=sharpe)
            both = single.required_sharpe(grid, grid, ref, kappa, cfg, pcfg, cp)
            known = single.required_sharpe(grid, cp.full_like(grid, -1), ref, kappa, cfg, pcfg, cp)
            for curve in (both, known):
                self.assertTrue(bool((cp.diff(curve) >= -1e-12).all()))
            self.assertAlmostEqual(float(known[-1]), (sharpe*sharpe+kappa*kappa)**.5)
        rho = cfg.anchor_correlation
        target = single.pool_target(cfg, pcfg)
        self.assertAlmostEqual(cfg.anchor_sharpe*(pcfg.max_members/(1+(pcfg.max_members-1)*rho))**.5, target)
        ref = SimpleNamespace(sharpe=1.5)
        required = single.required_appraisal(ref, -.4, .5, pcfg)
        self.assertAlmostEqual(required, -.4+(((1.5+pcfg.min_sharpe_gain)**2-1.5**2)**.5+.4)/.5)
        rng = cp.random.default_rng(3)
        members = rng.standard_normal((2, 500))
        measured = single.mean_correlation(cp.stack((members[0], cp.zeros(500))), members, cp)
        self.assertAlmostEqual(float(measured[0]), float((1+cp.corrcoef(members)[0, 1])/2))
        self.assertEqual(float(measured[1]), 0.)

    def test_generated_operators_fit_explicit_capabilities(self):
        backend = self.engine.backend
        limit = self.engine.settings.expr.max_cost
        policy = Policy(self.engine.vocabulary, 1, 13, limit, 2, (backend.max_window, backend.max_pair_window))
        for expression in policy.sample(256, 4, [0.]*16)['expressions']:
            parsed = validate(expression, self.engine.vocabulary, limit)
            self.assertTrue(parsed['ok'], parsed)
            self.assertEqual(list(backend.unsupported(parsed['tree'])), [])


class CheckpointTests(unittest.TestCase):
    def test_one_epoch_resume_and_old_protocol_rejection(self):
        from foundry.search import run
        settings = settings_for('us_equity')
        cfg = replace(settings.search, epochs=1, batch_size=16, probes=0, train_steps=1,
                      reweight_iterations=0, seed_portfolio='equal', accept=False)
        settings = replace(settings, search=cfg)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/'mining'
            with patch.object(run, 'LIVE', Path(folder)/'live.json'):
                run.run(output, settings)
                state = json.loads((output/'state.json').read_text())
                self.assertEqual(state['totals']['requested'], 16)
                self.assertEqual(state['signature']['snapshot']['study']['market'], 'us_equity')
                before = (output/'epochs.jsonl').read_bytes()
                run.run(output, settings)
                self.assertEqual((output/'epochs.jsonl').read_bytes(), before)
                self.assertEqual(json.loads((output/'status.json').read_text())['phase'], 'complete')
                state['signature']['protocol'] = 'obsolete-checkpoint-v1'
                (output/'state.json').write_text(json.dumps(state))
                with self.assertRaisesRegex(ValueError, 'mismatch'):
                    run.run(output, settings)


if __name__ == '__main__':
    unittest.main()
