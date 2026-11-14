"""US portfolio use cases on an isolated store, with real GPU execution."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import cupy as cp

from foundry.portfolio import admission
from foundry.portfolio.library import Library, Member
from foundry.portfolio.portfolio import Portfolio
from foundry.portfolio.allocation import cap_book
from foundry.candidates import Formula
from foundry.research_types import Candidate, ContextConflict
from foundry.adapters.serialization import evaluation_result
from foundry.service import schema
from jsonschema import validate as validate_schema
from fixtures import EXPRESSIONS, research_for, runtime_for


class AdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = runtime_for('us_equity')
        cls.engine = cls.runtime.engine

    @classmethod
    def tearDownClass(cls):
        cls.runtime.close()

    def test_structure_and_invalid_values_stop_at_g1(self):
        rows = self.engine.evaluate(('unknown(ma_dist_20)', 'div(ma_dist_20,0)', 'ts_rank(ma_dist_20,120)'))
        verdicts = research_for(self.runtime).portfolio.admit(rows)
        self.assertFalse(rows[0].ok)
        self.assertEqual([v.stopped_at for v in verdicts[:2]], ['G1', 'G1'])
        self.assertTrue(rows[2].ok)

    def test_query_keeps_state_and_commit_is_atomic(self):
        research = research_for(self.runtime)
        candidates = tuple(Candidate(str(i), Formula(expression)) for i, expression in enumerate(EXPRESSIONS))
        before = research.context
        result = research.evaluate(candidates, before.id)
        self.assertEqual(research.context, before)
        self.assertEqual(research.store.read(self.engine.study).members, ())
        self.assertTrue(all(row.row.book is None and row.row.returns is None for row in result.rows))
        passing = tuple(c for c, r in zip(candidates, result.rows) if r.verdict.passed)
        self.assertTrue(passing, [(r.row.source, r.verdict.message) for r in result.rows])
        original = research.portfolio
        with patch.object(research.store, 'commit', side_effect=ContextConflict('concurrent writer')):
            with self.assertRaises(ContextConflict):
                research.commit(passing[:1], before.id)
        self.assertIs(research.portfolio, original)
        self.assertEqual(research.context, before)
        committed = research.commit(passing[:1], before.id)
        self.assertEqual(committed.before, before)
        self.assertNotEqual(committed.context.portfolio, before.portfolio)
        self.assertTrue(committed.rows[0].accepted)
        validate_schema(evaluation_result(committed), schema.describe()['responses']['research.commit'])
        self.assertEqual(research.store.read(self.engine.study), research.portfolio.library)
        with self.assertRaises(ContextConflict):
            research.evaluate(candidates, before.id)
        with self.assertRaises(ContextConflict):
            research.commit(passing[:1], before.id)
        after = research.portfolio.reference
        row = committed.rows[0]
        expected = self.engine.metrics([cap_book(original.reference.U + row.weight * self.engine.evaluate([row.row.source.expression])[0].book, self.engine.allocation, cp)])[0]
        cp.testing.assert_allclose(after.rr, expected.returns, atol=1e-12, rtol=1e-12)

    def test_precision_budget_and_threshold_sides(self):
        research = research_for(self.runtime)
        rows = self.engine.evaluate(EXPRESSIONS)
        ref = research.portfolio.reference
        exact = admission.proposals(self.engine, rows, ref, 'f64')
        for row, proposal in zip(rows, exact):
            gain = proposal.wsharpe - ref.wsharpe
            for offset in (-1e-7, 1e-7):
                boundary = replace(ref, cfg=replace(ref.cfg, min_sharpe_gain=gain + offset),
                                   precision_errors={})
                a = admission.proposals(self.engine, [row], boundary, 'f64')[0]
                b = admission.proposals(self.engine, [row], boundary, 'paired')[0]
                self.assertEqual((a.passed, a.weight), (b.passed, b.weight))
        with patch.object(self.engine, 'cuda_cfg', replace(self.engine.cuda_cfg, mixed_return_atol=1e-14)):
            strict = admission.proposals(self.engine, rows, ref, 'auto')
        self.assertTrue(all(p.precision.refined for p in strict))
        self.assertEqual([p.weight for p in strict], [p.weight for p in exact])
        mixed = self.engine.evaluate(EXPRESSIONS[:1], precision='mixed')
        with self.assertRaises(AssertionError):
            research.portfolio.admit(mixed)

    def test_capacity_is_a_business_rejection(self):
        members = (Member(Formula(EXPRESSIONS[0]), .5),)
        portfolio = Portfolio(self.engine, Library(members, self.engine.study))
        ref = replace(portfolio.reference, cfg=replace(portfolio.reference.cfg, max_members=1))
        row = self.engine.evaluate((EXPRESSIONS[1],))[0]
        proposal = admission.proposals(self.engine, [row], ref, 'f64')[0]
        self.assertFalse(proposal.passed)
        self.assertFalse(next(c.passed for c in proposal.checks if c.name == 'capacity'))


if __name__ == '__main__':
    unittest.main()
