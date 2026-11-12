"""Portfolio admission on the real panel and the production library (local: needs the GPU and data/qnt_crypto).
Nothing here writes the production library."""
import unittest
from dataclasses import replace
from unittest.mock import patch

import cupy as cp

from foundry.config import CudaConfig
from foundry.engine import Engine
from foundry.portfolio import admission
from foundry.portfolio.library import STORE, Library
from foundry.portfolio.portfolio import Portfolio
from foundry.portfolio.reference import CFG, capped


class AdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = Engine()
        cls.stored = STORE.read_bytes()
        cls.expr = Library.load().members[0]['expr']

    def tearDown(self):
        self.assertEqual(STORE.read_bytes(), self.stored)

    def test_structure_causality_and_invalid_values_stop_at_g1(self):
        rows = self.engine.formulas(['ts_rank(log(abs(ts_mean(ma_dist_20,7))),120)', 'div(ma_dist_20, 0)', 'ts_rank(ma_dist_20,30)'])
        self.assertIn('非因果', rows[0]['error'])
        verdicts = Portfolio(self.engine, Library([])).admit(rows)
        self.assertEqual([v['stopped_at'] for v in verdicts[:2]], ['G1', 'G1'])
        self.assertTrue(rows[2]['ok'])

    def test_accept_rebuild_stale_and_duplicate(self):
        pf = Portfolio(self.engine, Library([]))
        ref = pf.reference
        r = self.engine.formulas([self.expr])[0]
        g = pf.admit([r])[0]
        self.assertTrue(g['passed'], admission.explain(g))
        weight = pf.accept(r, g)
        after = pf.reference
        expected = self.engine.metrics([capped(ref.U + weight * r['_W'])])[0]
        cp.testing.assert_allclose(after.rr, expected['rr'], atol=1e-12, rtol=1e-12)
        self.assertAlmostEqual(after.wsharpe, g['combo']['sharpe_after'], places=10)
        self.assertEqual(pf.library.members[0]['expr'], '__equal_weight__')              # judged as an addition to it
        self.assertAlmostEqual(sum(m['weight'] for m in pf.library.members), 1 + weight)  # additive scales, not shares
        with self.assertRaises(AssertionError):                                          # a verdict on the old reference
            pf.accept(r, g)
        clone = dict(r, expr='duplicate', _W=after.W.copy(), _rr=after.rr.copy())
        self.assertFalse(pf.admit([clone])[0]['passed'])

    def test_precision_budget_and_gate_boundary(self):
        pf = Portfolio(self.engine, Library([]))
        ref = pf.reference
        r = self.engine.formulas([self.expr])[0]
        exact = admission.proposals(self.engine, [r], ref, 'f64')[0]
        self.assertTrue(exact['passed'])
        # Move the gain threshold across a real candidate's decision boundary: auto must refine both sides.
        gain = exact['wsharpe'] - ref.wsharpe
        for threshold in (gain - 1e-7, gain + 1e-7):
            with patch.object(admission, 'CFG', replace(CFG, min_sharpe_gain=threshold)):
                a = admission.proposals(self.engine, [r], ref, 'f64')[0]
                b = admission.proposals(self.engine, [r], ref, 'auto')[0]
            self.assertEqual(a['passed'], b['passed'])
            self.assertTrue(b['precision']['refined'])
            self.assertEqual(b['backend'], 'cuda-f64')
        # A stricter accuracy requirement selects FP64 whatever the card.
        with patch.object(admission, 'CUDA', CudaConfig(mixed_return_atol=1e-14)):
            b = admission.proposals(self.engine, [r], ref, 'auto')[0]
        self.assertTrue(b['precision']['over_budget'] and b['precision']['refined'])
        self.assertEqual(exact['weight'], b['weight'])
        mixed = self.engine.formulas([self.expr], precision='mixed')[0]
        with self.assertRaises(AssertionError):
            pf.admit([mixed])
        verdict = pf.admit([r])[0]
        verdict['layers']['G2_performance']['backend'] = 'cuda-mixed'
        with self.assertRaises(AssertionError):
            pf.accept(r, verdict)

    def test_full_library_admits_no_new_member(self):
        members = Library.load().members
        pf = Portfolio(self.engine, Library(members))
        r = self.engine.formulas(['gt(ts_chanpos(ma_dist_60,28),0.98)'])[0]
        with patch.object(admission, 'CFG', replace(CFG, max_members=len(members))):
            p = admission.proposals(self.engine, [r], pf.reference, 'f64')[0]
        self.assertFalse(p['passed'])
        self.assertIn('组合成员数超过上限', p['vetoes'])


if __name__ == '__main__':
    unittest.main()
