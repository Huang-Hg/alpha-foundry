"""CUDA production regression tests; no persistent portfolio writes."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'tmp/q25_funnel'),str(REPO)]
import engine
import cupy as cp
import gates
import combolib
from combolib import ComboLib,STORE
from combo_gpu import capped


class ComboGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stored=STORE.read_bytes()
        cls.expr=json.loads(cls.stored)['members'][0]['expr']

    def test_causality_and_invalid_inputs(self):
        bad='ts_rank(log(abs(ts_mean(ma_dist_20,7))),120)'
        self.assertIsNotNone(gates.g1_causality(bad))
        self.assertIsNone(gates.g1_causality('ts_rank(ma_dist_20,30)'))
        self.assertFalse(engine.simulate(bad)['ok'])
        for value in (cp.nan,cp.inf):
            def alpha(data,liquid):return cp.full(liquid.shape,value)
            row=engine.simulate_python(alpha)
            self.assertFalse(row['ok'])
            json.dumps(gates.rowify(row,gates.run(row)),allow_nan=False)
        def leaking(data,liquid):return cp.broadcast_to(data['close'][-1:],liquid.shape)
        self.assertIn('前视',engine.simulate_python(leaking)['error'])

    def test_accept_rebuild_stale_and_duplicate(self):
        lib=ComboLib([]);ref=lib.ref()
        r=engine.simulate(self.expr);g=gates.run(r,ref)
        self.assertTrue(g['passed'],gates.explain(g))
        json.dumps(gates.rowify(r,g),allow_nan=False)
        weight=lib.add(r,g);after=lib.ref()
        expected=engine._metrics(capped(ref['U']+weight*r['_W']),ref['win'])
        cp.testing.assert_allclose(after['rr'],expected['rr'],atol=1e-12,rtol=1e-12)
        self.assertAlmostEqual(after['wsharpe'],g['combo']['sharpe_after'],places=10)
        self.assertAlmostEqual(sum(m['weight'] for m in lib.members),1+weight)      # additive scales, not shares
        with self.assertRaises(AssertionError):lib.add(r,g)
        clone=dict(r,expr='duplicate',_W=after['W'].copy(),_rr=after['rr'].copy())
        self.assertFalse(gates.run(clone,after)['passed'])
        self.assertEqual(STORE.read_bytes(),self.stored)

    def test_production_has_no_cpu_backtest(self):
        import rl.backtest_reward as cpu
        from daemon import Server
        server=Server.__new__(Server);lib=ComboLib.load()
        server._combo_fresh=lambda:lib;server._tick=lambda *a:None
        with patch.object(cpu,'run_bt',side_effect=AssertionError('CPU production execution')):
            rows=server._eval_exprs([self.expr,'gt(ma_dist_20,0)'],'is',False)
        self.assertEqual(len(rows),2)
        self.assertTrue(all(r['combo']['cpu_trials']==0 for r in rows))
        self.assertTrue(all(r['backend'] in ('cuda-f64','cuda-mixed') for r in rows))
        self.assertTrue(all(r['combo']['precision']['mode']=='auto' for r in rows))

    def test_python_source_rebuild(self):
        source=("from gpu import eval_node\nfrom fastexpr import validate\n"
                "def alpha_test(data,liquid):\n"
                f"    return eval_node(validate({self.expr!r},list(data))['tree'].root,data,{{}})\n")
        with tempfile.TemporaryDirectory(dir=REPO/'tmp') as td:
            base=Path(td)
            with patch.object(combolib,'HERE',base),patch.object(combolib,'STORE',base/'combo.json'):
                lib=ComboLib([])
                rows=combolib.evaluate_python_source(source,'test.py','is',lib,accept=True)
                self.assertTrue(rows[0]['accepted'],rows[0])
                cp.testing.assert_allclose(ComboLib.load().ref()['rr'],lib.ref()['rr'],atol=1e-12,rtol=1e-12)
        self.assertEqual(STORE.read_bytes(),self.stored)

    def test_precision_budget_and_gate_boundary(self):
        from combo_gpu import proposal_many,simulate_many
        lib=ComboLib([]);ref=lib.ref();r=engine.simulate(self.expr)
        exact=proposal_many([r],ref,precision='f64')[0]
        self.assertTrue(exact['passed'])
        # Change the gain threshold across an actual candidate's decision
        # boundary. Auto must refine both sides, including the rejection side.
        gain=exact['wsharpe']-ref['wsharpe']
        for threshold in (gain-1e-7,gain+1e-7):
            with patch.object(gates.CFG,'min_sharpe_gain',threshold):
                a=proposal_many([r],ref,precision='f64')[0]
                b=proposal_many([r],ref,precision='auto')[0]
            self.assertEqual(a['passed'],b['passed'])
            self.assertTrue(b['precision']['refined'])
            self.assertEqual(b['backend'],'cuda-f64')
        # A stricter accuracy requirement selects FP64 regardless of GPU model.
        with patch.object(engine.CUDA,'mixed_return_atol',1e-14):
            b=proposal_many([r],ref,precision='auto')[0]
        self.assertTrue(b['precision']['over_budget'])
        self.assertTrue(b['precision']['refined'])
        self.assertEqual(exact['weight'],b['weight'])
        mixed=simulate_many([self.expr],precision='mixed')[0]
        with self.assertRaises(ValueError):gates.run(mixed,ref)
        verdict=gates.run(r,ref)
        verdict['layers']['G2_performance']['backend']='cuda-mixed'
        with self.assertRaises(AssertionError):lib.add(r,verdict)
        self.assertEqual(STORE.read_bytes(),self.stored)


if __name__=='__main__':unittest.main(verbosity=2)
