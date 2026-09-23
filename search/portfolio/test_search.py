"""Native algorithm invariants and real CUDA portfolio integration tests."""
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch

from .native import Policy, CEM, ROOT
sys.path.insert(0, str(ROOT/'tmp/q25_funnel'))


class NativeTests(unittest.TestCase):
    def setUp(self):
        self.policy = Policy(['x', 'y', 'z'], seed=25)

    def tearDown(self):
        self.policy.close()

    def test_grammar_cost_and_causality(self):
        from fastexpr import validate
        batch = self.policy.sample(256, 4, [0.]*16)
        self.assertEqual(len(batch['expressions']), 256+12*batch['probe_count'])
        for expr in batch['expressions']:
            v = validate(expr, ['x','y','z'])
            self.assertTrue(v['ok'], (expr,v))
            self.assertLessEqual(v['cost'], 25)
            self.assertLessEqual(len(re.findall(r'[A-Za-z_]\w*|-?\d+\.?\d*', expr)), 13, expr)
            def check(node):
                """Time-series operators stacked on the deepest path below (and including) node."""
                name = str(getattr(node.op, 'value', ''))
                if name == 'ts_rank':
                    self.assertLessEqual(node.window, 31)
                return name.startswith('ts_') + max((check(child) for child in node.children), default=0)
            self.assertLessEqual(check(v['tree'].root), 2, expr)

    def test_tb_gradient_learning_and_stale_batch(self):
        batch = self.policy.sample(32, 0, [0.]*16)
        rewards = [-6.]*32
        self.assertLess(self.policy.gradient_check(rewards)['max_gradient_error'], 1e-5)
        learned = self.policy.train(rewards, batch['epoch'], opd=0)
        self.assertLess(learned['loss_after'], learned['loss_before'])
        with self.assertRaisesRegex(RuntimeError, 'stale'):
            self.policy.train(rewards, batch['epoch'])

    def test_reliable_teacher_and_ties(self):
        batch = self.policy.sample(32, 2, [0.]*16)
        self.assertGreater(batch['probe_count'], 0)
        rewards = [-6.]*32 + [1.,3.,-2.]*4*batch['probe_count']
        learned = self.policy.train(rewards, batch['epoch'])
        self.assertEqual(learned['teachers'], batch['probe_count'])
        batch = self.policy.sample(32, 2, [0.]*16)
        learned = self.policy.train([-6.]*len(batch['expressions']), batch['epoch'])
        self.assertEqual(learned['teachers'], 0)

    def test_checkpoint_restores_rng_and_rejects_grammar_mismatch(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'policy.bin'
            self.policy.checkpoint(path)
            first = self.policy.sample(32, 1, [0.]*16)
            self.policy.checkpoint(path, load=True)
            second = self.policy.sample(32, 1, [0.]*16)
            self.assertEqual(first['expressions'], second['expressions'])
            other = Policy(['different'])
            try:
                with self.assertRaisesRegex(RuntimeError, 'grammar mismatch'):
                    other.checkpoint(path, load=True)
            finally:
                other.close()


class CudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from . import evaluator as ev
        cls.ev = ev
        cls.stored = ev.combolib.STORE.read_bytes()

    def tearDown(self):
        self.assertEqual(self.ev.combolib.STORE.read_bytes(), self.stored)

    def test_return_trades_against_drawdown_but_sharpe_never(self):
        from combo_gpu import return_requirements
        cp=self.ev.cp
        need=cp.asnumpy(return_requirements(cp.asarray([-.3,-.33,-.27]),dict(maxdd=-.3,cagr=1.4,recent_growth=-.2)))
        self.assertTrue((need[:,0]==self.ev.gates.CFG.min_sharpe_gain).all() and (need[:,1]==0).all())
        self.assertEqual(need[0,2:].tolist(),[0.,0.])                         # same drawdown: return must not fall
        self.assertAlmostEqual(need[1,2],.14); self.assertAlmostEqual(need[1,3],.02)   # 10% deeper: 10% more return
        self.assertAlmostEqual(need[2,2],-.14); self.assertAlmostEqual(need[2,3],-.02) # 10% shallower: 10% may go

    def test_cash_correlation_freeze_and_behavior_dedup(self):
        ev=self.ev
        archive=ev.Archive()
        lib=ev.combolib.ComboLib.load()
        evaluator=ev.Evaluator(lib,archive)
        results=evaluator.evaluate(['sub(ma_dist_20,ma_dist_20)','gt(ma_dist_20,0)'])
        self.assertTrue(results[0]['g1'])
        self.assertFalse(results[0]['passed'])
        archive.update(evaluator.pending)
        with self.assertRaisesRegex(RuntimeError,'changed'):
            evaluator.evaluate(['gt(ma_dist_20,0)'])
        second=ev.Evaluator(lib,archive)
        duplicate=second.evaluate(['gt(ma_dist_20,0)'])[0]
        self.assertTrue(duplicate['duplicate'])
        self.assertFalse(duplicate['passed'])
        self.assertEqual(len(second.context()),16)

    def test_real_pass_accept_and_reference_change(self):
        ev=self.ev
        expr=json.loads(self.stored)['members'][0]['expr']
        lib=ev.combolib.ComboLib([])
        evaluator=ev.Evaluator(lib,ev.Archive(),rho_max=1.)
        result=evaluator.evaluate([expr])[0]
        self.assertTrue(result['g2'],result)
        self.assertTrue(result['passed'],result)
        self.assertEqual(result['backend'],'cuda-f64')
        before=evaluator.ref
        accepted=evaluator.accept()
        self.assertIsNotNone(accepted)
        after=lib.ref()
        ok,_,_=ev.weight_metrics(after['rr'][None],ev.cp.asarray([after['maxdd']]),before)
        self.assertTrue(bool(ok[0]))
        with self.assertRaisesRegex(RuntimeError,'changed'):
            evaluator.accept()

    def test_cem_additive_weights_and_real_reweight(self):
        ev=self.ev; cp=ev.cp
        sampler=CEM([0.,0.,0.])
        data=cp.asarray(sampler.sample(64,25)).view(cp.float64).reshape(64,3)
        self.assertTrue(bool(((data>0)&(data<1)).all()))
        self.assertGreater(float(cp.abs(data.sum(1)-1).max()),.1)          # independent scales, not shares
        sampler.update(cp.asnumpy(data[:,0]).tolist(),16)
        self.assertGreater(sampler.mean[0],sampler.mean[1])
        lib=ev.combolib.ComboLib.load()
        evaluator=ev.Evaluator(lib,ev.Archive())
        before=evaluator.ref
        result=evaluator.reweight(iterations=1,count=8,seed=25)
        self.assertGreater(result['evaluations'],8)
        if result['changed']:
            stored=lib.ref()
            ok,_,_=ev.weight_metrics(stored['rr'][None],cp.asarray([stored['maxdd']]),before)
            self.assertTrue(bool(ok[0]))

    def test_member_cache_and_mining_path_are_bitwise_the_report_path(self):
        ev=self.ev; cp=ev.cp
        members=json.loads(self.stored)['members']
        fresh=ev.engine.simulate_many([m['expr'] for m in members])
        lean=ev.engine.simulate_many([m['expr'] for m in members],report=False)
        for m,(book,rr),a,b in zip(members,ev.combolib.member_states(members,'is'),fresh,lean):
            self.assertTrue(bool(cp.array_equal(book,a['_W'])) and bool(cp.array_equal(rr,a['_rr'])),m['expr'])
            self.assertTrue(bool(cp.array_equal(b['_W'],a['_W'])) and bool(cp.array_equal(b['_rr'],a['_rr'])),m['expr'])
            self.assertEqual(b['_valid_rr'],a['_valid_rr'])
            self.assertIs(ev.combolib.member_book(m,'is'),book)              # second request: the same device array

    def test_null_shift_delays_the_signal_and_keeps_it_causal(self):
        ev=self.ev; cp=ev.cp
        expr='gt(ts_delta(btc_fees,7),0)'                                  # market-wide: in or out on the same days for every asset
        lo,hi=ev.engine.NULL_LAG
        lag=lo+int.from_bytes(hashlib.sha256(expr.encode()).digest()[:8],'big')%(hi-lo+1)
        real=ev.engine.simulate_many([expr],report=False)[0]['_W'].sum(1)>0
        with patch.object(ev.engine,'NULL_SHIFT',True):
            first=ev.engine.simulate_many([expr],report=False)[0]['_W']
            again=ev.engine.simulate_many([expr],report=False)[0]['_W']
        self.assertTrue(bool(cp.array_equal(first,again)))
        row0=ev.engine.load()['windows']['is']['row0']
        delayed=first.sum(1)>0
        self.assertTrue(bool(real[row0:].any()) and not bool(real[row0:].all()))
        self.assertTrue(bool((delayed[row0+lag:]==real[row0:-lag]).all()))  # day t holds what day t-lag decided, never later data

    def test_single_gate_is_the_treynor_black_frontier(self):
        from . import single
        ev=self.ev; cp=ev.cp
        members=json.loads(self.stored)['members']
        reference=ev.combolib.ComboLib(members[:1]).ref('is')
        rows=[r for r in ev.engine.simulate_many([m['expr'] for m in members[1:]]+['gt(ts_delta(btc_fees,7),0)'],report=False)]
        R=cp.stack([r['_rr'] for r in rows])
        sharpe,_,years,rho,appraisal=single.measures(R,reference)
        best=cp.zeros(len(R))
        for w in cp.linspace(0,40,4001):                                  # frictionless mixes P + w*A: the best one is the frontier
            mix=reference['rr'][None]+w*R
            best=cp.maximum(best,mix.mean(1)/mix.std(1)*365**.5)
        frontier=cp.sqrt(reference['sharpe']**2+cp.maximum(appraisal,0)**2)
        self.assertLess(float(cp.abs(best-frontier).max()),2e-3,(best,frontier))
        self.assertEqual(years.shape,(len(R),single.CFG.recent_years))
        kappa=single.required_appraisal(reference,-.4,.5)
        self.assertAlmostEqual(kappa,-.4+(((reference['sharpe']+ev.gates.CFG.min_sharpe_gain)**2-reference['sharpe']**2)**.5+.4)/.5)
        self.assertEqual(single.required_appraisal(reference,-.4,0.),float('inf'))       # no spread beyond noise: closed
        needed=rho*reference['sharpe']+kappa*cp.sqrt(1-rho*rho)
        self.assertTrue(bool(((sharpe>=needed)==(appraisal>=kappa)).all()))

    def test_single_gate_never_falls_as_correlation_rises(self):
        from . import single
        cp=self.ev.cp; n=self.ev.gates.CFG.max_members
        grid=cp.linspace(-.5,1,301)
        for reference,kappa in ((dict(sharpe=1.049),.692),(dict(sharpe=2.264),3.37),(dict(sharpe=1.5),.1)):
            with self.subTest(reference=reference,kappa=kappa):
                both=single.required_sharpe(grid,grid,reference,kappa)                 # equal-weight start: one correlation
                known=single.required_sharpe(grid,cp.full_like(grid,-1),reference,kappa)   # the portfolio part alone
                for curve in (both,known):
                    self.assertTrue(bool((cp.diff(curve)>=-1e-12).all()),curve)
                # no shorting the portfolio: past rho* the requirement stays at the peak, the best mix is the candidate alone
                self.assertAlmostEqual(float(known[-1]),(reference['sharpe']**2+kappa**2)**.5)
        # the anchor: exchangeable alphas at correlation 0.7 need Sharpe 1.5, together they build the target
        c,a=single.CFG.anchor_correlation,single.CFG.anchor_sharpe
        at=single.required_sharpe(cp.asarray([c]),cp.asarray([c]),dict(sharpe=1.049),.692)
        self.assertAlmostEqual(float(at[0]),a)
        self.assertAlmostEqual(a*(n/(1+(n-1)*c))**.5,single.pool_target())
        self.assertTrue(bool(cp.isinf(single.required_sharpe(grid,grid,dict(sharpe=1.049),float('inf'))).all()))
        # mean correlation with the members: a member against itself and an all-cash book
        rng=cp.random.default_rng(3); members=rng.standard_normal((2,500))
        rho=single.mean_correlation(cp.stack((members[0],cp.zeros(500))),members)
        self.assertAlmostEqual(float(rho[0]),float((1+cp.corrcoef(members)[0,1])/2))
        self.assertEqual(float(rho[1]),0.)

    def test_single_objective_rewards_and_admission(self):
        from . import single
        ev=self.ev
        lib=ev.combolib.ComboLib([])
        with patch.object(single.CFG,'calibration_sample',512):            # a small population keeps the test short
            evaluator=single.SingleEvaluator(lib,ev.Archive(),rho_max=1.)
        self.addCleanup(single._POPULATION.clear)
        exprs=[m['expr'] for m in json.loads(self.stored)['members']]+['sub(ma_dist_20,ma_dist_20)','gt(ma_dist_20,0)']
        reports=evaluator.evaluate(exprs)
        kappa=evaluator.kappa
        self.assertTrue(0<evaluator.gate['shrinkage']<1 and evaluator.gate['prior_mean']<0,evaluator.gate)
        json.dumps(evaluator.gate,allow_nan=False)
        for r in reports:
            self.assertTrue(r['g1'],r)
            worth=r['sharpe']>=r['required_sharpe'] and r['sharpe']!=0
            self.assertEqual(r['g2'],worth,r)                              # G2: Sharpe at the requirement of its correlation
            sp,rho_p=evaluator.ref['sharpe'],r['rho_portfolio']; peak=(sp*sp+kappa*kappa)**.5
            known=rho_p*sp+kappa*(1-rho_p**2)**.5 if rho_p*peak<=sp else peak
            self.assertGreaterEqual(r['required_sharpe'],known-1e-9,r)     # never looser than the portfolio part
            self.assertEqual(r['g3'],worth and all(y>single.CFG.recent_year_sharpe for y in r['recent_years']),r)   # G3: and every recent year
            self.assertTrue(1<=r['log_reward']<=4 if r['passed'] else -10<=r['log_reward']<=-6,r)
            self.assertEqual(len(r['recent_years']),single.CFG.recent_years)
            json.dumps(r,allow_nan=False)
        # The dashboard funnel is nested: sound books > appraisal ratio at the gate > every recent year as well.
        counters=evaluator.counters
        self.assertEqual((counters['g2'],counters['g3']),(sum(r['g2'] for r in reports),sum(r['g3'] for r in reports)))
        self.assertTrue(counters['g1']>=counters['g2']>=counters['g3']>=counters['low_corr'],counters)
        self.assertEqual((evaluator.gate['recent_years'],evaluator.gate['recent_year_sharpe']),
                         (single.CFG.recent_years,single.CFG.recent_year_sharpe))
        self.assertEqual(reports[4]['utility'],0.)                        # an all-cash book scores nothing on any objective
        accepted=evaluator.accept()
        if any(r['passed'] for r in reports):
            self.assertEqual(lib.members[-1],dict(expr=accepted['expr'],lane='fastexpr',weight=1.))
            self.assertEqual(accepted['utility'],max(r['utility'] for r in reports if r['passed']))
        else:
            self.assertIsNone(accepted)

    def test_generated_operators_fit_device(self):
        from gpu import MAX_W_1, MAX_W_2, can_gpu
        from fastexpr import validate
        policy=Policy(self.ev.engine.operands(),window_caps=(MAX_W_1,MAX_W_2))
        try:
            batch=policy.sample(256,4,[0.]*16)
            for expr in batch['expressions']:
                parsed=validate(expr,self.ev.engine.operands())
                self.assertTrue(parsed['ok'],parsed)
                self.assertTrue(can_gpu(parsed['tree']),expr)
        finally:
            policy.close()

    def test_store_compare_and_swap(self):
        ev=self.ev
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'combo.json'
            path.write_bytes(self.stored)
            with patch.object(ev.combolib,'STORE',path):
                first=ev.combolib.ComboLib.load()
                second=ev.combolib.ComboLib.load()
                first.members=first.members[:1]
                first.save()
                with self.assertRaisesRegex(RuntimeError,'changed on disk'):
                    second.save()
                self.assertFalse(path.with_suffix('.lock').exists())


if __name__ == '__main__':
    unittest.main()
