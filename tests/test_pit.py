"""Publication time, universe isolation and causal CPU operators; no live data/GPU."""
import unittest
import numpy as np
from foundry.core.pit import event_grid,membership
from foundry.expr.evaluate import evaluate
from foundry.runtime import prepare_backend
from foundry.expr.parse import parse
from foundry.ops.registry import OPS
from foundry.backtest.spec import ExecutionSpec

from foundry.adapters.data import published_bars,long_table


class PointInTimeTests(unittest.TestCase):
    def test_late_bar_revision_and_late_first_publication_are_not_backfilled(self):
        import pandas as pd
        df=pd.DataFrame(dict(date=pd.to_datetime(['2024-01-01','2024-01-01','2024-01-02','2024-01-03']),
                             published=pd.to_datetime(['2024-01-01','2024-01-03','2024-01-03','2024-01-03']),
                             symbol=['x']*4,close=[10.,999.,20.,30.]))
        frozen=published_bars(df)
        np.testing.assert_array_equal(frozen.close,[10.,np.nan,30.])
        dates,assets,grids,start=long_table(frozen,'2024-01-01','2024-01-02',('close',))
        self.assertEqual(assets,['x'])
        np.testing.assert_array_equal(grids['close'],[[10.],[np.nan]])

    def test_delayed_publication_and_revision_do_not_rewrite_history(self):
        dates=np.arange('2024-01-01','2024-01-08',dtype='datetime64[D]')
        records=[dict(symbol='x',effective='2024-01-01',published='2024-01-03',value=2.),
                 dict(symbol='x',effective='2024-01-01',published='2024-01-05',value=3.),
                 dict(symbol='x',effective='2024-01-04',published='2024-01-06',value=4.),
                 dict(symbol='x',effective='2024-01-01',published='2024-01-07',value=8.)]
        actual=event_grid(dates,['x'],records)[:,0]
        np.testing.assert_array_equal(actual,[np.nan,np.nan,2,2,3,4,4])
        np.testing.assert_array_equal(event_grid(dates[:4],['x'],records)[:,0],actual[:4])

    def test_membership_uses_effective_intervals(self):
        dates=np.arange('2024-01-01','2024-01-06',dtype='datetime64[D]')
        intervals=[dict(symbol='x',start='2024-01-02',end='2024-01-04')]
        np.testing.assert_array_equal(membership(dates,['x'],intervals)[:,0],[0,1,1,0,0])

    def test_future_assets_cannot_enter_cross_section_through_is_nan(self):
        B=prepare_backend('cpu');x=np.arange(90,dtype=float).reshape(30,3)+1
        expression=parse('cs_rank(is_nan(x))',['x']).root
        original=evaluate(expression,{'x':x},B,{},np.ones_like(x,bool))
        extended=np.column_stack((x,np.full((30,2),np.nan)))
        scope=np.column_stack((np.ones_like(x,bool),np.zeros((30,2),bool)))
        actual=evaluate(expression,{'x':extended},B,{},scope)
        np.testing.assert_array_equal(actual[:,:3],original)
        self.assertTrue(np.isnan(actual[:,3:]).all())

    def test_every_operator_ignores_poisoned_future(self):
        B=prepare_backend('cpu');rng=np.random.default_rng(25);cut=193
        values=[rng.uniform(.1,2,(320,7)) for _ in range(3)]
        values[0][30:33,1]=np.nan
        for name,op in OPS.items():
            args={'unary':[values[0]],'binary':values[:2],'binary_const':[values[0],.7],
                  'ternary':values,'ts':[values[0],60],'ts_param':[values[0],60,.7],
                  'pair':values[:2]+[60],'pair_param':values[:2]+[60,2.]}[op.kind]
            short=[np.ascontiguousarray(v[:cut]) if isinstance(v,np.ndarray) else v for v in args]
            poison=[v.copy() if isinstance(v,np.ndarray) else v for v in args]
            for v in poison:
                if isinstance(v,np.ndarray):v[cut:]=rng.uniform(1e10,1e11,v[cut:].shape)
            with self.subTest(op=name):
                np.testing.assert_allclose(B.impl[name](*poison)[:cut],B.impl[name](*short),rtol=1e-12,atol=1e-12)

    def test_open_eligibility_does_not_use_same_day_close(self):
        o=np.full((40,2),10.);c=o.copy();cost=o*.001
        before=ExecutionSpec.build(o,c,cost)
        c[20]=np.nan
        after=ExecutionSpec.build(o,c,cost)
        np.testing.assert_array_equal(before.live,after.live)
