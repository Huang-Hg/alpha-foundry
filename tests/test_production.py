"""The production portfolio reproduces its recorded books, returns and reference bit for bit (local: GPU + data).

The record lives next to the library (production/regression.npz, local only, like the library itself). A change
that moves it is either a bug or a deliberate semantic change; in the latter case re-record it with
    python -m tests.test_production record
and state the change. A data rebuild also moves it (the research window is fixed, but qnt revises history)."""
import json
import sys
import unittest

import cupy as cp
import numpy as np

from foundry.config import ROOT
from foundry.engine import Engine
from foundry.portfolio.library import Library
from foundry.portfolio.portfolio import Portfolio

RECORD = ROOT / 'production' / 'regression.npz'
HEADLINE = ('sharpe', 'cagr', 'maxdd', 'wsharpe', 'recent_growth')


def snapshot():
    engine = Engine()
    lib = Library.load()
    rows = engine.formulas([m['expr'] for m in lib.members])
    ref = Portfolio(engine, lib).reference
    out = {f'member{i}_{k}': cp.asnumpy(r[f'_{k}']) for i, r in enumerate(rows) for k in ('W', 'rr')}
    out.update(U=cp.asnumpy(ref.U), rr=cp.asnumpy(ref.rr), headline=np.array([getattr(ref, k) for k in HEADLINE]),
               members=np.array(json.dumps(lib.members)))
    return out


class ProductionTests(unittest.TestCase):
    def test_production_portfolio_is_reproduced_bitwise(self):
        record, now = np.load(RECORD), snapshot()
        self.assertEqual(str(record['members']), str(now['members']), 'the library changed: re-record')
        for k in record.files:
            if k != 'members':
                self.assertTrue(np.array_equal(record[k], now[k], equal_nan=True), k)


if __name__ == '__main__':
    if sys.argv[1:] == ['record']:
        np.savez_compressed(RECORD, **snapshot())
        print(f'recorded {RECORD}')
    else:
        unittest.main()
