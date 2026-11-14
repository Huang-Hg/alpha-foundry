"""Explicit production-baseline audit; never records or migrates a baseline.

This suite requires an independently provisioned current-protocol production library and
baseline on the authorized worker. Historical baselines remain untouched and are
rejected when their protocol differs. General integration uses tests.fixtures instead.
"""
from dataclasses import asdict
import json
import unittest

import numpy as np

from foundry.adapters.library import FileLibraryStore
from foundry.adapters.arrays import FileArrayStore
from foundry.config import ROOT, load_config, path
from foundry.contracts import Study
from foundry.portfolio.library import PROTOCOL
from foundry.research import Research
from foundry.runtime import open_runtime

RECORD = ROOT / 'production' / 'regression.npz'
HEADLINE = ('sharpe', 'cagr', 'maxdd', 'wsharpe', 'recent_growth')


class ProductionTests(unittest.TestCase):
    def test_production_portfolio_is_reproduced_bitwise(self):
        with np.load(RECORD) as record:
            self.assertIn('protocol', record.files, 'historical baseline has no current protocol; explicit review required')
            self.assertEqual(str(record['protocol']), PROTOCOL, 'baseline protocol differs; do not silently re-record')
            study = Study(**json.loads(str(record['study'])))
            settings = load_config()
            with open_runtime(settings, study) as runtime:
                engine = runtime.engine
                research = Research(engine, FileLibraryStore(ROOT / 'production' / 'library.json'),
                                    FileArrayStore(path(settings.portfolio.artifact_dir)))
                library = research.portfolio.library
                rows = research.evaluate_device(tuple(member.source for member in library.members),
                                                research.context.id, report=True).rows
                ref = research.portfolio.reference
                now = {f'member{i}_{key}': engine.to_host(value)
                       for i, row in enumerate(rows) for key, value in (('W', row.book), ('rr', row.returns))}
                now.update(U=engine.to_host(ref.U), rr=engine.to_host(ref.rr),
                           headline=np.array([getattr(ref, key) for key in HEADLINE]))
                self.assertEqual(json.loads(str(record['members'])), [asdict(member) for member in library.members])
                for key, value in now.items():
                    self.assertTrue(np.array_equal(record[key], value, equal_nan=True), key)


if __name__ == '__main__':
    unittest.main()
