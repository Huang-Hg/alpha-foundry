"""Boundary schemas and atomic storage protocols; no data or GPU required."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from jsonschema import ValidationError

from foundry.adapters.library import FileLibraryStore
from foundry.candidates import Formula
from foundry.contracts import Study
from foundry.portfolio.library import PROTOCOL
from foundry.research_types import ContextConflict
from foundry.service import schema


class ContractTests(unittest.TestCase):
    def test_library_protocol_study_and_compare_and_swap(self):
        study = Study('us_equity', '2025-07-01', '2026-06-30')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'isolated' / 'library.json'
            store = FileLibraryStore(path)
            empty = store.read(study)
            first = empty.add(Formula('gt(ma_dist_20,0)'), .5)
            store.commit(empty, first)
            record = json.loads(path.read_bytes())
            self.assertEqual(record['protocol'], PROTOCOL)
            self.assertEqual(store.read(study), first)
            before = path.read_bytes()
            with self.assertRaises(ContextConflict):
                store.commit(empty, empty.add(Formula('gt(ma_dist_60,0)'), .25))
            self.assertEqual(path.read_bytes(), before)
            self.assertFalse(path.with_suffix('.lock').exists())
            with self.assertRaisesRegex(ContextConflict, 'study'):
                store.read(Study('a_share', study.start, study.asof))
            record['protocol'] = 'research-v9'
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'protocol'):
                store.read(study)

    def test_failed_publication_leaves_the_previous_version(self):
        study = Study('us_equity', '2025-07-01', '2026-06-30')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'library.json'
            store = FileLibraryStore(path)
            empty = store.read(study)
            first = empty.add(Formula('gt(ma_dist_20,0)'), .5)
            store.commit(empty, first)
            before = path.read_bytes()
            with patch('foundry.adapters.library.publish', side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    store.commit(first, first.add(Formula('gt(ma_dist_60,0)'), .25))
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(store.read(study).version, first.version)
            self.assertFalse(path.with_suffix('.lock').exists())

    def test_query_and_commit_are_distinct_discoverable_methods(self):
        for method in ('research.evaluate', 'research.commit'):
            request = dict(jsonrpc='2.0', id='round-1', method=method,
                           params=dict(context='snapshot',
                                       candidates=[dict(id='a', source=dict(kind='formula', expression='gt(ma_dist_20,0)'))]))
            self.assertIs(schema.parse(request), request)
            self.assertEqual(schema.result(request, {'rows': []})['id'], 'round-1')
        self.assertIn('research.commit', schema.describe()['methods'])

    def test_obsolete_requests_and_unknown_parameters_are_rejected(self):
        for request in (
            dict(op='eval', exprs=['x']),
            dict(jsonrpc='2.0', id='x', method='research.evaluate',
                 params=dict(context='ctx', candidates=[dict(id='a', expression='x')], accept=True)),
            dict(jsonrpc='2.0', id='x', method='research.commit', params=dict(candidates=[])),
            dict(jsonrpc='2.0', id='x', method='system.status', params=dict(fallback=True)),
        ):
            with self.subTest(request=request), self.assertRaises(ValidationError):
                schema.parse(request)

    def test_candidate_ids_are_unique(self):
        request = dict(jsonrpc='2.0', id='r', method='research.evaluate',
                       params=dict(context='ctx', candidates=[dict(id='a', source=dict(kind='formula', expression='x')),
                                                              dict(id='a', source=dict(kind='formula', expression='y'))]))
        with self.assertRaisesRegex(ValueError, 'unique'):
            schema.parse(request)
