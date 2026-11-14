"""Candidate artifacts: owned buffers, provenance, integrity and transport protocols."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from foundry.adapters.arrays import FileArrayStore, MemoryArrayStore, load_bundle, write_bundle
from foundry.candidates import ArrayBatch


def batch_fixture(dtype=np.float64):
    values = np.arange(24, dtype=dtype).reshape(2, 4, 3) / 100
    observed = np.broadcast_to(np.arange(4, dtype=np.int64), (2, 4)).copy()
    return ArrayBatch('signals', ('first', 'second'), values, 'snapshot-a', 'model:v1',
                      ('2026-01-02', '2026-01-05', '2026-01-06', '2026-01-07'),
                      ('a', 'b', 'c'), observed, np.full((2, 4), -1, dtype=np.int64))


def rewrite_artifact(path, *, value_delta=0., protocol=None):
    with np.load(path, allow_pickle=False) as archive:
        record = {key: archive[key] for key in archive.files}
    if protocol is not None:
        metadata = json.loads(str(record['metadata'].item()))
        metadata['format'] = protocol
        record['metadata'] = np.asarray(json.dumps(metadata))
    record['values'] = record['values'] + value_delta
    np.savez(path, **record)


class CandidateArtifactTests(unittest.TestCase):
    def test_bundle_round_trip_preserves_axes_precision_and_cutoffs(self):
        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as folder:
                batch = batch_fixture(dtype)
                path = Path(folder)/'input.npz'
                write_bundle(path, batch)
                actual = load_bundle(path)
                for field in ('kind', 'ids', 'snapshot', 'producer', 'dates', 'assets'):
                    self.assertEqual(getattr(actual, field), getattr(batch, field))
                for field in ('values', 'observed_through', 'fit_through'):
                    np.testing.assert_array_equal(getattr(actual, field), getattr(batch, field))
                self.assertEqual(actual.values.dtype, dtype)
                self.assertEqual(actual.observed_through.dtype, np.int64)

    def test_memory_and_file_stores_own_readonly_content(self):
        with tempfile.TemporaryDirectory() as folder:
            batch = batch_fixture(np.float32)
            expected = batch.values.copy()
            memory, files = MemoryArrayStore(), FileArrayStore(Path(folder)/'arrays')
            in_memory, on_disk = memory.put(batch), files.put(batch)
            self.assertEqual(in_memory, on_disk)
            self.assertEqual(files.put(batch), on_disk)
            batch.values[:] = -999
            batch.observed_through[:] = -1
            for store in (memory, files):
                for index, source in enumerate(on_disk):
                    restored = store.read(source)
                    actual = restored.values[0]
                    self.assertEqual(restored.dates, batch.dates)
                    self.assertEqual(restored.assets, batch.assets)
                    self.assertEqual(restored.snapshot, batch.snapshot)
                    np.testing.assert_array_equal(restored.observed_through[0], np.arange(4, dtype=np.int64))
                    self.assertEqual(actual.dtype, np.float32)
                    np.testing.assert_array_equal(actual, expected[index])
                    self.assertFalse(actual.flags.writeable)
                    with self.assertRaises(ValueError):
                        actual[0, 0] = 1

    def test_identity_binds_provenance_and_causal_declarations(self):
        batch = batch_fixture()
        store = MemoryArrayStore()
        original = store.put(batch)
        self.assertEqual(store.put(replace(batch, ids=('alias-1', 'alias-2'))), original)
        fitted = batch.fit_through.copy()
        fitted[:, -1] = 1
        for changed in (replace(batch, producer='model:v2'), replace(batch, snapshot='snapshot-b'),
                        replace(batch, kind='target_weights'), replace(batch, fit_through=fitted)):
            with self.subTest(kind=changed.kind, producer=changed.producer, snapshot=changed.snapshot):
                self.assertNotEqual(store.put(changed)[0].digest, original[0].digest)

    def test_corruption_provenance_and_old_artifact_protocol_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            store = FileArrayStore(Path(folder))
            batch = batch_fixture()
            first, second = store.put(batch)
            with self.assertRaisesRegex(ValueError, 'provenance'):
                store.read(replace(first, producer='another-model'))
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                store.path_for(replace(first, digest='../../outside'))
            rewrite_artifact(store.path_for(first), value_delta=.01)
            with self.assertRaisesRegex(ValueError, 'digest'):
                store.read(first)
            with self.assertRaisesRegex(ValueError, 'digest'):
                store.put(batch)
            rewrite_artifact(store.path_for(second), protocol='candidate-array-obsolete')
            with self.assertRaisesRegex(ValueError, 'protocol'):
                store.read(second)

    def test_old_transport_protocol_and_pickle_arrays_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'input.npz'
            write_bundle(path, batch_fixture())
            rewrite_artifact(path, protocol='candidate-batch-obsolete')
            with self.assertRaisesRegex(ValueError, 'protocol'):
                load_bundle(path)
            write_bundle(path, replace(batch_fixture(), values=np.array([object()], dtype=object)))
            with self.assertRaisesRegex(ValueError, 'allow_pickle'):
                load_bundle(path)


if __name__ == '__main__':
    unittest.main()
