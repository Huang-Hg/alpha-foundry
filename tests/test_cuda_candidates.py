"""External signals/weights use the same PIT, trading and admission paths on Colab."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cupy as cp
import numpy as np
from jsonschema import validate as validate_schema

from foundry.adapters.arrays import FileArrayStore, MemoryArrayStore, write_bundle
from foundry.adapters.library import FileLibraryStore, MemoryLibraryStore
from foundry.adapters.serialization import candidate_from_json, evaluation_result
from foundry.candidates import ArrayBatch, EqualWeight, Formula
from foundry.expr.evaluate import evaluate
from foundry.expr.parse import parse
from foundry.portfolio.library import Library, Member
from foundry.research import Research
from foundry.research_types import Candidate, ContextConflict
from foundry.service import daemon, schema
from foundry.runtime import open_runtime
from fixtures import EXPRESSIONS, research_for, runtime_for, settings_for, study_for
from test_candidates import rewrite_artifact


def batch_for(engine, values, kind='signals', ids=None):
    count, days, _ = values.shape
    identities = tuple(f'input-{i}' for i in range(count)) if ids is None else ids
    observed = np.broadcast_to(np.arange(days, dtype=np.int64), (count, days)).copy()
    return ArrayBatch(kind, identities, values, engine.context.id, 'integration-producer:v1',
                      tuple(str(day) for day in engine.dates), engine.panel.assets,
                      observed, np.full((count, days), -1, dtype=np.int64))


def formula_signals(engine, expressions):
    return engine.xp.stack([evaluate(parse(expression, engine.vocabulary).root, engine.fields,
                                     engine.backend, {}, engine.universe) for expression in expressions])


class ArrayBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = runtime_for('us_equity')
        cls.engine = cls.runtime.engine

    @classmethod
    def tearDownClass(cls):
        cls.runtime.close()

    def test_axes_snapshot_cutoffs_and_dtypes_are_checked_before_writes(self):
        research = research_for(self.runtime)
        batch = batch_for(self.engine, np.ones((1, *self.engine.panel.shape), dtype=np.float32))
        future_observed = batch.observed_through.copy()
        future_observed[0, 0] = 1
        future_fit = batch.fit_through.copy()
        future_fit[0, -2] = len(self.engine.dates)-1
        below = batch.fit_through.copy()
        below[0, 0] = -2
        invalid = (
            replace(batch, snapshot='another-snapshot'),
            replace(batch, dates=batch.dates[::-1]),
            replace(batch, assets=batch.assets[::-1]),
            replace(batch, observed_through=future_observed),
            replace(batch, fit_through=future_fit),
            replace(batch, fit_through=below),
            replace(batch, observed_through=batch.observed_through.astype(np.int32)),
            replace(batch, fit_through=batch.fit_through[:, :-1]),
            replace(batch, values=batch.values[0]),
            replace(batch, values=batch.values.astype(np.float16)),
            replace(batch, ids=('duplicate', 'duplicate')),
            replace(batch, ids=('',)),
            replace(batch, producer=''),
        )
        context = research.context
        for changed in invalid:
            with self.subTest(changed=changed.kind, shape=changed.values.shape), \
                 patch.object(research.arrays, 'put', side_effect=AssertionError('invalid input reached storage')):
                with self.assertRaises(ValueError):
                    research.register_batch(changed, context.id)
                self.assertEqual(research.context, context)
        with self.assertRaises(ContextConflict):
            research.register_batch(batch, 'stale-portfolio-context')

    def test_nonfinite_signal_is_rejected_and_all_nan_is_a_g1_result(self):
        research = research_for(self.runtime)
        values = np.ones((1, *self.engine.panel.shape), dtype=np.float64)
        values[0, 0, 0] = np.inf
        with self.assertRaisesRegex(ValueError, 'infinity'):
            research.register_batch(batch_for(self.engine, values), research.context.id)
        values[:] = np.nan
        candidates = research.register_batch(batch_for(self.engine, values), research.context.id)
        result = research.evaluate(candidates, research.context.id)
        self.assertFalse(result.rows[0].row.ok)
        self.assertFalse(result.rows[0].verdict.passed)
        self.assertEqual(result.rows[0].verdict.stopped_at, 'G1')
        validate_schema(evaluation_result(result), schema.describe()['responses']['research.evaluate'])

    def test_target_weight_constraints_and_rounding_tolerance(self):
        research = research_for(self.runtime)
        zero = np.zeros((1, *self.engine.panel.shape), dtype=np.float64)
        universe = self.engine.to_host(self.engine.universe)
        live_day, live_asset = np.argwhere(universe)[-1]
        for value, point in ((-.1, (live_day, live_asset)), (np.nan, (live_day, live_asset)),
                             (1.01, (live_day, live_asset))):
            invalid = zero.copy()
            invalid[0, point[0], point[1]] = value
            with self.subTest(value=value, point=point), self.assertRaises(ValueError):
                research.register_batch(batch_for(self.engine, invalid, 'target_weights'), research.context.id)
        close = zero.copy()
        at = np.flatnonzero(universe.sum(1) >= 2)[-1]
        assets = np.flatnonzero(universe[at])[:2]
        close[0, at, assets] = [.5, .5+self.engine.portfolio_cfg.weight_sum_atol/2]
        candidates = research.register_batch(batch_for(self.engine, close, 'target_weights'), research.context.id)
        row = research.evaluate_device(tuple(c.source for c in candidates), research.context.id).rows[0]
        expected = close[0, at, assets]/close[0, at].sum()
        np.testing.assert_allclose(self.engine.to_host(row.book[at, assets]), expected, rtol=0, atol=1e-15)
        excessive = close.copy()
        excessive[0, at, assets[1]] = .5+2*self.engine.portfolio_cfg.weight_sum_atol
        with self.assertRaisesRegex(ValueError, 'exposure'):
            research.register_batch(batch_for(self.engine, excessive, 'target_weights'), research.context.id)

    def test_ineligible_a_share_weights_are_rejected(self):
        with runtime_for('a_share') as runtime:
            research = research_for(runtime)
            values = np.zeros((1, *runtime.panel.shape), dtype=np.float64)
            day, asset = np.argwhere(~runtime.engine.to_host(runtime.engine.universe))[0]
            values[0, day, asset] = .1
            with self.assertRaisesRegex(ValueError, 'ineligible'):
                research.register_batch(batch_for(runtime.engine, values, 'target_weights'), research.context.id)

    def test_registration_owns_input_buffers_and_queries_never_write(self):
        research = research_for(self.runtime)
        values = formula_signals(self.engine, EXPRESSIONS[:1]).astype(cp.float32)
        batch = batch_for(self.engine, values)
        context = research.context
        candidates = research.register_batch(batch, context.id)
        self.assertEqual(research.context, context)
        expected = research.evaluate_device(tuple(c.source for c in candidates), context.id, report=True).rows[0]
        expected_book, expected_returns = expected.book.copy(), expected.returns.copy()
        values[:] = 0
        batch.observed_through[:] = -1
        with patch.object(research.arrays, 'put', side_effect=AssertionError('query wrote an artifact')), \
             patch.object(research.store, 'commit', side_effect=AssertionError('query wrote a portfolio')):
            again = research.evaluate_device(tuple(c.source for c in candidates), context.id, report=True).rows[0]
            response = research.evaluate(candidates, context.id)
        cp.testing.assert_array_equal(again.book, expected_book)
        cp.testing.assert_array_equal(again.returns, expected_returns)
        self.assertEqual(research.context, context)
        self.assertIsNone(response.rows[0].row.book)
        restored = research.arrays.read(candidates[0].source)
        np.testing.assert_array_equal(restored.observed_through[0], np.arange(len(self.engine.dates)))

    def test_registration_handoff_survives_another_stream_mutating_the_input(self):
        research = research_for(self.runtime)
        original = self.engine.to_host(formula_signals(self.engine, EXPRESSIONS[:1])).astype(np.float32)
        producer = cp.cuda.Stream(non_blocking=True)
        with producer:
            values = cp.asarray(original)
            candidates = research.register_batch(batch_for(self.engine, values), research.context.id)
            self.assertTrue(producer.done, 'registration returned before taking ownership of producer data')
        overwriter = cp.cuda.Stream(non_blocking=True)
        with overwriter:
            values.fill(0)
        overwriter.synchronize()
        restored = research.arrays.read(candidates[0].source)
        np.testing.assert_array_equal(restored.values, original)
        actual = research.evaluate_device((candidates[0].source,), research.context.id).rows[0]
        expected = self.engine.evaluate(EXPRESSIONS[:1], report=False)[0]
        cp.testing.assert_array_equal(actual.book, expected.book)
        cp.testing.assert_array_equal(actual.returns, expected.returns)

    def test_reloaded_artifact_cannot_bypass_pit_validation(self):
        empty = Library((), self.engine.study)
        arrays = MemoryArrayStore()
        batch = batch_for(self.engine, np.zeros((1, *self.engine.panel.shape), dtype=np.float64))
        future = batch.fit_through.copy()
        future[0, 0] = 1
        source = arrays.put(replace(batch, fit_through=future))[0]
        research = Research(self.engine, MemoryLibraryStore(empty), arrays)
        with self.assertRaisesRegex(ValueError, 'future cutoff'):
            research.evaluate((Candidate('late-fit', source),), research.context.id)
        self.assertFalse(self.engine.has_array(source))
        invalid_library = empty.add(source, .5)
        with self.assertRaisesRegex(ValueError, 'future cutoff'):
            Research(self.engine, MemoryLibraryStore(invalid_library), arrays)


class ArrayMarketTests(unittest.TestCase):
    def test_measurement_cache_is_bounded_and_does_not_cache_mutable_rows_or_noise(self):
        settings = settings_for('us_equity')
        settings = replace(settings, cuda=replace(settings.cuda, candidate_batch=2))
        with open_runtime(settings, study_for(settings, 'us_equity')) as runtime:
            engine = runtime.engine
            sources = tuple(Formula(e) for e in EXPRESSIONS)
            with patch.object(engine, '_compute_sources', wraps=engine._compute_sources) as compute:
                first = engine.evaluate_sources(sources[:2])
                first[0].ok = False
                again = engine.evaluate_sources(sources[:2])
                self.assertTrue(again[0].ok)
                self.assertEqual(compute.call_count, 1)
                noise = engine.evaluate_sources(sources[:1], null_shift=True)
                self.assertEqual(compute.call_count, 2)
                self.assertFalse(bool(cp.any(noise[0].book)))
                engine.evaluate_sources(sources[2:])
                self.assertLessEqual(len(engine._reports), 2)
                engine.evaluate_sources(sources[:1])
                self.assertEqual(compute.call_count, 4)

    def test_batched_commit_preserves_greedy_order_versions_and_atomic_publication(self):
        for market in ('us_equity', 'a_share', 'crypto_spot', 'crypto_perpetual'):
            with self.subTest(market=market), runtime_for(market) as runtime:
                engine = runtime.engine
                candidates = tuple(Candidate(str(i), Formula(e)) for i, e in enumerate(EXPRESSIONS))
                serial, batched = research_for(runtime), research_for(runtime)
                batch_size = engine.commit_batch
                engine.commit_batch = 1
                expected = serial.commit(candidates, serial.context.id)
                engine.commit_batch = 2
                actual = batched.commit(candidates, batched.context.id)
                self.assertEqual(serial.portfolio.library, batched.portfolio.library)
                for a, b in zip(expected.rows, actual.rows):
                    self.assertEqual((a.id, a.accepted, a.weight, a.verdict.reason),
                                     (b.id, b.accepted, b.weight, b.verdict.reason))
                self.assertEqual(expected.before, actual.before)
                self.assertEqual(expected.context, actual.context)
                engine.commit_batch = batch_size
                full_research = research_for(runtime)
                full = full_research.commit(candidates, full_research.context.id)
                self.assertEqual([(r.accepted,r.weight) for r in full.rows],
                                 [(r.accepted,r.weight) for r in actual.rows])
                # A fresh writer fails at the only publication point; the live
                # portfolio and version must remain unchanged after evaluation.
                if any(row.accepted for row in actual.rows):
                    failed = research_for(runtime)
                    original = failed.portfolio
                    with patch.object(failed.store, 'commit', side_effect=OSError('publication failed')):
                        with self.assertRaisesRegex(OSError, 'publication failed'):
                            failed.commit(candidates, failed.context.id)
                    self.assertIs(failed.portfolio, original)
                    self.assertEqual(failed.store.read(engine.study), original.library)

    def test_bounded_array_cache_handles_larger_batches_and_cold_library_reload(self):
        settings = settings_for('us_equity')
        settings = replace(settings, cuda=replace(settings.cuda, candidate_batch=1))
        study = study_for(settings, 'us_equity')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            store, arrays = FileLibraryStore(root/'library.json'), FileArrayStore(root/'arrays')
            with open_runtime(settings, study) as runtime:
                engine = runtime.engine
                research = Research(engine, store, arrays)
                candidates = research.register_batch(batch_for(engine, formula_signals(engine, EXPRESSIONS[:3])),
                                                      research.context.id)
                self.assertEqual(engine.candidate_batch, 1)
                self.assertGreater(len(candidates), engine.candidate_batch)
                self.assertLessEqual(len(engine._arrays), engine.candidate_batch)
                actual = research.evaluate_device(tuple(c.source for c in candidates), research.context.id, report=True)
                expected = engine.evaluate(EXPRESSIONS[:3])
                for first, second in zip(actual.rows, expected):
                    cp.testing.assert_array_equal(first.book, second.book)
                    cp.testing.assert_array_equal(first.returns, second.returns)
                self.assertLessEqual(len(engine._arrays), engine.candidate_batch)
                library = Library(tuple(Member(c.source, .25) for c in candidates), study)
                store.commit(research.portfolio.library, library)
                combined = sum(row.book*.25 for row in expected)
                expected_state = engine.metrics([combined])[0]
                expected_book = engine.to_host(expected_state.book).copy()
                expected_returns = engine.to_host(expected_state.returns).copy()
            with open_runtime(settings, study) as runtime:
                engine = runtime.engine
                reopened = Research(engine, store, FileArrayStore(root/'arrays'))
                self.assertEqual(reopened.portfolio.library, library)
                self.assertLessEqual(len(engine._arrays), engine.candidate_batch)
                np.testing.assert_allclose(engine.to_host(reopened.portfolio.reference.W), expected_book, rtol=0, atol=1e-15)
                np.testing.assert_allclose(engine.to_host(reopened.portfolio.reference.rr), expected_returns, rtol=0, atol=1e-12)
                actual = reopened.evaluate_device(tuple(c.source for c in candidates), reopened.context.id)
                self.assertEqual(len(actual.rows), len(candidates))
                self.assertTrue(all(row.ok for row in actual.rows))
                self.assertLessEqual(len(engine._arrays), engine.candidate_batch)

    def test_mixed_sources_match_formula_books_and_cpu_cuda_decisions(self):
        expression = 'gt(ma_dist_20,0)'
        for market in ('us_equity', 'a_share', 'crypto_spot'):
            observations = []
            for backend in ('cpu', 'cuda'):
                with self.subTest(market=market, backend=backend), runtime_for(market, backend) as runtime:
                    engine, research = runtime.engine, research_for(runtime)
                    xp = engine.xp
                    expected = engine.evaluate((expression,))[0]
                    # Negative and >1 scores are intentionally legal; clipping reproduces the binary formula.
                    signals = xp.where(engine.panel['ma_dist_20'] > 0, 2., -3.).astype(xp.float32)[None]
                    signal = research.register_batch(batch_for(engine, signals, ids=('score',)), research.context.id)[0]
                    weight = research.register_batch(batch_for(engine, expected.book[None].copy(), 'target_weights',
                                                                 ids=('weights',)), research.context.id)[0]
                    candidates = (Candidate('formula', Formula(expression)), signal, weight, Candidate('equal', EqualWeight()))
                    result = research.evaluate_device(tuple(c.source for c in candidates), research.context.id, report=True)
                    for row in result.rows[:3]:
                        np.testing.assert_array_equal(engine.to_host(row.book > 0), engine.to_host(expected.book > 0))
                        np.testing.assert_allclose(engine.to_host(row.book), engine.to_host(expected.book), rtol=0, atol=1e-15)
                        np.testing.assert_allclose(engine.to_host(row.returns), engine.to_host(expected.returns), rtol=0, atol=1e-11)
                    for verdict in result.verdicts[:3]:
                        self.assertEqual(verdict.passed, result.verdicts[0].passed)
                        self.assertEqual(verdict.proposal.weight, result.verdicts[0].proposal.weight)
                    observations.append([(engine.to_host(row.book).copy(), engine.to_host(row.returns).copy(),
                                          verdict.passed, verdict.proposal.weight)
                                         for row, verdict in zip(result.rows, result.verdicts)])
            for cpu, gpu in zip(*observations):
                np.testing.assert_array_equal(cpu[0] > 0, gpu[0] > 0)
                np.testing.assert_allclose(cpu[0], gpu[0], rtol=0, atol=1e-15)
                np.testing.assert_allclose(cpu[1], gpu[1], rtol=0, atol=1e-10)
                self.assertEqual(cpu[2:], gpu[2:])

    def test_array_commit_reopens_without_formula_cache_and_detects_corruption(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            library_store, arrays = FileLibraryStore(root/'library.json'), FileArrayStore(root/'arrays')
            with runtime_for('us_equity') as runtime:
                research = Research(runtime.engine, library_store, arrays)
                candidates = research.register_batch(batch_for(runtime.engine, formula_signals(runtime.engine, EXPRESSIONS)),
                                                      research.context.id)
                result = research.evaluate(candidates, research.context.id)
                passing = tuple(c for c, row in zip(candidates, result.rows) if row.verdict.passed)
                self.assertTrue(passing, [row.verdict.message for row in result.rows])
                commit = research.commit(passing[:1], research.context.id)
                self.assertTrue(commit.rows[0].accepted)
                source = passing[0].source
                expected_library = research.portfolio.library
                expected_book = runtime.engine.to_host(research.portfolio.reference.W).copy()
                expected_returns = runtime.engine.to_host(research.portfolio.reference.rr).copy()
                record = json.loads((root/'library.json').read_text())
                self.assertEqual(record['members'][-1]['source']['kind'], 'signals')
                self.assertNotIn('expression', record['members'][-1]['source'])
            with runtime_for('us_equity') as runtime:
                self.assertFalse(runtime.engine.has_array(source))
                reopened = Research(runtime.engine, library_store, FileArrayStore(root/'arrays'))
                self.assertTrue(runtime.engine.has_array(source))
                self.assertEqual(reopened.portfolio.library, expected_library)
                np.testing.assert_array_equal(runtime.engine.to_host(reopened.portfolio.reference.W), expected_book)
                np.testing.assert_array_equal(runtime.engine.to_host(reopened.portfolio.reference.rr), expected_returns)
            rewrite_artifact(arrays.path_for(source), value_delta=.01)
            with runtime_for('us_equity') as runtime:
                with self.assertRaisesRegex(ValueError, 'digest'):
                    Research(runtime.engine, library_store, FileArrayStore(root/'arrays'))

    def test_rpc_batch_registration_and_evaluation_match_direct_use_case(self):
        with tempfile.TemporaryDirectory() as folder, runtime_for('us_equity') as runtime:
            root = Path(folder)
            research = research_for(runtime)
            server = daemon.Server(runtime, research)
            self.addCleanup(server.sock.close)
            context = research.context
            batch = batch_for(runtime.engine, runtime.engine.to_host(formula_signals(runtime.engine, EXPRESSIONS[:1])))
            path = root/'signals.npz'
            write_bundle(path, batch)
            request = dict(jsonrpc='2.0', id='register', method='research.register_batch',
                           params=dict(context=context.id, path=str(path)))
            registered = server.dispatch(schema.parse(request))
            validate_schema(registered, schema.describe()['responses']['research.register_batch'])
            candidates = tuple(candidate_from_json(row) for row in registered['candidates'])
            self.assertEqual(research.context, context)
            direct = evaluation_result(research.evaluate(candidates, context.id))
            request = dict(jsonrpc='2.0', id='evaluate', method='research.evaluate',
                           params=dict(context=context.id, candidates=[asdict(c) for c in candidates]))
            wire = server.dispatch(schema.parse(request))
            validate_schema(wire, schema.describe()['responses']['research.evaluate'])
            self.assertEqual(wire, direct)
            self.assertEqual(research.context, context)


if __name__ == '__main__':
    unittest.main()
