"""CUDA isolated-perpetual verification on the authorized Colab worker."""
from dataclasses import asdict, fields, replace
from pathlib import Path
import tempfile
import unittest

import cupy as cp
import numpy as np
from jsonschema import validate as validate_schema

from foundry.backtest import perpetual_cpu, perpetual_cuda
from foundry.adapters.arrays import FileArrayStore, write_bundle
from foundry.adapters.library import MemoryLibraryStore
from foundry.adapters.serialization import candidate_from_json, evaluation_result
from foundry.candidates import Formula
from foundry.portfolio.library import Library
from foundry.research import Research
from foundry.research_types import Candidate
from foundry.runtime import open_runtime
from foundry.service import daemon, schema
from fixtures import settings_for, study_for
from test_cuda_candidates import batch_for, formula_signals
from test_perpetual import HandCalculatedAccounts, flat_spec, host_result, target


class PerpetualCudaTests(HandCalculatedAccounts, unittest.TestCase):
    xp, to_host = cp, staticmethod(cp.asnumpy)

    @classmethod
    def setUpClass(cls):
        cfg = settings_for('crypto_perpetual')
        cls.executor = perpetual_cuda.prepare(cfg.cuda.mixed_return_atol, cfg.portfolio.metric_atol)

    @classmethod
    def tearDownClass(cls):
        cls.executor.close()

    def test_signed_book_fingerprints_match_cpu_including_zero_and_batch_tail(self):
        cpu = perpetual_cpu.prepare()
        self.addCleanup(cpu.close)
        books = np.random.default_rng(17).normal(size=(35, 9, 3))
        books[1] = books[0]
        books[2], books[3] = 0., -0.
        expected = cpu.fingerprints(books)
        actual = self.executor.fingerprints(cp.asarray(books))
        self.assertEqual(actual, expected)
        self.assertEqual(actual[0], actual[1])
        self.assertEqual(actual[2], actual[3])
        self.assertNotEqual(actual[0], actual[-1])

    def test_cpu_cuda_batch_tail_matches_all_cashflows_and_event_flags(self):
        cpu = perpetual_cpu.prepare()
        self.addCleanup(cpu.close)
        for assets in (1, 3, 33):
            with self.subTest(assets=assets):
                spec = flat_spec(days=5, assets=assets, fee=.0004)
                rng = np.random.default_rng(2025)
                prices = 100. * np.exp(np.cumsum(rng.normal(0., .025, spec.mark_open.shape), axis=0))
                opens = np.vstack((prices[:1], prices[:-1]))
                spec = replace(spec, trade_open=opens, mark_open=opens, mark_close=prices,
                               mark_high=np.maximum(opens, prices) * 1.2,
                               mark_low=np.minimum(opens, prices) * .8,
                               funding_mark=opens,
                               funding_rate=np.where(np.arange(len(opens))[:, None] % 8 == 0, .002, 0.)
                                            * np.ones((1, assets)))
                book = rng.normal(size=(35, *spec.shape))
                book *= 5. / np.abs(book).sum(2, keepdims=True)
                book[0] = 0.
                expected = cpu.simulate(book, cpu.device_spec(spec), 1, 365., 'f64')
                actual = host_result(self.executor.simulate(cp.asarray(book), self.executor.device_spec(spec),
                                                           1, 365., 'f64'), cp.asnumpy)
                np.testing.assert_allclose(actual.returns, expected.returns, rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(actual.statistics, expected.statistics, rtol=1e-10, atol=1e-10)
                for field in fields(expected.risk):
                    first, second = getattr(expected.risk, field.name), getattr(actual.risk, field.name)
                    if field.name == 'model':
                        self.assertEqual(first, second)
                    elif first.dtype.kind in 'biu':
                        np.testing.assert_array_equal(second, first, err_msg=field.name)
                    else:
                        np.testing.assert_allclose(second, first, rtol=1e-12, atol=1e-12, err_msg=field.name)
                self.assertGreater(actual.risk.liquidations.sum(), 0)

    def test_reduced_precision_refines_funding_and_touch_threshold_neighbors(self):
        cpu = perpetual_cpu.prepare()
        self.addCleanup(cpu.close)
        for risk in ('funding', 'touch'):
            for side in (-1, 0, 1):
                spec = flat_spec(days=2)
                book = target(spec, [[5.]])
                name, boundary = ('funding_rate', .195) if risk == 'funding' else ('mark_low', 4./.04975)
                value = np.nextafter(boundary, -np.inf if side < 0 else np.inf) if side else boundary
                changed = getattr(spec, name).copy()
                changed[25] = value
                spec = replace(spec, **{name: changed})
                expected = cpu.simulate(book, cpu.device_spec(spec), 0, 365., 'f64')
                ds = self.executor.device_spec(spec)
                for precision in self.executor.precisions:
                    with self.subTest(risk=risk, side=side, precision=precision):
                        actual = host_result(self.executor.simulate(cp.asarray(book), ds, 0, 365., precision), cp.asnumpy)
                        np.testing.assert_array_equal(actual.returns, expected.returns)
                        for field in fields(expected.risk):
                            np.testing.assert_equal(getattr(actual.risk, field.name), getattr(expected.risk, field.name))

    def test_reduced_precision_exponent_domain_uses_reference(self):
        cpu = perpetual_cpu.prepare()
        self.addCleanup(cpu.close)
        for price in (1.e-40, 1.e40):
            spec = flat_spec(days=3, fee=.0004)
            names = ('trade_open', 'mark_open', 'mark_high', 'mark_low', 'mark_close', 'funding_mark')
            spec = replace(spec, **{name: np.full_like(getattr(spec, name), price) for name in names})
            book = target(spec, [[5.], [-5.]])
            expected = cpu.simulate(book, cpu.device_spec(spec), 0, 365., 'f64')
            ds = self.executor.device_spec(spec)
            for precision in ('mixed', 'paired'):
                with self.subTest(price=price, precision=precision):
                    actual = host_result(self.executor.simulate(cp.asarray(book), ds, 0, 365., precision), cp.asnumpy)
                    np.testing.assert_array_equal(actual.returns, expected.returns)
                    for field in fields(expected.risk):
                        np.testing.assert_equal(getattr(actual.risk, field.name), getattr(expected.risk, field.name))

    def test_partial_days_and_inactive_warp_lanes_preserve_the_event_grid(self):
        from foundry.backtest.perpetual import ARRAY_FIELDS
        spec = flat_spec(days=3, assets=3, fee=.0004)
        indices = np.r_[0, np.arange(24, 41), np.arange(48, 72)]
        spec = replace(spec, **{name: getattr(spec, name)[indices] for name in ARRAY_FIELDS})
        book = target(spec, [[2., -2., 1.]])
        cpu = perpetual_cpu.prepare()
        self.addCleanup(cpu.close)
        expected = cpu.simulate(book, cpu.device_spec(spec), 0, 365., 'f64')
        actual = host_result(self.executor.simulate(cp.asarray(book), self.executor.device_spec(spec), 0, 365., 'f64'), cp.asnumpy)
        np.testing.assert_allclose(actual.returns, expected.returns, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(actual.risk.liquidation_events, expected.risk.liquidation_events)

    def test_fee_insolvency_threshold_is_refined_in_every_precision(self):
        cpu = perpetual_cpu.prepare()
        self.addCleanup(cpu.close)
        for side in (-np.inf, 80./(1.-.0004), np.inf):
            spec = flat_spec(days=3, fee=.0004)
            mark = spec.mark_open.copy(); mark[48:] = 80.5
            trade = spec.trade_open.copy(); trade[48:] = np.nextafter(80./(1.-.0004), side)
            spec = replace(spec, mark_open=mark, mark_high=mark, mark_low=mark,
                           mark_close=mark, funding_mark=mark, trade_open=trade)
            book = target(spec, [[5.]])
            expected = cpu.simulate(book, cpu.device_spec(spec), 0, 365., 'f64')
            ds = self.executor.device_spec(spec)
            for precision in self.executor.precisions:
                with self.subTest(side=side, precision=precision):
                    actual = host_result(self.executor.simulate(cp.asarray(book), ds, 0, 365., precision), cp.asnumpy)
                    np.testing.assert_array_equal(actual.returns, expected.returns)
                    np.testing.assert_array_equal(actual.risk.bankrupt, expected.risk.bankrupt)
                    np.testing.assert_array_equal(actual.risk.trading_fees, expected.risk.trading_fees)


class PerpetualResearchTests(unittest.TestCase):
    def test_signed_sources_precision_plan_and_rpc_match_on_both_backends(self):
        observations = []
        expression = 'mul_const(gt(ma_dist_20,0),-1)'
        for backend in ('cpu', 'cuda'):
            settings = settings_for('crypto_perpetual', backend)
            settings = replace(settings, cuda=replace(settings.cuda, precision='paired'))
            study = study_for(settings, 'crypto_perpetual')
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as folder:
                arrays = FileArrayStore(Path(folder)/'arrays')
                store = MemoryLibraryStore(Library((), study))
                with open_runtime(settings, study) as runtime:
                    engine = runtime.engine
                    research = Research(engine, store, arrays)
                    plan = engine.context.precision
                    supported = ('f64', 'mixed', 'paired') if backend == 'cuda' else ('f64',)
                    self.assertEqual((plan.requested, plan.effective, plan.supported), ('paired', 'f64', supported))
                    self.assertEqual(plan.reason, 'risk_screening_requires_f64' if backend == 'cuda' else 'executor_f64_only')
                    self.assertEqual(engine.allocation.max_gross, settings.crypto_perpetual.leverage)
                    self.assertTrue(engine.allocation.allow_short)
                    self.assertEqual(engine.ppy, 365.)
                    context = research.context
                    signals = research.register_batch(batch_for(engine, formula_signals(engine, (expression,))),
                                                      context.id)
                    values = np.zeros((1, *engine.panel.shape))
                    values[0, :, :2] = [3., -2.]
                    values = np.where(engine.to_host(engine.universe)[None], values, 0.)
                    batch = batch_for(engine, values, 'target_exposure', ids=('leveraged',))
                    with self.assertRaisesRegex(ValueError, 'target_exposure'):
                        research.register_batch(replace(batch, kind='target_weights'), context.id)
                    with self.assertRaisesRegex(ValueError, 'gross exposure'):
                        research.register_batch(replace(batch, values=values * 1.01), context.id)
                    future = batch.fit_through.copy()
                    future[0, 0] = 1
                    with self.assertRaisesRegex(ValueError, 'future cutoff'):
                        research.register_batch(replace(batch, fit_through=future), context.id)
                    server = daemon.Server(runtime, research)
                    self.addCleanup(server.sock.close)
                    bundle = Path(folder)/'exposure.npz'
                    write_bundle(bundle, batch)
                    registered = server.dispatch(schema.parse(dict(jsonrpc='2.0', id='register',
                        method='research.register_batch', params=dict(context=context.id, path=str(bundle)))))
                    validate_schema(registered, schema.describe()['responses']['research.register_batch'])
                    exposure = tuple(candidate_from_json(row) for row in registered['candidates'])
                    candidates = (Candidate('formula', Formula(expression)), *signals, *exposure)
                    device = research.evaluate_device(tuple(c.source for c in candidates), context.id, report=True)
                    first, second, leveraged = device.rows
                    np.testing.assert_array_equal(engine.to_host(first.book), engine.to_host(second.book))
                    np.testing.assert_array_equal(engine.to_host(first.returns), engine.to_host(second.returns))
                    expected_book = values[0].copy()
                    expected_book[:engine.start] = 0.
                    np.testing.assert_allclose(engine.to_host(leveraged.book), expected_book, rtol=0, atol=1e-14)
                    self.assertGreater(np.abs(expected_book).sum(1).max(), 1.)
                    self.assertTrue(all(row.precision == 'f64' and row.risk is not None for row in device.rows))
                    if backend == 'cpu':
                        with self.assertRaisesRegex(ValueError, 'does not support paired'):
                            engine.simulate(leveraged.book[None], precision='paired')
                    else:
                        low = engine.simulate(leveraged.book[None], precision='paired')
                        cp.testing.assert_allclose(low.returns[0, engine.start:], leveraged.returns, rtol=0, atol=1e-10)
                    direct = evaluation_result(research.evaluate(candidates, context.id))
                    wire = server.dispatch(schema.parse(dict(jsonrpc='2.0', id='evaluate',
                        method='research.evaluate', params=dict(context=context.id, candidates=[asdict(c) for c in candidates]))))
                    validate_schema(wire, schema.describe()['responses']['research.evaluate'])
                    self.assertEqual(wire, direct)
                    self.assertEqual(research.context, context)
                    self.assertEqual(store.read(study).members, ())
                    expected_returns = engine.to_host(leveraged.returns).copy()
                    observations.append((expected_book, expected_returns,
                                         [(v.passed, v.stopped_at, v.reason, v.proposal.weight) for v in device.verdicts]))
                # No expression cache is available on reopen: the source must restore its own PIT artifact.
                with open_runtime(settings, study) as reopened:
                    fresh = Research(reopened.engine, store, FileArrayStore(Path(folder)/'arrays'))
                    restored = fresh.evaluate_device((exposure[0].source,), fresh.context.id, report=True).rows[0]
                    np.testing.assert_array_equal(reopened.engine.to_host(restored.book), expected_book)
                    np.testing.assert_array_equal(reopened.engine.to_host(restored.returns), expected_returns)
        np.testing.assert_allclose(observations[0][0], observations[1][0], rtol=0, atol=1e-14)
        np.testing.assert_allclose(observations[0][1], observations[1][1], rtol=0, atol=1e-11)
        self.assertEqual(observations[0][2], observations[1][2])


if __name__ == '__main__':
    unittest.main()
