"""PIT prefixes across equity/crypto grids and the actual resident API."""
from dataclasses import replace
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import cupy as cp
import numpy as np
import pandas as pd
from jsonschema import validate as validate_schema

from foundry.adapters.serialization import evaluation_result
from foundry.candidates import Formula
from foundry.contracts import Study
from foundry.ops.registry import OPS
from foundry.research_types import Candidate
from foundry.runtime import open_runtime, prepare_backend
from foundry.search.evaluator import Archive, Evaluator
from foundry.service import client, schema
from fixtures import research_for, runtime_for

EXPRESSIONS = ('gt(ma_dist_20,0)', 'gt(ts_rank(ma_dist_20,120),0.7)',
               'gt(cs_rank(is_nan(ma_dist_60)),0.5)', 'gt(cs_rank(ts_mean(rangepos,20)),0.7)')


class CudaPointInTimeTests(unittest.TestCase):
    def test_real_history_prefix_and_poisoned_future(self):
        for name in ('us_equity', 'a_share', 'crypto_spot'):
            with self.subTest(market=name), runtime_for(name) as runtime:
                full = runtime.engine
                rows = full.evaluate(EXPRESSIONS)
                cutoffs = [str(full.dates[full.start + (len(full.dates) - full.start) * n // 4])
                           for n in (2, 3)]
                for cutoff in cutoffs:
                    with open_runtime(runtime.settings, Study(name, full.study.start, cutoff)) as short_runtime:
                        short = short_runtime.engine
                        indices = [full.panel.assets.index(a) for a in short.panel.assets]
                        stop = len(short.dates)
                        for field in short.panel.fields:
                            cp.testing.assert_allclose(short.panel[field], full.panel[field][:stop, indices],
                                                       rtol=1e-12, atol=1e-12)
                        for actual, expected in zip(short.evaluate(EXPRESSIONS), rows):
                            cp.testing.assert_allclose(actual.book, expected.book[:stop, indices], rtol=1e-12, atol=1e-12)
                            cp.testing.assert_array_equal(actual.book > 0, expected.book[:stop, indices] > 0)
                            cp.testing.assert_allclose(actual.returns, expected.returns[:stop-full.start], rtol=0, atol=1e-11)
                        self.assertNotEqual(short.context.id, full.context.id)
                read = pd.read_parquet
                columns = ('open', 'close', 'high', 'low') + (('exec_open', 'exec_close') if name == 'a_share' else ())

                def poisoned(file, *args, **kwargs):
                    frame = read(file, *args, **kwargs)
                    future = frame.date > pd.Timestamp(cutoffs[0])
                    frame.loc[future, list(columns)] *= 1e8
                    return frame

                with patch.object(pd, 'read_parquet', side_effect=poisoned):
                    dirty_runtime = open_runtime(runtime.settings, full.study)
                with dirty_runtime:
                    stop = int(np.searchsorted(full.dates, np.datetime64(cutoffs[0]), side='right'))
                    self.assertNotEqual(dirty_runtime.engine.context.data_snapshot, full.context.data_snapshot)
                    for actual, expected in zip(dirty_runtime.engine.evaluate(EXPRESSIONS), rows):
                        cp.testing.assert_array_equal(actual.book[:stop], expected.book[:stop])
                        cp.testing.assert_array_equal(actual.returns[:stop-full.start], expected.returns[:stop-full.start])

    def test_every_cuda_operator_ignores_poisoned_future(self):
        backend = prepare_backend('cuda')
        self.addCleanup(backend.close)
        rng = np.random.default_rng(25)
        cut = 193
        values = [rng.uniform(.1, 2, (320, 7)) for _ in range(3)]
        values[0][30:33, 1] = np.nan
        for name, op in OPS.items():
            args = {'unary': [values[0]], 'binary': values[:2], 'binary_const': [values[0], .7],
                    'ternary': values, 'ts': [values[0], 60], 'ts_param': [values[0], 60, .7],
                    'pair': values[:2]+[60], 'pair_param': values[:2]+[60, 2.]}[op.kind]
            short = [cp.asarray(v[:cut]) if isinstance(v, np.ndarray) else v for v in args]
            poison = [v.copy() if isinstance(v, np.ndarray) else v for v in args]
            for value in poison:
                if isinstance(value, np.ndarray):
                    value[cut:] = rng.uniform(1e10, 1e11, value[cut:].shape)
            poison = [cp.asarray(v) if isinstance(v, np.ndarray) else v for v in poison]
            with self.subTest(op=name):
                cp.testing.assert_allclose(backend.impl[name](*poison)[:cut], backend.impl[name](*short),
                                           rtol=1e-12, atol=1e-12)

    def test_close_decision_first_changes_next_open(self):
        with runtime_for('us_equity') as runtime:
            engine = runtime.engine
            weights = cp.stack([engine.equal_weight(), engine.equal_weight()])
            at = (engine.start + len(engine.dates)) // 2
            weights[1, at] = 0
            returns = engine.simulate(weights).returns
            cp.testing.assert_array_equal(returns[0, :at+1], returns[1, :at+1])
            self.assertGreater(float(cp.abs(returns[0, at+1:]-returns[1, at+1:]).max()), 0.)

    def test_miner_and_agent_share_read_only_admission(self):
        with runtime_for('us_equity') as runtime:
            research = research_for(runtime)
            context = research.context
            device = research.evaluate_device(tuple(Formula(e) for e in EXPRESSIONS), context.id)
            self.assertIsInstance(device.rows[0].book, cp.ndarray)
            candidates = tuple(Candidate(str(i), Formula(e)) for i, e in enumerate(EXPRESSIONS))
            result = research.evaluate(candidates, context.id)
            wire = evaluation_result(result)
            json.dumps(wire, allow_nan=False)
            validate_schema(wire, schema.describe()['responses']['research.evaluate'])
            miner = Evaluator(research, Archive()).evaluate(EXPRESSIONS)
            self.assertEqual([r['id'] for r in wire['rows']], [str(i) for i in range(4)])
            self.assertEqual([r['decision']['passed'] for r in wire['rows']], [v.passed for v in device.verdicts])
            self.assertEqual([r.g2 for r in miner], [v.passed for v in device.verdicts])
            self.assertEqual(research.context, context)


class ResidentApiTests(unittest.TestCase):
    def test_socket_and_direct_use_cases_match_and_shutdown(self):
        candidates = [dict(id='a', source=dict(kind='formula', expression=EXPRESSIONS[0])),
                      dict(id='b', source=dict(kind='formula', expression=EXPRESSIONS[0])),
                      dict(id='invalid', source=dict(kind='formula', expression='unknown(ma_dist_20)'))]
        with runtime_for('us_equity') as runtime:
            research = research_for(runtime)
            expected = evaluation_result(research.evaluate(tuple(Candidate(c['id'], Formula(c['source']['expression'])) for c in candidates),
                                                           research.context.id))['rows']
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            boot = """
import sys
from pathlib import Path
from foundry.service import daemon
from fixtures import research_for, runtime_for
root=Path(sys.argv[1])
daemon.STATE=root
daemon.PORTFILE=root/'port.json'
daemon.STATUSFILE=root/'status.json'
daemon.LOGFILE=root/'service.log'
with runtime_for('us_equity') as runtime:
    daemon.Server(runtime,research_for(runtime)).run()
"""
            with (root/'process.log').open('w') as log:
                process = subprocess.Popen([sys.executable, '-u', '-c', boot, folder], stdout=log, stderr=subprocess.STDOUT)
                try:
                    deadline = time.monotonic()+90
                    while not (root/'port.json').exists():
                        self.assertIsNone(process.poll(), (root/'process.log').read_text())
                        self.assertLess(time.monotonic(), deadline, 'service warmup timeout')
                        time.sleep(.1)
                    with patch.object(client, 'PORTFILE', root/'port.json'):
                        description = client.call('research.describe', {})
                        context = description['context']['id']
                        self.assertEqual(description['context']['snapshot']['study']['market'], 'us_equity')
                        self.assertIn('unavailable', description['context']['snapshot']['data_contract']['publication'])
                        request = dict(jsonrpc='2.0', id='round1', method='research.evaluate',
                                       params=dict(context=context, candidates=candidates))
                        response = client.invoke(request)
                        self.assertEqual(response['id'], 'round1')
                        self.assertEqual(response['result']['rows'], expected)
                        validate_schema(response['result'], schema.describe()['responses']['research.evaluate'])
                        self.assertEqual(client.call('research.describe', {})['context']['id'], context)
                        families = client.call('research.family', dict(context=context, seeds=[
                            dict(id='skip', expression='unknown(ma_dist_20)', hypothesis='invalid'),
                            dict(id='keep', expression='ma_dist_20', hypothesis='trend')]))
                        self.assertEqual(families['families'][0]['id'], 'keep')
                        self.assertEqual(families['skipped'][0]['id'], 'skip')
                        validate_schema(families, schema.describe()['responses']['research.family'])
                        request['params']['context'] = 'stale'
                        self.assertIn('stale research context', client.invoke(request)['error']['message'])
                        port = client.info()['port']
                        with socket.create_connection(('127.0.0.1', port)) as sock:
                            sock.sendall(b'{"op":"eval","exprs":["x"]}\n')
                            self.assertIn('error', json.loads(sock.makefile('rb').readline()))
                        client.stop()
                    self.assertEqual(process.wait(timeout=15), 0, (root/'process.log').read_text())
                    self.assertFalse((root/'port.json').exists())
                finally:
                    if process.poll() is None:
                        process.terminate()
                        process.wait(timeout=15)


if __name__ == '__main__':
    unittest.main()
