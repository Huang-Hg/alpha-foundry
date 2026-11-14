"""Socket boundary: errors, bounded queue, disconnect and deterministic draining."""
import json
from pathlib import Path
import socket
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from foundry.service import daemon


class ServiceLifecycleTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        root = Path(folder.name)
        paths = patch.multiple(daemon, STATE=root, PORTFILE=root/'port.json',
                               STATUSFILE=root/'status.json', LOGFILE=root/'service.log')
        paths.start()
        self.addCleanup(paths.stop)
        engine = SimpleNamespace(study=SimpleNamespace(market='us_equity'), spec=SimpleNamespace(backend='cpu'))
        reference = SimpleNamespace(n=1, wsharpe=0., cagr=0., maxdd=0.)
        research = SimpleNamespace(engine=engine, portfolio=SimpleNamespace(reference=reference))
        self.server = daemon.Server(SimpleNamespace(), research)
        self.started, self.release = threading.Event(), threading.Event()
        self.executed = []

        def dispatch(request):
            identity = request['id']
            if identity == 'held':
                self.started.set()
                if not self.release.wait(15):
                    raise TimeoutError('test did not release held work')
            if identity == 'broken':
                raise ValueError('invalid external request')
            self.executed.append(identity)
            return dict(completed=identity)

        self.server.dispatch = dispatch
        self.worker = threading.Thread(target=self.server._worker, daemon=True)
        self.worker.start()
        self.addCleanup(self.cleanup_server)

    def cleanup_server(self):
        self.release.set()
        if not self.server.stopping:
            self.server.stopping = True
            self.server.q.put(None, timeout=5)
        self.worker.join(timeout=5)
        self.server.sock.close()
        self.assertFalse(self.worker.is_alive(), 'worker failed to drain')

    def begin(self, identity, method='research.describe'):
        client, accepted = socket.socketpair()
        client.settimeout(5)
        handler = threading.Thread(target=self.server._handle, args=(accepted,), daemon=True)
        handler.start()
        client.sendall(json.dumps(dict(jsonrpc='2.0', id=identity, method=method, params={})).encode()+b'\n')
        return client, handler

    def receive(self, pending):
        client, handler = pending
        with client:
            result = json.loads(client.makefile('rb').readline())
        handler.join(timeout=5)
        self.assertFalse(handler.is_alive())
        return result

    def wait_for(self, condition):
        deadline = time.monotonic()+5
        while not condition():
            self.assertLess(time.monotonic(), deadline, 'service state transition timeout')
            time.sleep(.01)

    def test_request_failure_does_not_kill_the_worker(self):
        error = self.receive(self.begin('broken'))
        self.assertEqual(error['error']['data']['type'], 'ValueError')
        self.assertFalse(error['error']['data']['client_disconnect_cancels'])
        good = self.receive(self.begin('next'))
        self.assertEqual(good['result'], dict(completed='next'))
        self.assertEqual(self.server.served, 2)
        self.assertTrue(self.worker.is_alive())

    def test_backpressure_and_repeated_stop_drain_all_accepted_jobs(self):
        held = self.begin('held')
        self.assertTrue(self.started.wait(5))
        jobs = [daemon.Job(dict(jsonrpc='2.0', id=f'queued-{i}', method='research.describe', params={}))
                for i in range(daemon.MAX_PENDING)]
        for job in jobs:
            self.server.q.put_nowait(job)
        rejected = self.receive(self.begin('over-capacity'))
        self.assertEqual(rejected['error']['data']['type'], 'Full')
        self.assertNotIn('over-capacity', self.executed)
        first_stop = self.begin('stop-1', 'system.stop')
        self.wait_for(lambda: self.server.stopping)
        second_stop = self.begin('stop-2', 'system.stop')
        self.assertTrue(self.worker.is_alive())
        self.assertFalse(jobs[-1].done.is_set())
        self.release.set()
        self.assertEqual(self.receive(held)['result']['completed'], 'held')
        for pending in (first_stop, second_stop):
            response = self.receive(pending)['result']
            self.assertEqual(response['state'], 'draining')
            self.assertFalse(response['client_disconnect_cancels'])
        self.worker.join(timeout=5)
        self.assertFalse(self.worker.is_alive())
        self.assertTrue(all(job.done.is_set() for job in jobs))
        self.assertEqual(self.server.served, daemon.MAX_PENDING+1)
        self.assertTrue(self.server.q.empty(), 'repeated stop enqueued a second sentinel')

    def test_client_disconnect_does_not_cancel_accepted_work(self):
        client, handler = self.begin('held')
        self.assertTrue(self.started.wait(5))
        client.close()
        self.assertTrue(self.worker.is_alive())
        self.release.set()
        self.wait_for(lambda: self.server.served == 1)
        handler.join(timeout=5)
        self.assertFalse(handler.is_alive())
        self.assertEqual(self.executed, ['held'])


class ServiceHeartbeatTests(unittest.TestCase):
    def test_idle_service_publishes_until_stop_and_joins_publisher(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.multiple(daemon, STATE=root, PORTFILE=root/'port.json', STATUSFILE=root/'status.json',
                                LOGFILE=root/'service.log', HEARTBEAT_SECONDS=.02):
                engine = SimpleNamespace(study=SimpleNamespace(market='us_equity'),
                                         spec=SimpleNamespace(backend='cpu'))
                reference = SimpleNamespace(n=1, wsharpe=0., cagr=0., maxdd=0.)
                research = SimpleNamespace(engine=engine, portfolio=SimpleNamespace(reference=reference))
                server = daemon.Server(SimpleNamespace(), research)
                observations, observed = [], threading.Event()
                publish = server._publish

                def record_publication():
                    publish()
                    observations.append(server.status()['now'])
                    if len(observations) >= 3:
                        observed.set()

                with patch.object(server, '_publish', side_effect=record_publication):
                    serving = threading.Thread(target=server.run, daemon=True)
                    serving.start()
                    try:
                        self.assertTrue(observed.wait(5), 'idle status did not refresh')
                        self.assertGreater(observations[-1], observations[0])
                        self.assertEqual(server.served, 0)
                        self.assertEqual(json.loads((root/'status.json').read_text())['phase'], 'serve')
                    finally:
                        with socket.create_connection((daemon.HOST, server.port), timeout=5) as connection:
                            request = dict(jsonrpc='2.0', id='shutdown', method='system.stop', params={})
                            connection.sendall(json.dumps(request).encode()+b'\n')
                            response = json.loads(connection.makefile('rb').readline())
                        serving.join(timeout=5)
                    self.assertEqual(response['result']['state'], 'draining')
                    self.assertFalse(serving.is_alive())
                    self.assertTrue(server.publisher_stop.is_set())
                    self.assertFalse(server.publisher.is_alive())
                    self.assertEqual(json.loads((root/'status.json').read_text())['phase'], 'stopped')
                    self.assertFalse((root/'port.json').exists())


if __name__ == '__main__':
    unittest.main()
