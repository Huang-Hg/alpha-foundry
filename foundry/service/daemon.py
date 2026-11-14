"""Bounded JSON-RPC adapter. One worker owns research state and all writes."""
from dataclasses import asdict
import collections
import json
import os
import queue
import socket
import threading
import time
import traceback
from pathlib import Path

from foundry.adapters.files import publish
from foundry.adapters.arrays import load_bundle
from foundry.adapters.serialization import (RegisteredBatch, candidate_from_json, describe,
                                            evaluation_result, json_value)
from foundry.portfolio.library import PROTOCOL
from foundry.research_types import FamilySeed, ContextConflict
from foundry.service import schema
from foundry.service.protocol import HOST, LOGFILE, PORTFILE, STATE, STATUSFILE

MAX_PENDING = 64
MAX_REQUEST_BYTES = 1048576
HEARTBEAT_SECONDS = 10.


class Job:
    def __init__(self, request):
        self.request = request
        self.done = threading.Event()
        self.result = None
        self.error = None
        self.state = 'queued'


class Server:
    def __init__(self, runtime, research, port=0):
        self.runtime, self.research = runtime, research
        self.q = queue.Queue(maxsize=MAX_PENDING)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((HOST, port))
        self.sock.listen(MAX_PENDING)
        self.port = self.sock.getsockname()[1]
        self.started = time.time()
        self.stopping = False
        self.stopped = False
        self.served = self.n_expr = 0
        self.jobs = collections.deque(maxlen=32)
        self.current = None
        self.submit_lock = threading.Lock()
        self.publisher_stop = threading.Event()

    def status(self):
        ref = self.research.portfolio.reference
        return dict(protocol=PROTOCOL, kind='service', market=self.research.engine.study.market,
            phase='stopped' if self.stopped else 'stopping' if self.stopping else 'serve', pid=os.getpid(), port=self.port,
            now=time.time(), elapsed_seconds=time.time() - self.started, busy_seconds=0.,
            device=self.research.engine.spec.backend, vram='', warm='', queue=self.q.qsize(), clients=0,
            served=self.served, exprs=self.n_expr, batches=self.served, cur=self.current,
            portfolio=dict(n=ref.n, sharpe=ref.wsharpe, cagr=ref.cagr, maxdd=ref.maxdd), jobs=list(self.jobs), sources={})

    def _publish(self):
        publish(STATUSFILE, json.dumps(json_value(self.status()), ensure_ascii=False, allow_nan=False).encode())

    def _heartbeat(self):
        while not self.publisher_stop.wait(HEARTBEAT_SECONDS):
            self._publish()

    def run(self):
        STATE.mkdir(parents=True, exist_ok=True)
        publish(PORTFILE, json.dumps(dict(port=self.port, pid=os.getpid(), started=self.started)).encode())
        worker = threading.Thread(target=self._worker)
        self.publisher = threading.Thread(target=self._heartbeat, name='foundry-status')
        worker.start()
        self.publisher.start()
        try:
            self._publish()
            while not self.stopping:
                conn, _ = self.sock.accept()
                threading.Thread(target=self._handle, args=(conn,), daemon=True).start()
        finally:
            self.sock.close()
            with self.submit_lock:
                if not self.stopping:
                    self.stopping = True
                    self.q.put(None)
            worker.join()
            self.stopped = True
            self.publisher_stop.set()
            self.publisher.join()
            PORTFILE.unlink(missing_ok=True)
            self._publish()

    def _handle(self, conn):
        request_id = None
        try:
            conn.settimeout(30.)
            line = conn.makefile('rb').readline(MAX_REQUEST_BYTES + 1)
            if len(line) > MAX_REQUEST_BYTES:
                raise ValueError('request exceeds protocol byte limit')
            if not line:
                conn.close()
                return
            raw = json.loads(line)
            request_id = raw.get('id')
            request = schema.parse(raw)
            method = request['method']
            if method == 'system.status':
                value = self.status()
            elif method == 'system.stop':
                with self.submit_lock:
                    already_stopping = self.stopping
                    if not already_stopping:
                        self.stopping = True
                        self.q.put(None)
                value = dict(state='draining', pending=self.q.qsize(), running=self.current,
                             client_disconnect_cancels=False)
                if not already_stopping:
                    socket.create_connection((HOST, self.port), timeout=2.).close()
            else:
                job = Job(request)
                with self.submit_lock:
                    if self.stopping:
                        raise RuntimeError('service is draining; request was not accepted')
                    self.q.put(job, timeout=1.)
                job.done.wait()
                if job.error is not None:
                    raise job.error
                value = job.result
            response = schema.result(request, value)
        except Exception as error:
            if isinstance(error, ContextConflict):
                code, reason, message = -32009, 'context_conflict', str(error)
            elif isinstance(error, queue.Full):
                code, reason, message = -32001, 'queue_full', 'service queue is full; request was not accepted'
            elif isinstance(error, socket.timeout):
                code, reason, message = -32002, 'transport_timeout', str(error)
            elif isinstance(error, json.JSONDecodeError):
                code, reason, message = -32700, 'invalid_json', str(error)
            else:
                code, reason, message = -32602, 'request_failed', str(error)
            response = dict(jsonrpc='2.0', id=request_id, error=dict(code=code, message=message,
                            data=dict(type=type(error).__name__, reason=reason, client_disconnect_cancels=False)))
        try:
            conn.sendall(json.dumps(response, ensure_ascii=False, allow_nan=False).encode() + b'\n')
        except OSError:
            # A disconnected client does not cancel an accepted command.
            pass
        finally:
            conn.close()

    def _worker(self):
        while True:
            job = self.q.get()
            if job is None:
                return
            request = job.request
            started = time.time()
            job.state = 'running'
            self.current = dict(op=request['method'], id=request['id'], t0=started, state=job.state)
            self._publish()
            try:
                job.result = self.dispatch(request)
                job.state = 'succeeded'
            except Exception as error:
                job.error = error
                job.state = 'failed'
                with LOGFILE.open('a', encoding='utf-8') as log:
                    log.write(traceback.format_exc())
            self.served += 1
            self.jobs.append(dict(t=time.time(), op=request['method'], id=request['id'], state=job.state, dt=time.time() - started,
                                  ok=job.error is None, n=0, variants=0, verdicts={}, source=None))
            self.current = None
            self._publish()
            job.done.set()

    def dispatch(self, request):
        method, params = request['method'], request['params']
        if method == 'research.describe':
            return describe(self.research)
        if method == 'research.register_batch':
            candidates = self.research.register_batch(load_bundle(Path(params['path'])), params['context'])
            return json_value(asdict(RegisteredBatch(self.research.context, candidates)))
        if method == 'research.family':
            result = self.research.family(tuple(FamilySeed(**row) for row in params['seeds']), params['context'])
            self.n_expr += result.variants
            return json_value(asdict(result))
        candidates = tuple(candidate_from_json(row) for row in params['candidates'])
        result = (self.research.commit(candidates, params['context']) if method == 'research.commit'
                  else self.research.evaluate(candidates, params['context']))
        self.n_expr += len(candidates)
        return evaluation_result(result)
