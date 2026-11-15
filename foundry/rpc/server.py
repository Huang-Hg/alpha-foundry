"""Bounded single-worker JSON-RPC service with explicit ingress and shutdown ownership."""
from __future__ import annotations

from collections import deque
from contextlib import ExitStack
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import queue
import threading
import time
import traceback
from typing import TYPE_CHECKING

from foundry.rpc import schema
from foundry.rpc.codec import candidate_from_json, decode_bundle, from_json, json_value
from foundry.study.store import publish
from foundry.types import CandidateGrid, ContextConflict, FamilySeed, SessionBackend

if TYPE_CHECKING:
    import socket


class Job:
    def __init__(self, request: dict, payload: bytes = b'', upload_bytes: int = 0) -> None:
        self.request, self.payload, self.upload_bytes = request, payload, upload_bytes
        self.done = threading.Event()
        self.result = None
        self.error: Exception | None = None
        self.state = 'queued'
        self.started: float | None = None


@dataclass
class Ingress:
    thread: threading.Thread
    phase: str = 'reading'


class Server:
    """Startup reads metadata; one worker owns research calls and the caller owns backend lifetime."""

    def __init__(self, backend: SessionBackend, port: int = 0, *, state_dir: Path = schema.STATE,
                 max_handlers: int = schema.MAX_PENDING + 4, upload_budget: int = schema.MAX_ARRAY_BYTES) -> None:
        import selectors
        import socket

        if max_handlers < 1 or not 0 < upload_budget <= schema.MAX_ARRAY_BYTES:
            raise ValueError('ingress requires positive handler and bounded upload capacities')
        self.backend = backend
        self._summary = self._read_summary()
        self.state_dir = Path(state_dir)
        self.portfile, self.statusfile, self.logfile = (self.state_dir / name for name in
                                                     ('port.json', 'status.json', 'service.log'))
        self.q: queue.Queue[Job] = queue.Queue(maxsize=schema.MAX_PENDING)
        self.started = time.time()
        self.stopping = self.stopped = False
        self.failure: Exception | None = None
        self.served = self.n_expr = 0
        self.jobs = deque(maxlen=32)
        self.current = None
        self._running: Job | None = None
        self.submit_lock = threading.Lock()
        self.publisher_stop = threading.Event()
        self._ingress_lock = threading.Lock()
        self._handler_slots = threading.BoundedSemaphore(max_handlers)
        self._connections: dict[socket.socket, Ingress] = {}
        self._threads: set[threading.Thread] = set()
        self._upload_lock = threading.Lock()
        self.upload_budget, self.upload_bytes = upload_budget, 0
        with ExitStack() as cleanup:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            cleanup.callback(self.sock.close)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.bind((schema.HOST, port))
            self.sock.listen(schema.MAX_PENDING)
            self.port = self.sock.getsockname()[1]
            self._wake_read, self._wake_write = socket.socketpair()
            cleanup.callback(self._wake_read.close)
            cleanup.callback(self._wake_write.close)
            self._selector = selectors.DefaultSelector()
            cleanup.callback(self._selector.close)
            self._selector.register(self.sock, selectors.EVENT_READ, 'listener')
            self._selector.register(self._wake_read, selectors.EVENT_READ, 'stop')
            self._io = cleanup.pop_all()

    def _read_summary(self) -> dict:
        context = self.backend.context()
        portfolio = self.backend.portfolio(context.id)
        description = self.backend.describe()
        return dict(market=context.snapshot.study.market,
                    device=description['parameters']['backend'],
                    portfolio=dict(n=len(portfolio.sources), sharpe=portfolio.metrics.sharpe,
                                   cagr=portfolio.metrics.cagr, maxdd=portfolio.metrics.maxdd))

    def status(self) -> dict:
        with self.submit_lock:
            return dict(protocol=schema.PROTOCOL, kind='service', **self._summary,
                phase='failed' if self.failure is not None else 'stopped' if self.stopped else 'stopping' if self.stopping else 'serve',
                pid=os.getpid(), port=self.port, now=time.time(), elapsed_seconds=time.time() - self.started,
                busy_seconds=0., vram='', warm='', queue=self.q.qsize(), clients=0,
                served=self.served, exprs=self.n_expr, batches=self.served, cur=self.current,
                jobs=list(self.jobs), sources={}, error=None if self.failure is None else str(self.failure))

    def _publish(self) -> None:
        publish(self.statusfile, json.dumps(json_value(self.status()), ensure_ascii=False, allow_nan=False).encode())

    def _heartbeat(self) -> None:
        while not self.publisher_stop.wait(schema.HEARTBEAT_SECONDS):
            try:
                self._publish()
            except OSError as error:
                self._fatal(error)
                return

    def _stop_locked(self) -> None:
        if not self.stopping:
            self.stopping = True
            self._wake_write.sendall(b'1')

    def request_stop(self) -> None:
        """Reject new work, interrupt unqueued readers and drain all accepted jobs."""
        with self.submit_lock:
            self._stop_locked()
        self._close_readers()

    def _close_readers(self) -> None:
        import socket

        with self._ingress_lock:
            readers = tuple(connection for connection, ingress in self._connections.items() if ingress.phase == 'reading')
        for connection in readers:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                # A peer or its owner may already have closed this reader.
                pass

    def _reserve_upload(self, size: int) -> None:
        with self._upload_lock:
            if self.upload_bytes + size > self.upload_budget:
                raise queue.Full('aggregate upload byte budget is exhausted; request was not accepted')
            self.upload_bytes += size

    def _release_upload(self, size: int) -> None:
        with self._upload_lock:
            self.upload_bytes -= size

    def _phase(self, connection: socket.socket, phase: str) -> None:
        with self._ingress_lock:
            self._connections[connection].phase = phase

    def _start_handler(self, connection: socket.socket) -> threading.Thread:
        """Reserve ingress before starting a reader; ownership transfers on successful start."""
        if not self._handler_slots.acquire(blocking=False):
            raise queue.Full('ingress handler capacity is exhausted; request was not accepted')
        with ExitStack() as cleanup:
            cleanup.callback(self._handler_slots.release)
            cleanup.callback(connection.close)
            with self._ingress_lock:
                finished = tuple(thread for thread in self._threads if not thread.is_alive())
                for thread in finished:
                    thread.join()
                    self._threads.remove(thread)
                thread = threading.Thread(target=self._handle, args=(connection,), name='foundry-rpc-handler')
                self._connections[connection] = Ingress(thread)
                self._threads.add(thread)
            def untrack():
                with self._ingress_lock:
                    self._connections.pop(connection)
                    self._threads.remove(thread)
            cleanup.callback(untrack)
            thread.start()
            cleanup.pop_all()
        return thread

    def join_handlers(self) -> None:
        """Join all owned handlers after the worker has drained or failed its accepted jobs."""
        with self._ingress_lock:
            threads = tuple(self._threads)
        for thread in threads:
            thread.join()
        with self._ingress_lock:
            self._threads.clear()

    def close(self) -> None:
        """Release transport resources after request_stop, worker drain and handler joins."""
        self._io.close()

    def _fatal(self, error: Exception) -> None:
        with self.submit_lock:
            if self.failure is None:
                self.failure = error
            self._stop_locked()
        self._close_readers()
        while True:
            try:
                job = self.q.get_nowait()
            except queue.Empty:
                return
            self._complete(job, error=self.failure)

    def run(self) -> None:
        self.worker = threading.Thread(target=self._worker, name='foundry-rpc-worker')
        self.publisher = threading.Thread(target=self._heartbeat, name='foundry-rpc-status')
        with ExitStack() as startup:
            startup.callback(self.close)
            self.state_dir.mkdir(parents=True, exist_ok=True)
            publish(self.portfile, json.dumps(dict(port=self.port, pid=os.getpid(), started=self.started)).encode())
            startup.callback(self.portfile.unlink, missing_ok=True)
            self._publish()
            self.worker.start()
            startup.callback(self.worker.join)
            startup.callback(self.request_stop)
            self.publisher.start()
            startup.callback(self.publisher.join)
            startup.callback(self.publisher_stop.set)
            startup.pop_all()
        try:
            while not self.stopping:
                for key, _ in self._selector.select():
                    if key.data == 'stop':
                        self._wake_read.recv(1)
                        break
                    connection, _ = self.sock.accept()
                    try:
                        with self.submit_lock:
                            if self.stopping:
                                connection.close()
                                break
                            self._start_handler(connection)
                    except queue.Full as error:
                        with connection:
                            connection.settimeout(1.)
                            try:
                                connection.sendall(json.dumps(error_response(None, error)).encode() + b'\n')
                            except OSError:
                                pass
        except Exception as error:
            self._fatal(error)
        finally:
            self.request_stop()
            self._selector.unregister(self.sock)
            self.sock.close()
            self.worker.join()
            self.publisher_stop.set()
            self.publisher.join()
            self.join_handlers()
            self.stopped = True
            try:
                try:
                    self.portfile.unlink(missing_ok=True)
                except OSError as error:
                    if self.failure is None:
                        self.failure = error
                    else:
                        self.failure.add_note(f'Endpoint cleanup also failed: {error}')
                try:
                    self._publish()
                except OSError as error:
                    if self.failure is None:
                        self.failure = error
                    else:
                        self.failure.add_note(f'Final status publication also failed: {error}')
            finally:
                self.close()
        if self.failure is not None:
            raise self.failure

    def _handle(self, connection: socket.socket) -> None:
        reserved = 0
        try:
            with connection:
                request_id = None
                try:
                    connection.settimeout(30.)
                    with connection.makefile('rb') as reader:
                        line = reader.readline(schema.MAX_REQUEST_BYTES + 1)
                        if not line:
                            return
                        if len(line) > schema.MAX_REQUEST_BYTES or not line.endswith(b'\n'):
                            raise ValueError('request exceeds its newline-delimited protocol limit')
                        raw = json.loads(line)
                        request_id = raw.get('id') if isinstance(raw, dict) else None
                        request = schema.parse(raw)
                        method = request['method']
                        payload = b''
                        if method in ('system.status', 'system.stop'):
                            self._phase(connection, 'control')
                        if method == 'research.register_batch':
                            size = request['params']['bytes']
                            self._reserve_upload(size)
                            reserved = size
                            payload = reader.read(size)
                            if len(payload) != size:
                                raise ValueError('array upload ended before its declared length')
                    self._phase(connection, 'ready')
                    if method == 'system.status':
                        value = self.status()
                    elif method == 'system.stop':
                        self.request_stop()
                        value = dict(state='draining', pending=self.q.qsize(), running=self.current,
                                     client_disconnect_cancels=False)
                    else:
                        job = Job(request, payload, reserved)
                        with self.submit_lock:
                            if self.stopping:
                                raise RuntimeError('service is draining; request was not accepted')
                            self.q.put_nowait(job)
                            reserved = 0
                        payload = b''
                        self._phase(connection, 'accepted')
                        job.done.wait()
                        if job.error is not None:
                            raise job.error
                        value = job.result
                    response = schema.result(request, value)
                except Exception as error:
                    response = error_response(request_id, error)
                try:
                    encoded = json.dumps(response, ensure_ascii=False, allow_nan=False).encode() + b'\n'
                    if len(encoded) > schema.MAX_RESPONSE_BYTES:
                        encoded = json.dumps(error_response(request_id, RuntimeError('response exceeds protocol byte limit'))).encode() + b'\n'
                    connection.sendall(encoded)
                except OSError:
                    # Disconnecting a client does not cancel an accepted command.
                    pass
        finally:
            self._release_upload(reserved)
            with self._ingress_lock:
                self._connections.pop(connection)
            self._handler_slots.release()

    def _complete(self, job: Job, *, result=None, error: Exception | None = None) -> None:
        job.result, job.error = result, error
        job.state = 'failed' if error is not None else 'succeeded'
        job.payload = b''
        self._release_upload(job.upload_bytes)
        job.upload_bytes = 0
        with self.submit_lock:
            self.served += 1
            self.jobs.append(dict(t=time.time(), op=job.request['method'], id=job.request['id'], state=job.state,
                dt=0. if job.started is None else time.time() - job.started,
                ok=error is None, n=0, variants=0, verdicts={}, source=None))
            if self._running is job:
                self.current = None
                self._running = None
        job.done.set()

    def _worker(self) -> None:
        while True:
            try:
                job = self.q.get(timeout=.1)
            except queue.Empty:
                if self.stopping:
                    return
                continue
            with self.submit_lock:
                failure = self.failure
                if failure is None:
                    job.started = time.time()
                    job.state = 'running'
                    self._running = job
                    self.current = dict(op=job.request['method'], id=job.request['id'], t0=job.started, state=job.state)
            if failure is not None:
                self._complete(job, error=failure)
                self._fatal(failure)
                return
            try:
                self._publish()
            except OSError as error:
                self._complete(job, error=error)
                self._fatal(error)
                return
            try:
                result = self.dispatch(job.request, job.payload)
            except Exception as error:
                self._complete(job, error=error)
                try:
                    with self.logfile.open('a', encoding='utf-8') as log:
                        log.write(traceback.format_exc())
                except OSError as logging_error:
                    self._fatal(logging_error)
                    return
            else:
                # Publication of status can fail after a successful CAS; the command result remains authoritative.
                self._complete(job, result=result)
            if self.failure is not None:
                return
            try:
                summary = self._read_summary()
                with self.submit_lock:
                    self._summary = summary
                self._publish()
            except Exception as error:
                self._fatal(error)
                return

    def dispatch(self, request: dict, payload: bytes = b''):
        method, parameters = request['method'], request['params']
        if method == 'research.describe':
            return self.backend.describe()
        if method == 'research.refresh':
            result = self.backend.refresh()
        elif method == 'research.register_batch':
            result = self.backend.register(decode_bundle(payload), parameters['context'])
        elif method == 'research.portfolio':
            result = self.backend.portfolio(parameters['context'])
        elif method == 'research.trial_weights':
            result = self.backend.trial_weights(parameters['weights'], parameters['context'])
        elif method == 'research.commit_weights':
            result = self.backend.commit_weights(parameters['weights'], parameters['context'])
        elif method == 'research.diagnose':
            result = self.backend.diagnose(parameters['context'])
        elif method == 'research.judge_family':
            families = tuple(from_json(FamilySeed | CandidateGrid, family) for family in parameters['families'])
            result = self.backend.judge_family(families, parameters['context'])
            self.n_expr += result.variants
        elif method == 'research.reweight':
            result = self.backend.reweight(parameters['context'], parameters['iterations'], parameters['count'], parameters['seed'])
        else:
            candidates = tuple(candidate_from_json(row) for row in parameters['candidates'])
            result = (self.backend.commit(candidates, parameters['context']) if method == 'research.commit'
                      else self.backend.evaluate(candidates, parameters['context']))
            self.n_expr += len(candidates)
        return json_value(asdict(result))


def error_response(request_id: str | None, error: Exception) -> dict:
    """Business rejection remains a result; only failures use JSON-RPC errors."""
    if isinstance(error, ContextConflict):
        code, reason = -32009, 'context_conflict'
    elif isinstance(error, queue.Full):
        code, reason = -32001, 'queue_full'
    elif isinstance(error, TimeoutError):
        code, reason = -32002, 'transport_timeout'
    elif isinstance(error, json.JSONDecodeError):
        code, reason = -32700, 'invalid_json'
    elif isinstance(error, (ValueError, TypeError)):
        code, reason = -32602, 'input_contract'
    else:
        code, reason = -32000, 'resource_failure'
    return dict(jsonrpc='2.0', id=request_id, error=dict(code=code, message=str(error),
        data=dict(type=type(error).__name__, reason=reason, client_disconnect_cancels=False)))
