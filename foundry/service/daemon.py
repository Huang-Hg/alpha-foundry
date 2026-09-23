"""The resident evaluation service: one process owns the GPU, the panel and the portfolio reference; every client's
requests queue into it.

Why resident: a one-shot evaluation pays import, parquet load and feature derivation, CUDA context creation and the
panel upload every time, and six agents iterating concurrently would hold six panel copies on the card. Resident,
each is paid once and requests are pure computation.

Concurrency (out-of-memory is structurally impossible):
    accept thread -> one handler thread per connection -> one FIFO queue -> ONE worker thread
The worker is the only computer, so one GPU batch is in flight at any time; clients queue rather than compete.
Coalescing: after taking an evaluation job the worker waits COALESCE_WAIT for more evaluation jobs without --accept
and merges them into one GPU batch (up to COALESCE_MAX formulas), then splits the rows back per client.

Protocol: TCP on 127.0.0.1 (the Windows interpreter under WSL cannot use AF_UNIX on DrvFs), one request per
connection, one JSON line each way; the port is in tmp/service/port.json. The service writes its status for the
dashboard to tmp/service/status.json (on change, and every HEARTBEAT_SECONDS when idle).
"""
from __future__ import annotations

import collections
import json
import os
import queue
import socket
import sys
import threading
import time
import traceback
from pathlib import PurePosixPath
from typing import Dict, List, Optional

from foundry.config import MarketConfig
from foundry.core.files import publish
from foundry.portfolio.library import PROTOCOL, STORE, Library
from foundry.service.protocol import HOST, LOGFILE, PORTFILE, STATE, STATUSFILE

COALESCE_WAIT = 0.005     # seconds: merge only requests that arrive together, never wait longer than a computation
COALESCE_MAX = 512        # formulas per GPU batch
SLOW_JOB = 120.0          # seconds: logged by name (head-of-line blocking)
PUBLISH_SECONDS = 0.5
HEARTBEAT_SECONDS = 10.0  # rewrite the status even when nothing changed, so the dashboard tells idle from lost


class Job:
    __slots__ = ('op', 'req', 'result', 'done', 't_in')

    def __init__(self, op: str, req: Dict):
        self.op, self.req = op, req
        self.result: Optional[Dict] = None
        self.done = threading.Event()
        self.t_in = time.time()


class Server:
    def __init__(self, port: int = 0):
        STATE.mkdir(parents=True, exist_ok=True)
        self.q: 'queue.Queue[Optional[Job]]' = queue.Queue()
        self.pending: 'collections.deque[Job]' = collections.deque()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((HOST, port))
        self.sock.listen(64)
        self.port = self.sock.getsockname()[1]
        self.t0 = time.time()
        self.served = self.n_expr = self.n_batch = 0
        self.stop_flag = False
        self.engine = self.portfolio = None
        self._mtime = -1.0
        # Dashboard state. The worker is the only writer; jobs/sources are replaced whole or appended, the publisher
        # only reads references, so no lock is needed.
        self.jobs: 'collections.deque[Dict]' = collections.deque(maxlen=32)
        self.sources: Dict[str, Dict] = {}   # submitting directory (one agent each) -> rounds, formulas, variants, verdicts
        self.cur: Optional[Dict] = None      # the job on the GPU
        self.clients = 0
        self.t_busy = 0.0
        self.phase, self.warm = 'warm', '启动中…'

    # ------------------------------------------------------------ lifecycle
    def run(self):
        print(f'评价服务  pid={os.getpid()}  port={self.port}  预热中(首次约 20-60s)…\n'
              f'  日志 {LOGFILE}\n  面板数据 {STATUSFILE}', flush=True)
        threading.Thread(target=self._publish, daemon=True).start()
        self._warm()
        PORTFILE.write_text(json.dumps(dict(port=self.port, pid=os.getpid(), started=self.t0, python=sys.executable),
                                       ensure_ascii=False), encoding='utf-8')
        log(f'serve  port={self.port} pid={os.getpid()}')
        self.phase = 'serve'
        threading.Thread(target=self._worker, daemon=True).start()
        try:
            while not self.stop_flag:
                conn, _ = self.sock.accept()
                threading.Thread(target=self._handle, args=(conn,), daemon=True).start()
        finally:
            PORTFILE.unlink(missing_ok=True)
            log('stopped')

    def _publish(self):
        last, written = None, 0.0
        while not self.stop_flag:
            st = self._status()
            key = json.dumps({k: v for k, v in st.items() if k not in ('now', 'elapsed_seconds', 'busy_seconds')}, ensure_ascii=False)
            if key != last or time.time() - written >= HEARTBEAT_SECONDS:
                publish(STATUSFILE, json.dumps(st, ensure_ascii=False).encode('utf-8'))
                last, written = key, time.time()
            time.sleep(PUBLISH_SECONDS)

    def _warm(self):
        """Load the panel, create the CUDA context, upload, build the reference: once."""
        from foundry.engine import Engine
        t = time.time()
        self.warm = '装载面板与派生特征…'
        self.engine = Engine()
        log(f'load   {len(self.engine.vocabulary)} fields  {time.time() - t:.1f}s')
        t = time.time()
        self.warm = '建组合参照…'
        self._portfolio().reference
        log(f'portfolio reference ready {time.time() - t:.1f}s')

    # ------------------------------------------------------------ connections
    def _handle(self, conn: socket.socket):
        try:  # a client connection is a system boundary: any failure is reported back to that client
            conn.settimeout(30.0)
            buf = b''
            while not buf.endswith(b'\n'):
                b = conn.recv(1 << 20)
                if not b:
                    return
                buf += b
            req = json.loads(buf.decode('utf-8'))
            op = req.get('op', '')
            if op == 'ping':
                res = dict(ok=True, protocol=PROTOCOL, pid=os.getpid(), port=self.port, uptime=round(time.time() - self.t0, 1),
                           served=self.served, exprs=self.n_expr, batches=self.n_batch,
                           queue_depth=self.q.qsize() + len(self.pending))
            elif op == 'stop':
                res = dict(ok=True, stopping=True)
                self.stop_flag = True
                threading.Timer(0.2, self._selfpoke).start()
            elif op in ('eval', 'family'):
                assert req.get('protocol') == PROTOCOL, 'client admission protocol differs; sync the code'
                job = Job(op, req)
                self.clients += 1
                try:
                    self.q.put(job)
                    job.done.wait()
                finally:
                    self.clients -= 1
                res = job.result
            else:
                res = dict(ok=False, error=f'unknown op {op!r}')
            conn.settimeout(120.0)
            conn.sendall(json.dumps(res, ensure_ascii=False).encode('utf-8') + b'\n')
        except Exception:
            log('handle error\n' + traceback.format_exc())
            conn.sendall(json.dumps(dict(ok=False, error=traceback.format_exc()), ensure_ascii=False).encode('utf-8') + b'\n')
        finally:
            conn.close()

    def _selfpoke(self):
        socket.create_connection((HOST, self.port), timeout=2.0).close()

    # ------------------------------------------------------------ computation (the only computer)
    def _next(self, timeout=None) -> Optional[Job]:
        if self.pending:
            return self.pending.popleft()
        return self.q.get(timeout=timeout) if timeout is not None else self.q.get()

    def _worker(self):
        while True:
            job = self._next()
            if job is None:
                return
            batch = [job] + (self._coalesce(job) if job.op == 'eval' and not job.req.get('accept') else [])
            t = time.time()
            self.cur = dict(op=job.op, clients=len(batch), t0=t, source=job.req.get('source'),
                            n=sum(len(j.req.get('exprs', j.req.get('seeds', []))) for j in batch))
            try:  # a failing job is reported to its clients; the service keeps serving
                self._run(batch)
            except Exception:
                err = traceback.format_exc()
                log('worker error\n' + err)
                for j in batch:
                    j.result = dict(ok=False, error=err)
            dt = time.time() - t
            self.t_busy += dt
            self._log_job(batch, dt)   # merge into the totals before clearing the in-flight job
            self.cur = None
            release_pool()
            for j in batch:
                j.done.set()
            self.served += len(batch)
            self.n_batch += 1
            if dt > SLOW_JOB:
                log(f'slow   {dt:.0f}s  ops={[j.op for j in batch]}')

    def _log_job(self, batch: List[Job], dt: float) -> None:
        for j in batch:
            res, source = j.result, j.req.get('source')
            job = dict(t=time.time(), op=j.op, source=source, dt=round(dt, 1), ok=bool(res.get('ok')))
            if j.op == 'family' and res.get('ok'):
                verdicts = dict(collections.Counter(f['verdict'] for f in res['families']))
                job.update(n=len(j.req['seeds']), variants=res['variants'], verdicts=verdicts)
                log(f'family{self.n_batch + 1:<4d} {job["n"]:3d} formulas {res["variants"]:6d} variants  {dt:5.1f}s  '
                    + ' '.join(f'{k} {v}' for k, v in verdicts.items()) + f'  {source or ""}')
            else:
                rows = res.get('rows', [])
                verdicts = dict(collections.Counter('通过' if r.get('passed') else '未过' for r in rows))
                job.update(n=len(rows), variants=len(rows), verdicts=verdicts)
                log(f'{j.op:<6s}{self.n_batch + 1:<4d} {len(rows):5d} rows  {dt:5.1f}s  passed {verdicts.get("通过", 0)}  {source or ""}')
            self.jobs.append(job)
            if source:                            # one submitting directory = one agent
                key = PurePosixPath(source.replace('\\', '/')).parent.name
                old = self.sources.get(key, dict(rounds=0, formulas=0, variants=0, verdicts={}))
                merged = collections.Counter(old['verdicts']) + collections.Counter(job['verdicts'])
                self.sources = dict(self.sources, **{key: dict(
                    rounds=old['rounds'] + 1, formulas=old['formulas'] + job['n'],
                    variants=old['variants'] + job['variants'], verdicts=dict(merged), last=job['t'])})

    def _status(self) -> Dict:
        """The dashboard's only data source: numbers, no layout (foundry/monitor/view.ServiceScreen lays it out)."""
        ref = self.portfolio._reference if self.portfolio is not None else None
        cur = self.cur
        return dict(kind='service', market=MarketConfig().name, phase=self.phase, warm=self.warm, pid=os.getpid(), now=time.time(),
                    elapsed_seconds=time.time() - self.t0, busy_seconds=self.t_busy,
                    device=device(self.engine) if self.phase == 'serve' else None, vram=vram(),
                    queue=self.q.qsize() + len(self.pending), clients=self.clients,
                    served=self.served, exprs=self.n_expr, batches=self.n_batch,
                    portfolio=dict(n=ref.n, sharpe=ref.wsharpe, cagr=ref.cagr, maxdd=ref.maxdd) if ref else None,
                    cur=dict(cur) if cur else None, jobs=list(self.jobs), sources=self.sources)

    def _coalesce(self, first: Job) -> List[Job]:
        """Gather evaluation jobs that arrive together; a job that cannot join goes back in order."""
        got, n, deadline = [], len(first.req.get('exprs', [])), time.time() + COALESCE_WAIT
        while n < COALESCE_MAX:
            left = deadline - time.time()
            if left <= 0:
                break
            try:
                j = self._next(timeout=left)
            except queue.Empty:
                break
            if j is None:
                self.q.put(None)
                break
            if j.op == 'eval' and not j.req.get('accept'):
                got.append(j)
                n += len(j.req.get('exprs', []))
            else:
                self.pending.append(j)
                break
        return got

    def _run(self, batch: List[Job]) -> None:
        if batch[0].op == 'family':               # never coalesced: one round of one agent is thousands of variants
            from foundry.portfolio import family
            j = batch[0]
            j.result = dict(ok=True, **family.expand(self._portfolio(), j.req['seeds'], self._tick))
            self.n_expr += j.result['variants']
            return
        owner, exprs = [], []
        for i, j in enumerate(batch):
            for s in j.req.get('exprs', []):
                owner.append(i)
                exprs.append(s)
        rows = self._evaluate(exprs, bool(batch[0].req.get('accept')))
        self.n_expr += len(exprs)
        out: List[List[Dict]] = [[] for _ in batch]
        for o, row in zip(owner, rows):
            out[o].append(row)
        for j, rs in zip(batch, out):
            j.result = dict(ok=True, rows=rs, coalesced=len(batch), batch_exprs=len(exprs))

    def _portfolio(self):
        """The library, reloaded whenever the file changes; writes are serialised by the single worker."""
        from foundry.portfolio.portfolio import Portfolio
        m = STORE.stat().st_mtime if STORE.exists() else 0.0
        if self.portfolio is None or m != self._mtime:
            self.portfolio, self._mtime = Portfolio(self.engine, Library.load()), m
        return self.portfolio

    def _tick(self, stage: str, done: int, total: int) -> None:
        """Progress inside a batch (a batch can take tens of seconds)."""
        cur = self.cur
        if cur.get('stage') != stage:
            cur['stage_t0'] = time.time()        # remaining time is estimated from this stage's own speed
        cur['stage'], cur['done'], cur['total'] = stage, done, total

    def _evaluate(self, exprs: List[str], accept: bool) -> List[Dict]:
        return evaluate(self._portfolio(), exprs, accept, self._tick)


def evaluate(portfolio, exprs: List[str], accept: bool, tick=lambda stage, done, total: None) -> List[Dict]:
    """Rows of formulas against the portfolio. accept=True admits the passing ones one by one (each next formula is
    judged against the portfolio that includes the previous ones) and saves the library."""
    from foundry.portfolio.admission import rowify
    engine = portfolio.engine
    uniq, rows = list(dict.fromkeys(exprs)), {}
    if not accept:
        for lo in range(0, len(uniq), engine.candidate_batch):
            tick('GPU 自适应精度组合批量', lo, len(uniq))
            batch = engine.formulas(uniq[lo:lo + engine.candidate_batch])
            for r, g in zip(batch, portfolio.admit(batch)):
                rows[r['expr']] = rowify(r, g)
    else:
        for k, s in enumerate(uniq):
            tick('CUDA 组合准入', k, len(uniq))
            r = engine.formulas([s])[0]
            g = portfolio.admit([r])[0]
            rows[s] = rowify(r, g)
            if g['passed']:
                rows[s].update(accepted=True, weight=portfolio.accept(r, g))
        if any(r['accepted'] for r in rows.values()):
            portfolio.library.save()
    tick('收尾', len(uniq), len(uniq))
    return [rows[s] for s in exprs]


def release_pool() -> None:
    """Return cached device memory only past a high-water mark; ordinary batches reuse their allocations."""
    import cupy as cp
    pool = cp.get_default_memory_pool()
    if pool.total_bytes() > 20 * 1024**3:
        pool.free_all_blocks()


def vram() -> str:
    import cupy as cp
    pool = cp.get_default_memory_pool()
    return f'{pool.used_bytes() / 2**20:.0f}/{pool.total_bytes() / 2**20:.0f}MiB'


def device(engine) -> str:
    import cupy as cp
    return f'{cp.cuda.runtime.getDeviceProperties(0)["name"].decode()} · combo {engine.precision}'


_LOG = None


def log(msg: str) -> None:
    global _LOG
    if _LOG is None:
        STATE.mkdir(parents=True, exist_ok=True)
        _LOG = open(LOGFILE, 'a', encoding='utf-8', buffering=1)
    _LOG.write(f'[{time.strftime("%H:%M:%S")}] {msg}\n')
