"""Q25 常驻回测服务 —— 面板/CUDA context 常驻,请求自动排队,显存只有一份。

为什么要常驻:一次性 `run.py` 每次调用都要重付 ① import(numpy/pyarrow/torch)
② parquet 装载 + 特征派生 ③ CUDA context 创建 ④ **面板上传显存**。6 个 agent 并发迭代时
这四项被重复付 6×N 次,而 ④ 会同时存在 6 份副本 —— 这才是显存 OOM 的真正来源。
常驻后四项各付一次,请求侧只剩纯计算。

并发模型(OOM 在结构上不可能发生):
    accept 线程 ──▶ 每连接一个 handler 线程 ──▶ 全局 FIFO 队列 ──▶ **单** worker 线程
worker 是唯一的计算者 ⇒ 任一时刻只有一个 GPU 批在飞。多客户端不是竞争显存,是排队。

汇流(auto-batching):worker 取到一个 eval 作业后,再等 `COALESCE_WAIT` 秒,把队列里
**同窗、同不带 accept** 的 eval 作业并进同一个 GPU 批(上限 `COALESCE_MAX` 条),
算完按归属切回各自的客户端。6 个 agent 同时提交 = 一个大批而不是六个小批。

协议:TCP 127.0.0.1(WSL 下跑的是 Windows python,AF_UNIX 在 DrvFs 上不可用),
单请求单连接,收发各一行 JSON。端口写在 `.q25d.json`。

    python tmp/q25_funnel/daemon.py serve      # 前台起(调试用)
    python tmp/q25_funnel/daemon.py status
    python tmp/q25_funnel/daemon.py stop

正常情况下**不用手动起** —— `run.py` 首次调用会自动拉起并托管(见 client.py)。

面板:服务把状态写进 `.q25d_status.json`(有变化才写,空闲时 10s 一次心跳),由 `python -m monitoring`
显示;云端服务的这个文件由 `scripts/mirror_q25_status.py` 经一条 SSH 长连接按变化推到本地。
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
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

PORTFILE = HERE / '.q25d.json'
LOGFILE = HERE / '.q25d.log'
STATUSFILE = HERE / '.q25d_status.json'
HOST = '127.0.0.1'

COALESCE_WAIT = 0.005     # 秒:只合并同时到达的请求，避免等待超过计算时间
COALESCE_MAX = 512        # 单个 GPU 批的表达式上限
SLOW_JOB = 120.0          # 秒:超过则在日志里点名(排查队头阻塞)
PUBLISH_SECONDS = 0.5     # 秒:状态文件检查间隔
HEARTBEAT_SECONDS = 10.0  # 秒:状态没变也重写一次,面板据此区分"空闲"和"失联"


class Job:
    __slots__ = ('op', 'req', 'result', 'done', 't_in')

    def __init__(self, op: str, req: Dict):
        self.op, self.req = op, req
        self.result: Optional[Dict] = None
        self.done = threading.Event()
        self.t_in = time.time()


class Server:
    def __init__(self, port: int = 0):
        self.q: 'queue.Queue[Optional[Job]]' = queue.Queue()
        self.pending: 'collections.deque[Job]' = collections.deque()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((HOST, port))
        self.sock.listen(64)
        self.port = self.sock.getsockname()[1]
        self.t0 = time.time()
        self.served = 0
        self.n_expr = 0
        self.n_batch = 0
        self.stop_flag = False
        self._combo = None
        self._combo_mtime = -1.0
        # ---- 面板状态(写进 STATUSFILE)。worker 线程是唯一写者;jobs / sources 整体替换或 deque 追加,
        # 发布线程只做一次引用读取 + C 层拷贝 ⇒ 无需加锁。
        self.jobs: 'collections.deque[Dict]' = collections.deque(maxlen=32)   # 最近完成的作业
        self.sources: Dict[str, Dict] = {}    # 提交文件所在目录(一个 agent 一个)→ 累计轮数/公式/变体/判决
        self.cur: Optional[Dict] = None       # 在飞作业 {'op','n','source','t0','clients', 'stage','done','total'}
        self.clients = 0                      # 活跃连接数(handler 线程数)
        self.t_busy = 0.0                     # 累计计算耗时 ⇒ 真实吞吐 = n_expr/t_busy
        self.phase = 'warm'                   # warm | serve
        self.warm = '启动中…'

    # ------------------------------------------------------------ 生命周期
    def run(self):
        print(f'Q25 回测服务  pid={os.getpid()}  port={self.port}  预热中(首次约 20-60s)…\n'
              f'  日志 {LOGFILE}\n  面板数据 {STATUSFILE}', flush=True)
        threading.Thread(target=self._publish, daemon=True).start()
        self._warm()
        PORTFILE.write_text(json.dumps(
            dict(port=self.port, pid=os.getpid(), started=self.t0,
                 python=sys.executable), ensure_ascii=False), encoding='utf-8')
        _log(f'serve  port={self.port} pid={os.getpid()}')
        self.phase = 'serve'
        threading.Thread(target=self._worker, daemon=True).start()
        try:
            while not self.stop_flag:
                conn, _ = self.sock.accept()
                threading.Thread(target=self._handle, args=(conn,), daemon=True).start()
        finally:
            PORTFILE.unlink(missing_ok=True)
            _log('stopped')

    def _publish(self):
        """状态有变化(或满 HEARTBEAT_SECONDS)就原子重写 STATUSFILE。`now` 与累计时长每次都变,不算变化。"""
        last, written = None, 0.0
        while not self.stop_flag:
            st = self._status()
            key = json.dumps({k: v for k, v in st.items() if k not in ('now', 'elapsed_seconds', 'busy_seconds')},
                             ensure_ascii=False)
            if key != last or time.time() - written >= HEARTBEAT_SECONDS:
                temp = STATUSFILE.with_suffix('.tmp')
                temp.write_text(json.dumps(st, ensure_ascii=False), encoding='utf-8')
                while True:
                    try:
                        os.replace(temp, STATUSFILE)
                        break
                    except PermissionError:      # Windows:面板此刻正打开着它
                        time.sleep(0.02)
                last, written = key, time.time()
            time.sleep(PUBLISH_SECONDS)

    def _warm(self):
        """预热:装载面板 + 建 CUDA context + 上传显存。全部只此一次。"""
        import engine
        t = time.time()
        self.warm = '装载面板与派生特征…'
        d = engine.load()
        n = len(d['vocab'])
        _log(f'load   {n} operand  {time.time() - t:.1f}s')
        t = time.time()
        self.warm = '建组合参照 + 预热所选回测设备…'
        self._combo_fresh().ref('is')
        _log(f'combo  参照与双精度回测就绪 {time.time()-t:.1f}s')

    # ------------------------------------------------------------ 连接
    def _handle(self, conn: socket.socket):
        try:
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
                from gates import PROTOCOL
                res = dict(ok=True, protocol=PROTOCOL, pid=os.getpid(), port=self.port,
                           uptime=round(time.time() - self.t0, 1), served=self.served,
                           exprs=self.n_expr, batches=self.n_batch,
                           queue_depth=self.q.qsize() + len(self.pending))
            elif op == 'stop':
                res = dict(ok=True, stopping=True)
                self.stop_flag = True
                threading.Timer(0.2, self._selfpoke).start()
            elif op in ('eval', 'python', 'family'):
                from gates import PROTOCOL
                assert req.get('protocol') == PROTOCOL, '客户端 gate 协议过期;请同步代码'
                job = Job(op, req)
                self.clients += 1        # 活跃客户端 = 正卡在 done.wait() 上的连接数
                try:
                    self.q.put(job)
                    job.done.wait()
                finally:
                    self.clients -= 1
                res = job.result
            else:
                res = dict(ok=False, error=f'未知 op {op!r}')
            conn.settimeout(120.0)
            conn.sendall(json.dumps(res, ensure_ascii=False).encode('utf-8') + b'\n')
        except Exception:
            _log('handle 异常\n' + traceback.format_exc())
            try:
                conn.sendall(json.dumps(dict(ok=False, error=traceback.format_exc()),
                                        ensure_ascii=False).encode('utf-8') + b'\n')
            except Exception:
                pass
        finally:
            conn.close()

    def _selfpoke(self):
        try:
            socket.create_connection((HOST, self.port), timeout=2.0).close()
        except Exception:
            pass

    # ------------------------------------------------------------ 计算(唯一计算者)
    def _next(self, timeout=None) -> Optional[Job]:
        if self.pending:
            return self.pending.popleft()
        return self.q.get(timeout=timeout) if timeout is not None else self.q.get()

    def _worker(self):
        while True:
            job = self._next()
            if job is None:
                return
            batch = [job]
            if job.op == 'eval' and not job.req.get('accept'):
                batch += self._coalesce(job)
            t = time.time()
            self.cur = dict(op=job.op, clients=len(batch), t0=t, source=job.req.get('source') or job.req.get('path'),
                            n=sum(len(j.req.get('exprs', j.req.get('seeds', []))) for j in batch))
            try:
                self._run(batch)
            except Exception:
                err = traceback.format_exc()
                _log('worker 异常\n' + err)
                for j in batch:
                    j.result = dict(ok=False, error=err)
            dt = time.time() - t
            self.t_busy += dt
            self._log_job(batch, dt)   # 先并入累计、再撤在飞批次:反过来漏斗会在两者之间塌一下
            self.cur = None
            _free_gpu()          # 仅在高水位回收缓存；常规批次复用显存分配
            for j in batch:
                if j.result is None:
                    j.result = dict(ok=False, error='worker 未产出结果')
                j.done.set()
            self.served += len(batch)
            self.n_batch += 1
            if dt > SLOW_JOB:
                _log(f'slow   {dt:.0f}s  ops={[j.op for j in batch]}')

    def _log_job(self, batch: List[Job], dt: float) -> None:
        """作业结果并入面板状态(jobs 逐作业一行、sources 按提交目录累计),日志打一行摘要。"""
        for j in batch:
            res, source = j.result or {}, j.req.get('source') or j.req.get('path')
            job = dict(t=time.time(), op=j.op, source=source, dt=round(dt, 1), ok=bool(res.get('ok')))
            if j.op == 'family' and res.get('ok'):
                verdicts = dict(collections.Counter(f['verdict'] for f in res['families']))
                job.update(n=len(j.req['seeds']), variants=res['variants'], verdicts=verdicts)
                _log(f'族{self.n_batch + 1:<4d} {job["n"]:3d} 条公式 {res["variants"]:6d} 个变体  {dt:5.1f}s  '
                     + ' '.join(f'{k} {v}' for k, v in verdicts.items()) + f'  {source or ""}')
            else:
                rows = res.get('rows', [])
                verdicts = dict(collections.Counter('通过' if r.get('passed') else '未过' for r in rows))
                job.update(n=len(rows), variants=len(rows), verdicts=verdicts)
                _log(f'{j.op:<6s}{self.n_batch + 1:<4d} {len(rows):5d} 条  {dt:5.1f}s  过 {verdicts.get("通过", 0)}  {source or ""}')
            self.jobs.append(job)
            if source:                            # 一个提交目录 = 一个 agent;整体替换,发布线程读到的永远是完整的一份
                key = PurePosixPath(source.replace('\\', '/')).parent.name
                old = self.sources.get(key, dict(rounds=0, formulas=0, variants=0, verdicts={}))
                merged = collections.Counter(old['verdicts']) + collections.Counter(job['verdicts'])
                self.sources = dict(self.sources, **{key: dict(
                    rounds=old['rounds'] + 1, formulas=old['formulas'] + job['n'],
                    variants=old['variants'] + job['variants'], verdicts=dict(merged), last=job['t'])})

    def _status(self) -> Dict:
        """面板的唯一数据源:只给数,不排版(版面在 monitoring/service.py)。时刻一律是本机时钟。"""
        ref = self._combo._ref.get('is') if self._combo is not None else None
        cur = self.cur
        return dict(
            kind='service', phase=self.phase, warm=self.warm, pid=os.getpid(), now=time.time(),
            elapsed_seconds=time.time() - self.t0, busy_seconds=self.t_busy,
            device=_device() if self.phase == 'serve' else None, vram=_vram(),
            queue=self.q.qsize() + len(self.pending), clients=self.clients,
            served=self.served, exprs=self.n_expr, batches=self.n_batch,
            portfolio={k: ref[k] for k in ('n', 'sharpe', 'cagr', 'maxdd')} if ref else None,
            cur=dict(cur) if cur else None, jobs=list(self.jobs), sources=self.sources)

    def _coalesce(self, first: Job) -> List[Job]:
        """收拢同窗 eval 作业。遇到不可合并的作业放回 pending,**不改变其顺序**。"""
        win = first.req.get('window', 'is')
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
            if (j.op == 'eval' and not j.req.get('accept')
                    and j.req.get('window', 'is') == win):
                got.append(j)
                n += len(j.req.get('exprs', []))
            else:
                self.pending.append(j)
                break
        return got

    def _run(self, batch: List[Job]) -> None:
        if batch[0].op == 'family':                  # never coalesced: one round of one agent is already thousands of variants
            import family
            j = batch[0]
            j.result = dict(ok=True, served_by='daemon', **family.expand(j.req['seeds'], self._combo_fresh().ref('is'), self._tick))
            self.n_expr += j.result['variants']
            return
        if batch[0].op == 'python':
            j = batch[0]
            j.result = dict(ok=True, rows=self._eval_python(j.req), served_by='daemon',
                            coalesced=1)
            return
        window = batch[0].req.get('window', 'is')
        accept = bool(batch[0].req.get('accept'))
        owner, exprs = [], []
        for i, j in enumerate(batch):
            for s in j.req.get('exprs', []):
                owner.append(i)
                exprs.append(s)
        rows = self._eval_exprs(exprs, window, accept)
        self.n_expr += len(exprs)
        out: List[List[Dict]] = [[] for _ in batch]
        for o, row in zip(owner, rows):
            out[o].append(row)
        for j, rs in zip(batch, out):
            j.result = dict(ok=True, rows=rs, served_by='daemon',
                            coalesced=len(batch), batch_exprs=len(exprs))

    # ------------------------------------------------------------ 求值
    def _combo_fresh(self):
        """组合库(门的参照)随盘上的文件变动自动重载;写入由单 worker 串行化 ⇒ 并发 --accept 不互相覆盖。"""
        from combolib import STORE, ComboLib
        m = STORE.stat().st_mtime if STORE.exists() else 0.0
        if self._combo is None or m != self._combo_mtime:
            self._combo, self._combo_mtime = ComboLib.load(), m
        return self._combo

    def _tick(self, stage: str, done: int, total: int) -> None:
        """批内进度 → 面板。一个批可能跑几十秒,只在批末更新会让中途看着像卡死。"""
        cur = self.cur
        if cur.get('stage') != stage:
            cur['stage_t0'] = time.time()        # 剩余时间按**本阶段**的速度估,不能拿 GPU 段的耗时去摊 CPU 段
        cur['stage'], cur['done'], cur['total'] = stage, done, total

    def _eval_exprs(self, exprs: List[str], window: str, accept: bool) -> List[Dict]:
        import engine
        import gates
        from fastexpr import validate
        combo = self._combo_fresh()
        ref = combo.ref(window)
        vocab = engine.operands()
        rows: Dict[str, Dict] = {}
        uniq = list(dict.fromkeys(exprs))            # 去重:同式只算一次,回填时再展开
        if not accept:
            for lo in range(0,len(uniq),engine.CUDA.candidate_batch):
                self._tick('GPU 自适应精度组合批量',lo,len(uniq))
                batch = engine.simulate_many(uniq[lo:lo+engine.CUDA.candidate_batch],window)
                for r,g in zip(batch,gates.run_batch(batch,ref)):
                    rows[r['expr']] = gates.rowify(r,g)
            self._tick('收尾',len(uniq),len(uniq))
            return [rows[s] for s in exprs]
        for k,s in enumerate(uniq):
            self._tick('CUDA 组合准入',k,len(uniq))
            r=engine.simulate(s,window)
            g=gates.run(r,ref)
            row=gates.rowify(r,g)
            if g['passed']:
                row.update(accepted=True,weight=combo.add(r,g))
                ref=combo.ref(window)
            rows[s]=row
        n_surv=len(uniq)
        self._tick('收尾', n_surv, n_surv)
        if accept and any(rows[s]['accepted'] for s in uniq):
            combo.save()
            self._combo_mtime = -1.0                 # 强制下次重载,与外部写入对齐
        return [rows[s] for s in exprs]

    def _eval_python(self, req: Dict) -> List[Dict]:
        import gates
        from combolib import evaluate_python_source
        try:  # user-supplied Python source is a system boundary
            rows = evaluate_python_source(req['src'],req.get('path','<python-alpha>'),
                                          req.get('window','is'),self._combo_fresh(),req.get('accept',False),req.get('names'))
        except Exception:
            return [gates.reject_row(req.get('path','<python-alpha>'),traceback.format_exc())]
        if any(r['accepted'] for r in rows):
            self._combo_mtime = -1.0
        return rows


# ---------------------------------------------------------------- CLI
def _vram() -> str:
    """CuPy 内存池高水位(= 本进程真实占用的显存量级)。无卡 → '-'。"""
    try:
        import cupy as cp
        p = cp.get_default_memory_pool()
        return f'{p.used_bytes() / 2**20:.0f}/{p.total_bytes() / 2**20:.0f}MiB'
    except Exception:
        return '-'


_DEVICE: Optional[str] = None


def _device() -> str:
    """报告实际组合回测设备,不沿用旧因子面板的 f32 标签。"""
    global _DEVICE
    if _DEVICE is None:
        try:
            import engine
            import cupy as cp
            _DEVICE = f'{cp.cuda.runtime.getDeviceProperties(0)["name"].decode()} · combo {engine.CUDA.precision}'
        except Exception:
            _DEVICE = 'CUDA unavailable'
    return _DEVICE


def _free_gpu() -> None:
    try:
        import cupy as cp
        pool = cp.get_default_memory_pool()
        if pool.total_bytes() > 20 * 1024**3:
            pool.free_all_blocks()
    except Exception:
        _log('CUDA memory pool cleanup failed\n' + traceback.format_exc())


_LOG_FH = None


def _log(msg: str) -> None:
    global _LOG_FH
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    if _LOG_FH is None:
        _LOG_FH = open(LOGFILE, 'a', encoding='utf-8', buffering=1)
    _LOG_FH.write(line + '\n')


def main() -> None:
    argv = sys.argv[1:]
    cmd = argv[0] if argv else 'serve'
    if cmd == 'serve':
        try:
            port = next((int(x) for x in argv[1:] if x.isdigit()), 0)
            Server(port).run()
        except BaseException:
            # stderr 就是那个控制台窗口,窗口一关 traceback 也没了 ⇒ 显式落盘。
            _log('CRASH\n' + traceback.format_exc())
            raise
        return
    import client
    if cmd == 'status':
        p = client.request(dict(op='ping'), timeout=5.0)
        print(json.dumps(p, ensure_ascii=False, indent=2) if p else '未运行')
    elif cmd == 'stop':
        print(client.request(dict(op='stop'), timeout=5.0) or '未运行')
    else:
        sys.exit(f'未知命令 {cmd!r}(serve | status | stop)')


if __name__ == '__main__':
    main()
