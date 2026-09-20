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
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

PORTFILE = HERE / '.q25d.json'
LOGFILE = HERE / '.q25d.log'
HOST = '127.0.0.1'

COALESCE_WAIT = 0.005     # 秒:只合并同时到达的请求，避免等待超过计算时间
COALESCE_MAX = 512        # 单个 GPU 批的表达式上限
SLOW_JOB = 120.0          # 秒:超过则在日志里点名(排查队头阻塞)


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
        # ---- 面板状态(dash.py 通过 op=stats 读)。只由 worker 单线程写 ⇒ 无需加锁。
        self.funnel: 'collections.Counter[str]' = collections.Counter()
        self.lanes: 'collections.Counter[str]' = collections.Counter()
        self.best: Optional[Dict] = None      # {'sharpe','expr'} —— 全程最高
        self.events: 'collections.deque[Dict]' = collections.deque(maxlen=64)
        self.cur: Optional[Dict] = None       # 在飞批次 {'n','t0','op','clients'}
        self.clients = 0                      # 活跃连接数(handler 线程数)
        self.t_busy = 0.0                     # 累计计算耗时 ⇒ 真实吞吐 = n_expr/t_busy
        self.phase = 'warm'                   # warm | serve
        self.warm = '启动中…'

    # ------------------------------------------------------------ 生命周期
    def run(self):
        """**一个进程、一个窗口**:服务在这儿算,面板也在这儿画。

        ⚠️ 第一件事永远是无条件打一段横幅。以前这里先判 `isatty` 再决定要不要起面板线程,
        而面板线程内部又把异常只写进日志文件 —— 两条路都通向**一个永远全黑的窗口**,
        服务却在正常跑。窗口是黑的时候人唯一能得到的信息就是"没信息",无法诊断。
        横幅无条件打 ⇒ "窗口全黑"这个现象从此只可能意味着"进程根本没起来"。"""
        banner = (f'Q25 回测服务  pid={os.getpid()}  port={self.port}\n'
                  f'  预热中(装载面板 + CUDA context + 上传显存,首次约 20-60s)…\n'
                  f'  日志 {LOGFILE}')
        print(banner, flush=True)
        _con = self._out()                  # 横幅必须也进控制台窗口:否则预热那 20-60s
        if _con is not sys.stdout:          # 它还是全黑的,而这正是用户最需要反馈的一段
            try:
                _con.write(banner + '\n')
            except Exception:
                pass
        threading.Thread(target=self._display, daemon=True).start()
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

    def _out(self):
        """面板往哪写。**优先 `CONOUT$`(那个黑窗口),而不是 stdout。**

        2026-08-13 实测,推翻了之前的结论:
          · WSL interop 起的 Windows 进程**一律有控制台窗口** —— `GetConsoleWindow()≠0`,
            连 `DETACHED_PROCESS` 也一样(interop 压根不理会 `creationflags`)。
            用户看到的那个黑框就是它。
          · 但进程的 stdout 被接回 WSL 管道 ⇒ `sys.stdout.isatty()` 是 **False**,
            而且写进去的东西跑到 WSL/日志文件里,**永远不会出现在那个窗口上**。
            这就是"黑窗口什么都不显示"的全部原因。
          · `CONOUT$` 绕开 stdout 重定向直写控制台,实测 `isatty()` 是 **True**
            ⇒ 那个窗口支持 `rich.Live` 原地重绘。

        ⚠️ 之前我用 `MainWindowHandle == 0` 判定"拿不到窗口"是**错的**:控制台窗口属于
        `conhost.exe`,不属于本进程,所以任何控制台程序的 `MainWindowHandle` 都是 0。
        判有没有控制台只能用 `GetConsoleWindow()`。
        """
        if sys.platform == 'win32':
            try:
                f = open('CONOUT$', 'w', encoding='utf-8', buffering=1)
                if f.isatty():
                    return f
                f.close()
            except Exception:
                pass
        return sys.stdout

    def _mode(self, out=None) -> str:
        """面板模式:live(原地重绘)| frames(定时整帧)| off。命令行 `--panel <mode>` 覆盖。

        ⚠️ **覆盖开关必须走 `sys.argv`,不能走环境变量。** 本机是 WSL 调 Windows python,
        `FOO=bar win_python.exe` 里的 `FOO` **根本到不了子进程**(实测 `os.environ` 里是
        None)。用环境变量做开关会静默失效 —— 开关像是没生效,实际是压根没传进来。

        自动判断按 `_out()` 选出的目标来:CONOUT$ 是真终端 ⇒ live(画进那个黑窗口)。
        只有连控制台都没有(纯管道 / Linux 重定向)才退到 off,`--panel frames` 可强制出帧。"""
        a = sys.argv
        for i, x in enumerate(a):
            if x == '--panel' and i + 1 < len(a):
                m = a[i + 1].strip().lower()
                if m in ('live', 'frames', 'off'):
                    return m
            elif x.startswith('--panel='):
                m = x.split('=', 1)[1].strip().lower()
                if m in ('live', 'frames', 'off'):
                    return m
        out = out if out is not None else sys.stdout
        return 'live' if out.isatty() else 'off'

    def _display(self):
        """面板线程。渲染失败绝不能拖垮服务 ⇒ 整体兜底。

        兜底**不能只写日志文件**:窗口是给人看的,rich 挂了就退化成纯文本状态行,
        让窗口继续有东西。以前这里失败即静默,于是"面板没出来"变成了不可诊断的现象。"""
        out = self._out()
        mode = self._mode(out)
        _log(f'panel  target={"CONOUT$" if out is not sys.stdout else "stdout"} '
             f'isatty={out.isatty()} mode={mode}')
        if mode == 'off':
            return
        try:
            import dash
            from rich.console import Console
            if mode == 'live':
                from rich.live import Live
                console = Console(file=out, highlight=False)
                with Live(dash.render(self._stats(), dash.fit(console, out), owner=True), console=console,
                          refresh_per_second=4, screen=False) as live:
                    while not self.stop_flag:
                        live.update(dash.render(self._stats(), dash.fit(console, out), owner=True))
                        time.sleep(0.25)
            else:
                console = Console(file=out, width=84, highlight=False)
                while not self.stop_flag:
                    console.print(dash.render(self._stats(), owner=True))
                    time.sleep(5.0)
        except Exception:
            _log('面板渲染失败,退化为纯文本状态行\n' + traceback.format_exc())
            print('[panel] rich 渲染失败,退化为纯文本(详见日志)', flush=True)
            while not self.stop_flag:
                s = self._stats()
                cur = s.get('cur')
                print(f"[{time.strftime('%H:%M:%S')}] "
                      + (f"计算中 {cur['n']} 条 {cur['elapsed']:.0f}s "
                         f"{cur.get('stage', '')} {cur.get('done', 0)}/{cur.get('total', 0)}"
                         if cur else '空闲')
                      + f" · 队列 {s['queue']} · 已算 {s['exprs']:,} · {s['vram']}", flush=True)
                time.sleep(5.0)

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
            elif op == 'stats':
                res = self._stats()
            elif op == 'stop':
                res = dict(ok=True, stopping=True)
                self.stop_flag = True
                threading.Timer(0.2, self._selfpoke).start()
            elif op in ('eval', 'python'):
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
            self.cur = dict(op=job.op, clients=len(batch), t0=t, funnel=collections.Counter(), best=None,
                            n=sum(len(j.req.get('exprs', [])) for j in batch))
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
        """把一个作业的结果**并入面板状态**,并在服务端终端打一行紧凑摘要。

        分工:详细的漏斗/吞吐/峰值留给 `dash.py`(它按 op=stats 拉结构化状态,能重绘、
        能排版);这里只打一行"发生了什么",够在服务端终端确认没堵、没崩即可。
        以前把 12 个字段挤在一行里 —— 信息全在,但没人读得下去。"""
        rows = [r for j in batch for r in (j.result or {}).get('rows', [])]
        if not rows:
            _log(f'{batch[0].op} 无产出  {dt:.1f}s')
            return
        for r in rows:
            self.funnel[r.get('stopped_at') or 'PASS'] += 1
            self.lanes[f'{r.get("lane", "?")}/{r.get("backend", "cpu")}'] += 1
        n_pass = sum(1 for r in rows if r.get('passed'))
        best = max(((r['sharpe'], r['expr']) for r in rows
                    if r.get('sharpe') is not None), default=None)
        if best and (self.best is None or best[0] > self.best['sharpe']):
            self.best = dict(sharpe=best[0], expr=best[1], t=time.time())
        self.events.append(dict(
            t=time.time(), n=len(rows), dt=round(dt, 1), n_pass=n_pass,
            clients=len(batch), best=(round(best[0], 4) if best else None),
            stop={k: v for k, v in collections.Counter(
                r.get('stopped_at') or 'PASS' for r in rows).items()}))
        _log(f'批{self.n_batch + 1:<4d} {len(rows):5d} 条  {dt:5.1f}s  '
             f'过 {n_pass:<5d} 队列 {self.q.qsize() + len(self.pending)}')

    def _stats(self) -> Dict:
        """面板的唯一数据源。**只给数,不排版** —— 版面全在 dash.py。"""
        cur = self.cur
        funnel, best = collections.Counter(self.funnel), self.best
        if cur:                                  # 在飞批次的**已判部分**并进来:漏斗是实时的,不等整批结束
            funnel.update(dict(cur['funnel']))   # dict():worker 线程正在写它,先原子拷贝再迭代
            cb = cur['best']
            if cb and (best is None or cb['sharpe'] > best['sharpe']):
                best = cb
            cur = dict({k: v for k, v in cur.items() if k not in ('funnel', 'best')},
                       elapsed=time.time() - cur['t0'],
                       stage_elapsed=time.time() - cur.get('stage_t0', cur['t0']))
        ref=self._combo._ref.get('is') if self._combo is not None else None
        portfolio={k:ref[k] for k in ('n','sharpe','wsharpe','cagr','maxdd')} if ref else None
        return dict(
            ok=True, pid=os.getpid(), port=self.port, t=time.time(),
            phase=self.phase, warm=self.warm, log=list(_LINES)[-8:],
            uptime=time.time() - self.t0, busy=self.t_busy,
            served=self.served, exprs=self.n_expr, batches=self.n_batch,
            queue=self.q.qsize() + len(self.pending), clients=self.clients,
            vram=_vram(), device=(_device() if self.phase == 'serve' else '…'),
            funnel=dict(funnel), lanes=dict(self.lanes), best=best, events=list(self.events),
            cur=cur,portfolio=portfolio)

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

    def _judged(self, rows: Dict[str, Dict], row: Dict) -> None:
        """一条候选判完:记进本批结果,同时并入在飞批次的漏斗 / 峰值 ⇒ 面板随批内进度实时长
        (批末由 `_log_job` 并入累计)。"""
        rows[row['expr']] = row
        cur = self.cur
        cur['funnel'][row.get('stopped_at') or 'PASS'] += 1
        sh = row.get('sharpe')
        if sh is not None and (cur['best'] is None or sh > cur['best']['sharpe']):
            cur['best'] = dict(sharpe=sh, expr=row['expr'])

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
                self._tick('GPU f64 组合批量',lo,len(uniq))
                batch = engine.simulate_many(uniq[lo:lo+engine.CUDA.candidate_batch],window)
                for r,g in zip(batch,gates.run_batch(batch,ref)):
                    row = gates.rowify(r,g)
                    row['backend'] = 'cuda-f64'
                    self._judged(rows,row)
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
            self._judged(rows,row)
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
            import cupy as cp
            _DEVICE = f'{cp.cuda.runtime.getDeviceProperties(0)["name"].decode()} · combo f64'
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
_LINES: 'collections.deque[str]' = collections.deque(maxlen=200)


def _log(msg: str) -> None:
    """全量落 LOGFILE + 进内存环,由**同一个窗口**的面板显示。

    绝不写 stderr:那个窗口正被 rich.Live 占着原地重绘,插一行原始文本就会把画面撕开。"""
    global _LOG_FH
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    _LINES.append(line)
    if _LOG_FH is None:
        _LOG_FH = open(LOGFILE, 'a', encoding='utf-8', buffering=1)
    _LOG_FH.write(line + '\n')


def main() -> None:
    argv = sys.argv[1:]
    cmd = argv[0] if argv else 'serve'
    if cmd == 'serve':
        try:
            port = next((int(x) for x in argv[1:] if x.isdigit()), 0)
            Server(port).run()   # 只认纯数字为端口,其余是开关(如 --panel)
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
        sys.exit(f'未知命令 {cmd!r}(serve | dash | status | stop)')


if __name__ == '__main__':
    main()
