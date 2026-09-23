"""Q25 CUDA 常驻服务客户端：自动启动、FIFO 排队、SSH 远端调用。
本机进程内入口也只支持 CUDA；远端失联不会切到本机计算。
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent
PORTFILE = HERE / '.q25d.json'
LOCKFILE = HERE / '.q25d.lock'
LOGFILE = HERE / '.q25d.log'
HOST = '127.0.0.1'

BOOT_BUDGET = 180.0       # 秒:等服务预热(装载面板 + 上传显存)的上限
LOCK_STALE = 300.0        # 秒:锁文件超过此龄视为上一次启动崩了,可抢占


def info() -> Optional[Dict]:
    if not PORTFILE.exists():
        return None
    try:
        return json.loads(PORTFILE.read_text(encoding='utf-8'))
    except Exception:
        return None


def request(req: Dict, timeout: float = 900.0) -> Optional[Dict]:
    """发一个请求。服务不可达 / 超时 ⇒ None(调用方降级),不抛。"""
    d = info()
    if not d:
        return None
    if req.get('op') in ('eval', 'python', 'family'):
        status = request(dict(op='ping'), timeout=10.0)
        if status is None:
            return None
        if status.get('protocol') != 'combo-cuda-v6':
            raise RuntimeError('回测服务仍是旧 gate 协议;请部署本地新代码并重启服务后再评估/入库')
        req = dict(req, protocol='combo-cuda-v6')
    if d.get('host'):
        import remote
        try:
            res = remote.request(d, req, timeout)
        except Exception:
            return None
        if res is not None:
            res['via'] = d['host']                 # 这份读数来自哪台机器(--accept 据此拉回组合库)
        return res
    try:
        with socket.create_connection((HOST, d['port']), timeout=10.0) as s:
            s.settimeout(timeout)
            s.sendall(json.dumps(req, ensure_ascii=False).encode('utf-8') + b'\n')
            buf = b''
            while not buf.endswith(b'\n'):
                b = s.recv(1 << 20)
                if not b:
                    break
                buf += b
        return json.loads(buf.decode('utf-8')) if buf.strip() else None
    except Exception:
        return None


def ensure(autostart: bool = True, quiet: bool = False) -> bool:
    """确保服务在跑。→ True 表示可用。"""
    if request(dict(op='ping'), timeout=10.0):
        return True
    d = info()
    if d and d.get('host'):                        # 云端:不降级,重新 attach(见模块说明)
        import remote
        remote.attach(d['host'], quiet)
        return True
    PORTFILE.unlink(missing_ok=True)               # 端口文件是陈的
    if not autostart:
        return False
    if _claim_lock():
        _spawn(quiet)
    return _wait_ready(quiet)


def _cloud_failed(res: Optional[Dict]) -> None:
    """云端已接上却拿不到结果 ⇒ 大声崩,绝不退回本机计算(模块说明:以为在云端算、实际在本机爬)。"""
    d = info()
    if d and d.get('host') and (not res or not res.get('ok')):
        raise RuntimeError(f'云端 {d["host"]} 作业失败或超时:'
                           f'{(res or {}).get("error", "无响应")}。不切换到本机计算 —— '
                           f'查 `run.py --daemon status`,必要时由主控重新 attach')


def evaluate(exprs: List[str], window: str = 'is', accept: bool = False,
             autostart: bool = True, quiet: bool = False, source: Optional[str] = None) -> Optional[List[Dict]]:
    """source = 表达式文件路径(面板按它的目录归到提交者名下)。"""
    if not ensure(autostart, quiet):
        return None
    res = request(dict(op='eval', exprs=list(exprs), window=window, accept=bool(accept), source=source))
    _cloud_failed(res)
    if not res or not res.get('ok'):
        return None
    if accept and res.get('via'):
        import remote
        remote.pull_combo(res['via'])
    if not quiet and res.get('coalesced', 1) > 1:
        print(f'[daemon] 与另 {res["coalesced"] - 1} 个客户端的请求汇流为单批 '
              f'{res.get("batch_exprs")} 条')
    return res['rows']


def evaluate_family(seeds: List[Dict], source: str, autostart: bool = True, quiet: bool = False) -> Optional[Dict]:
    """seeds [{expr, hypothesis}] -> family.expand() result, computed by the service (thousands of variants per round:
    the queue may hold other agents' rounds ahead of this one, hence the long timeout). source = the hypothesis file."""
    if not ensure(autostart, quiet):
        return None
    res = request(dict(op='family', seeds=list(seeds), source=source), timeout=3600.0)
    _cloud_failed(res)
    return res if res and res.get('ok') else None


def evaluate_python(path: Path, window: str = 'is', autostart: bool = True,
                    quiet: bool = False, accept: bool = False) -> Optional[List[Dict]]:
    if not ensure(autostart, quiet):
        return None
    res = request(dict(op='python', src=Path(path).read_text(encoding='utf-8'),
                       path=str(path), window=window, accept=accept))
    _cloud_failed(res)
    if accept and res and res.get('via') and any(r['accepted'] for r in res.get('rows',[])):
        import remote
        remote.pull_combo(res['via'])
    return res['rows'] if res and res.get('ok') else None


# ---------------------------------------------------------------- 启动竞争
def _claim_lock() -> bool:
    """抢启动权。返回 True = 由本进程去起服务;False = 别人在起,我等就行。"""
    if LOCKFILE.exists():
        try:
            if time.time() - LOCKFILE.stat().st_mtime > LOCK_STALE:
                LOCKFILE.unlink(missing_ok=True)   # 上一次启动崩了,抢占
        except OSError:
            pass
    try:
        fd = os.open(str(LOCKFILE), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    return True


DETACHED = 0x00000008        # Windows DETACHED_PROCESS


def _spawn(quiet: bool) -> None:
    """自动拉起服务端 —— **无窗口**,日志重定向进 LOGFILE。

    ⚠️ 2026-08-13 实测(`Get-Process | MainWindowHandle`):从 WSL interop 启动的 Windows
    进程再用 `CREATE_NEW_CONSOLE` 拉子进程,**不会得到可见的 conhost 窗口**(全部
    `HasWin=False`)。所以"让服务自带一个终端"在本环境里做不到,别再试。

    ⇒ 服务不画任何东西,只写状态文件;实时面板是 `python -m monitoring --source tmp/q25_funnel/.q25d_status.json`。
    """
    if not quiet:
        print('[daemon] 未运行,正在拉起(装载面板 + 上传显存,首次约 20-60s)…\n'
              '          它没有窗口 —— 实时面板:python -m monitoring --source tmp/q25_funnel/.q25d_status.json', flush=True)
    kw: Dict = {'cwd': str(HERE.parents[1]), 'stdin': subprocess.DEVNULL}
    if sys.platform == 'win32':
        kw['creationflags'] = DETACHED | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kw['start_new_session'] = True
    with open(LOGFILE, 'a', encoding='utf-8') as log:
        subprocess.Popen([sys.executable, '-X', 'utf8', str(HERE / 'daemon.py'), 'serve'],
                         stdout=log, stderr=log, **kw)


def _wait_ready(quiet: bool) -> bool:
    t0 = time.time()
    while time.time() - t0 < BOOT_BUDGET:
        if request(dict(op='ping'), timeout=10.0):
            LOCKFILE.unlink(missing_ok=True)
            if not quiet:
                print(f'[daemon] 就绪({time.time() - t0:.1f}s)', flush=True)
            return True
        time.sleep(0.5)
    LOCKFILE.unlink(missing_ok=True)
    if not quiet:
        print(f'[daemon] {BOOT_BUDGET:.0f}s 内未就绪 ⇒ 改用本机 CUDA 入口。'
              f'日志:{LOGFILE}', flush=True)
    return False
