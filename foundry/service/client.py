"""Client of the evaluation service: starts a local service on demand, or talks to an attached cloud service over
SSH (remote.py). An attached cloud service that fails is an error - evaluation never silently falls back to the
local card (thinking a job runs in the cloud while it crawls locally is worse than failing).
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from typing import Dict, List, Optional

from foundry.config import ROOT
from foundry.portfolio.library import PROTOCOL
from foundry.service.protocol import HOST, LOCKFILE, LOGFILE, PORTFILE, STATE

BOOT_BUDGET = 180.0       # seconds to wait for a starting service (panel load + upload)
LOCK_STALE = 300.0        # a start lock this old belongs to a crashed start
DETACHED = 0x00000008     # Windows DETACHED_PROCESS


def info() -> Optional[Dict]:
    return json.loads(PORTFILE.read_text(encoding='utf-8')) if PORTFILE.exists() else None


def request(req: Dict, timeout: float = 900.0) -> Optional[Dict]:
    """One request; None when no service answers (the caller decides whether to start one)."""
    d = info()
    if not d:
        return None
    if req.get('op') in ('eval', 'family'):
        status = request(dict(op='ping'), timeout=10.0)
        if status is None:
            return None
        if status.get('protocol') != PROTOCOL:
            raise RuntimeError('the service runs another admission protocol: deploy this code and restart it')
        req = dict(req, protocol=PROTOCOL)
    if d.get('host'):
        from foundry.service import remote
        res = remote.request(d, req, timeout)
        res['via'] = d['host']
        return res
    try:  # network boundary: a refused or dropped connection means no service
        with socket.create_connection((HOST, d['port']), timeout=10.0) as s:
            s.settimeout(timeout)
            s.sendall(json.dumps(req, ensure_ascii=False).encode('utf-8') + b'\n')
            buf = b''
            while not buf.endswith(b'\n'):
                b = s.recv(1 << 20)
                if not b:
                    break
                buf += b
    except OSError:
        return None
    return json.loads(buf.decode('utf-8')) if buf.strip() else None


def ensure(quiet: bool = False) -> None:
    """Make sure a service is reachable: re-attach a cloud one, or start a local one."""
    if request(dict(op='ping'), timeout=10.0):
        return
    d = info()
    if d and d.get('host'):
        from foundry.service import remote
        remote.attach(d['host'], quiet)
        return
    PORTFILE.unlink(missing_ok=True)              # stale port file
    if _claim_lock():
        _spawn(quiet)
    _wait_ready(quiet)


def _result(res: Optional[Dict]) -> Dict:
    if not res or not res.get('ok'):
        where = (info() or {}).get('host') or 'local service'
        raise RuntimeError(f'{where}: job failed or timed out: {(res or {}).get("error", "no response")}')
    return res


def evaluate(exprs: List[str], accept: bool = False, quiet: bool = False, source: Optional[str] = None) -> List[Dict]:
    """source = the submitted file (the dashboard groups rounds by its directory)."""
    ensure(quiet)
    res = _result(request(dict(op='eval', exprs=list(exprs), accept=bool(accept), source=source)))
    if accept and res.get('via'):
        from foundry.service import remote
        remote.pull_library(res['via'])
    if not quiet and res.get('coalesced', 1) > 1:
        print(f'[service] 与另 {res["coalesced"] - 1} 个客户端的请求汇流为单批 {res.get("batch_exprs")} 条')
    return res['rows']


def evaluate_family(seeds: List[Dict], source: str, quiet: bool = False) -> Dict:
    """seeds [{expr, hypothesis}] -> family.expand() result; a round is thousands of variants and may queue behind
    other agents' rounds, hence the long timeout."""
    ensure(quiet)
    return _result(request(dict(op='family', seeds=list(seeds), source=source), timeout=3600.0))


def stop() -> Optional[Dict]:
    return request(dict(op='stop'), timeout=10.0)


# ---------------------------------------------------------------- start-up race
def _claim_lock() -> bool:
    """True: this process starts the service; False: another one is starting it."""
    STATE.mkdir(parents=True, exist_ok=True)
    if LOCKFILE.exists() and time.time() - LOCKFILE.stat().st_mtime > LOCK_STALE:
        LOCKFILE.unlink(missing_ok=True)
    try:
        fd = os.open(str(LOCKFILE), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    return True


def _spawn(quiet: bool) -> None:
    """Start the service without a window, logging to tmp/service/service.log. (A Windows process started from WSL
    interop cannot open a visible console for its child; the dashboard is `python -m foundry monitor`.)"""
    if not quiet:
        print('[service] 未运行,正在拉起(装载面板 + 上传显存,首次约 20-60s)…', flush=True)
    kw: Dict = {'cwd': str(ROOT), 'stdin': subprocess.DEVNULL}
    if sys.platform == 'win32':
        kw['creationflags'] = DETACHED | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kw['start_new_session'] = True
    with open(LOGFILE, 'a', encoding='utf-8') as log:
        subprocess.Popen([sys.executable, '-X', 'utf8', '-m', 'foundry', 'serve'], stdout=log, stderr=log, **kw)


def _wait_ready(quiet: bool) -> None:
    t0 = time.time()
    while time.time() - t0 < BOOT_BUDGET:
        if request(dict(op='ping'), timeout=10.0):
            LOCKFILE.unlink(missing_ok=True)
            if not quiet:
                print(f'[service] 就绪({time.time() - t0:.1f}s)', flush=True)
            return
        time.sleep(0.5)
    LOCKFILE.unlink(missing_ok=True)
    raise RuntimeError(f'the service did not come up within {BOOT_BUDGET:.0f}s; see {LOGFILE}')
