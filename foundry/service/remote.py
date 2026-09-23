"""Cloud evaluation service: the local client reaches a cloud service through an SSH channel, same protocol.

    python -m foundry attach [host]     attach this workspace to the cloud service (start it if needed)
    python -m foundry eval ...          evaluation now runs on the cloud GPU
    python -m foundry detach            forget the cloud service (it keeps running); stop = stop it too

Transport: a paramiko direct-tcpip channel per request (like ssh -L, in process, no tunnel to manage). The cloud
service still listens on 127.0.0.1 only (the host exposes only ssh, and the JSON port has no authentication);
login details come from mcp-ssh/hosts.json. Where to talk is state, not configuration: attach writes {host, port}
into the service port file, and every client of this workspace follows it.

attach first checks code parity (md5 of every source file and the config, local vs cloud): a cloud running other
code gives silently different readings, which is worse than no connection. Deployment itself (tar + upload, rebuild
of changed native sources on the server) is not done here.
"""
from __future__ import annotations

import hashlib
import json
import re
import shlex
import threading
import time
from typing import Dict, List

import paramiko

from foundry.config import ROOT
from foundry.core.files import publish
from foundry.portfolio.library import STORE
from foundry.service.protocol import PORTFILE

REMOTE_ROOT = '/root/quant'                  # deployment convention: the code root on the system disk
PY = '/root/miniconda3/bin/python'          # sshd exec is a non-interactive shell: the full interpreter path
HOSTS = ROOT / 'mcp-ssh' / 'hosts.json'
DEFAULT_HOST = 'quant-server'
BOOT_BUDGET = 240.0
SOURCES = ('*.py', '*.cu', '*.cpp', '*.hpp', '*.h')

_LOGIN = re.compile(r'^\s*ssh\s+(?:-p\s*(\d+)\s+)?([^@\s]+)@(\S+?)(?:\s+-p\s*(\d+))?\s*$')
_SSH: Dict[str, paramiko.SSHClient] = {}
_LOCK = threading.Lock()


def parity_files() -> List[str]:
    """Everything that decides a reading: the package sources and the configuration."""
    files = sorted({p for pattern in SOURCES for p in (ROOT / 'foundry').rglob(pattern)})
    return [p.relative_to(ROOT).as_posix() for p in files] + ['config/config.ini']


def connect(name: str) -> paramiko.SSHClient:
    with _LOCK:
        c = _SSH.get(name)
        if c is not None and c.get_transport() is not None and c.get_transport().is_active():
            return c
        h = json.loads(HOSTS.read_text(encoding='utf-8'))[name]
        m = _LOGIN.match(h['login'])
        assert m, f'hosts.json[{name}].login is not an ssh command: {h["login"]!r}'
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        c.connect(m.group(3), port=int(m.group(1) or m.group(4) or 22), username=m.group(2),
                  pkey=paramiko.PKey.from_path(h['key']), look_for_keys=False, allow_agent=False, timeout=15.0)
        c.get_transport().set_keepalive(30)         # long jobs leave the channel idle; keep NAT from dropping it
        _SSH[name] = c
        return c


def request(d: Dict, req: Dict, timeout: float) -> Dict:
    """Same shape as the local TCP request: one JSON line out, one back."""
    ch = connect(d['host']).get_transport().open_channel('direct-tcpip', ('127.0.0.1', d['port']), ('127.0.0.1', 0), timeout=15.0)
    try:
        ch.settimeout(timeout)
        ch.sendall(json.dumps(req, ensure_ascii=False).encode('utf-8') + b'\n')
        buf = b''
        while not buf.endswith(b'\n'):
            b = ch.recv(1 << 20)
            if not b:
                break
            buf += b
    finally:
        ch.close()
    return json.loads(buf.decode('utf-8'))


def sh(ssh: paramiko.SSHClient, cmd: str) -> str:
    _, out, err = ssh.exec_command(cmd, timeout=60.0)
    o, e = out.read().decode('utf-8'), err.read().decode('utf-8')
    assert out.channel.recv_exit_status() == 0, f'remote command failed: {cmd}\n{e}'
    return o


def parity(ssh: paramiko.SSHClient) -> List[str]:
    """Files whose md5 differs between here and the cloud (or that the cloud lacks)."""
    files = parity_files()
    far = dict(reversed(line.split(None, 1)) for line in sh(ssh, f'cd {REMOTE_ROOT} && md5sum {" ".join(files)} 2>/dev/null; true').splitlines())
    return [f for f in files if far.get(f) != hashlib.md5((ROOT / f).read_bytes()).hexdigest()]


def _alive(name: str, ssh: paramiko.SSHClient):
    """The cloud service's port file if it answers, else None (never started, or the instance restarted)."""
    sftp = ssh.open_sftp()
    try:
        with sftp.open(f'{REMOTE_ROOT}/{PORTFILE.relative_to(ROOT).as_posix()}') as f:
            d = json.loads(f.read().decode('utf-8'))
    except FileNotFoundError:
        return None
    finally:
        sftp.close()
    try:
        request(dict(host=name, port=d['port']), dict(op='ping'), timeout=10.0)
    except paramiko.ChannelException:                # nobody listens on the port any more
        return None
    return d


def attach(name: str = DEFAULT_HOST, quiet: bool = False) -> Dict:
    say = (lambda *a: None) if quiet else (lambda *a: print(*a, flush=True))
    ssh = connect(name)
    diff = parity(ssh)
    assert not diff, 'cloud code differs from this workspace; deploy first (tar + upload, rebuild native sources):\n  ' + '\n  '.join(diff)
    push_library(ssh)                                # a running service reloads the library by mtime
    d = _alive(name, ssh)
    if d is None:
        if not sh(ssh, 'pgrep -f "[f]oundry serve" || true').strip():
            say(f'[remote] {name} 上服务未运行,正在拉起(装载面板 + CUDA context,约 20-60s)…')
            # `cd X && cmd &` keeps a subshell holding this channel's stdout, so read() never sees EOF: with `;`
            # only the last simple command is backgrounded and its redirections apply to it.
            sh(ssh, f'rm -f {PORTFILE.relative_to(ROOT).as_posix()}; cd {REMOTE_ROOT}; mkdir -p tmp/service; '
                    f'setsid nohup {PY} -X utf8 -m foundry serve > tmp/service/out.log 2>&1 < /dev/null &')
        t0 = time.time()
        while d is None:
            assert time.time() - t0 < BOOT_BUDGET, f'cloud service not ready within {BOOT_BUDGET:.0f}s; see {name}:{REMOTE_ROOT}/tmp/service/'
            time.sleep(1.0)
            d = _alive(name, ssh)
        say(f'[remote] 就绪({time.time() - t0:.1f}s)')
    PORTFILE.parent.mkdir(parents=True, exist_ok=True)
    PORTFILE.write_text(json.dumps(dict(host=name, port=d['port'], pid=d['pid'], started=d['started'], python=d['python']),
                                   ensure_ascii=False), encoding='utf-8')
    say(f'[remote] 已接到 {name}(pid {d["pid"]} · 云端 :{d["port"]})。此后评价全走云端。')
    return d


def push_library(ssh: paramiko.SSHClient) -> None:
    """The local library is the source of truth (admission is judged against it): upload it, or remove the cloud
    copy when there is none here."""
    target = f'{REMOTE_ROOT}/{STORE.relative_to(ROOT).as_posix()}'
    if STORE.exists():
        sh(ssh, f'mkdir -p {target.rsplit("/", 1)[0]}')
        sftp = ssh.open_sftp()
        sftp.put(str(STORE), target)
        sftp.close()
    else:
        sh(ssh, f'rm -f {target}')


def pull_library(name: str) -> None:
    """--accept changed the cloud copy: bring it back so the local file stays the only original."""
    sftp = connect(name).open_sftp()
    sftp.get(f'{REMOTE_ROOT}/{STORE.relative_to(ROOT).as_posix()}', str(STORE))
    sftp.close()


def mirror(host: str, source: str, target, push_seconds: float) -> None:
    """Stream a cloud status file into a local one for the dashboard: one SSH session, no polling - a tiny watcher on
    the server prints a snapshot when the file changed, at most once per push_seconds, and exits with the run."""
    watcher = '''
import json, os, sys, time
path, gap = sys.argv[1], float(sys.argv[2])
seen, sent = None, -gap
while True:
    stamp = os.stat(path).st_mtime_ns if os.path.exists(path) else None
    if stamp != seen:
        seen = stamp
        state = json.load(open(path, encoding='utf-8'))
        ended = state.get('phase') in ('complete', 'failed', 'stopped', 'interrupted')
        if ended or time.monotonic()-sent >= gap:
            sent = time.monotonic()
            print(json.dumps(state, ensure_ascii=True, separators=(',', ':')), flush=True)
            if ended:
                break
        else:
            seen = None
    time.sleep(.25)
'''
    ssh = connect(host)
    _, stdout, _ = ssh.exec_command(f'{PY} -u -c {shlex.quote(watcher)} {shlex.quote(source)} {push_seconds}', get_pty=True)
    for line in stdout:
        state = json.loads(line)
        publish(target, json.dumps(state, ensure_ascii=False).encode('utf-8'))
        print(time.strftime('%H:%M:%S'), state.get('phase'), state.get('epoch'), flush=True)
