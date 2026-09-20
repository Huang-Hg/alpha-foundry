"""云端回测服务的接入层 —— 本地客户端经 SSH 通道直连云端 daemon,**协议一字不改**。

    run.py --daemon attach [host]     把本工作区接到云端服务(没在跑就拉起),默认 quant-server
    run.py "表达式"                    照常用:回测在云端 GPU 上算,实时面板照样画在本终端
    run.py --daemon dash              只看面板
    run.py --daemon detach            断开(云端服务留着);stop = 连云端服务一起停

通信方式 = paramiko `direct-tcpip` 通道(等价 `ssh -L`,但在进程内,没有常驻隧道进程要管):
  · 云端 daemon 照旧只监听 127.0.0.1 —— AutoDL 只放行 ssh 口,也不该把无鉴权的 JSON 口放上公网
  · 鉴权 / 加密复用 ssh-bridge 那把 key;登录信息单源 = `mcp-ssh/hosts.json`(控制台登录指令原样)
  · 每个请求开一条通道:发一行 JSON、收一行 JSON,与本地 TCP 同形 ⇒ daemon 无感
  · 一个进程一条 SSH 连接;面板轮询(op=stats)与作业(op=eval)是同一连接上的不同通道,
    所以作业在飞的时候面板照样查得动(daemon 侧 stats 由 handler 线程直接应答,不排队)

"接到哪儿"是**状态**不是配置:attach 把 `{host, port}` 写进 `.q25d.json`(本来就是"服务在哪"
的那个文件),此后 client / dash / 各 agent 的 run.py 全部自动走云端,不需要任何开关。

attach 先做**代码对账**(md5,本地 ↔ 云端):云端跑的是旧代码时读数静默不一致,比连不上糟得多。
不一致直接拒绝并点名文件 —— 部署走 tar + scp_upload(本地是唯一编辑源),这里不代劳。
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import paramiko

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
HOSTS = REPO / 'mcp-ssh' / 'hosts.json'
PORTFILE = HERE / '.q25d.json'

ROOT = '/root/quant'                        # AutoDL 部署约定:代码根在系统盘
PY = '/root/miniconda3/bin/python'          # sshd exec 是非交互 shell,python 必须全路径
FUNNEL = f'{ROOT}/tmp/q25_funnel'
DEFAULT_HOST = 'quant-server'
BOOT_BUDGET = 240.0                         # 秒:云端预热(装载面板 + CUDA context + 上传显存)上限

# 对账清单：CUDA 生产路径的源码和配置。CPU 黄金参考不参与服务。
PARITY = ['tmp/q25_funnel/daemon.py', 'tmp/q25_funnel/engine.py', 'tmp/q25_funnel/gates.py',
          'tmp/q25_funnel/combo_gpu.py',
          'tmp/q25_funnel/alpha_cuda.py', 'backtest/ops_cuda/q25.py', 'backtest/ops_cuda/q25_native.cu',
          'tmp/q25_funnel/gpu.py', 'tmp/q25_funnel/fusion.py', 'tmp/q25_funnel/fastexpr.py', 'tmp/q25_funnel/combolib.py',
          'backtest/ops_cuda/__init__.py', 'backtest/ops_cuda/kernels.cu',
          'evaluation/ast.py', 'evaluation/grammar.py', 'markets/profile.py', 'markets/vocabulary.py',
          'config/config.ini', 'config/config.py']

_LOGIN = re.compile(r'^\s*ssh\s+(?:-p\s*(\d+)\s+)?([^@\s]+)@(\S+?)(?:\s+-p\s*(\d+))?\s*$')
_SSH: Dict[str, paramiko.SSHClient] = {}
_LOCK = threading.Lock()


def connect(name: str) -> paramiko.SSHClient:
    with _LOCK:
        c = _SSH.get(name)
        if c is not None and c.get_transport() is not None and c.get_transport().is_active():
            return c
        h = json.loads(HOSTS.read_text(encoding='utf-8'))[name]
        m = _LOGIN.match(h['login'])
        assert m, f'hosts.json[{name}].login 不是 ssh 登录指令:{h["login"]!r}'
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        c.connect(m.group(3), port=int(m.group(1) or m.group(4) or 22), username=m.group(2),
                  pkey=paramiko.PKey.from_path(h['key']), look_for_keys=False, allow_agent=False,
                  timeout=15.0)
        c.get_transport().set_keepalive(30)         # 长作业期间通道空转,别被 NAT 掐掉
        _SSH[name] = c
        return c


def request(d: Dict, req: Dict, timeout: float) -> Optional[Dict]:
    """与 client.request 的本地 TCP 分支同形:一行 JSON 去、一行 JSON 回。"""
    ch = connect(d['host']).get_transport().open_channel(
        'direct-tcpip', ('127.0.0.1', d['port']), ('127.0.0.1', 0), timeout=15.0)
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
    return json.loads(buf.decode('utf-8')) if buf.strip() else None


def _sh(ssh: paramiko.SSHClient, cmd: str) -> str:
    _, out, err = ssh.exec_command(cmd, timeout=60.0)
    o, e = out.read().decode('utf-8'), err.read().decode('utf-8')
    assert out.channel.recv_exit_status() == 0, f'远端命令失败:{cmd}\n{e}'
    return o


def parity(ssh: paramiko.SSHClient) -> List[str]:
    """→ 本地与云端 md5 不一致(或云端缺失)的文件。"""
    far = dict(reversed(l.split(None, 1)) for l in _sh(
        ssh, f'cd {ROOT} && md5sum {" ".join(PARITY)} 2>/dev/null; true').splitlines())
    return [f for f in PARITY
            if far.get(f) != hashlib.md5((REPO / f).read_bytes()).hexdigest()]


def _alive(name: str, ssh: paramiko.SSHClient) -> Optional[Dict]:
    """云端服务的 `{port, pid, …}`;没在跑 / 端口文件是陈的 → None。"""
    sftp = ssh.open_sftp()
    try:
        with sftp.open(f'{FUNNEL}/.q25d.json') as f:
            d = json.loads(f.read().decode('utf-8'))
    except FileNotFoundError:
        return None
    finally:
        sftp.close()
    try:
        ok = request(dict(host=name, port=d['port']), dict(op='ping'), timeout=10.0)
    except paramiko.ChannelException:              # 端口没人听 = 上次的服务已死(实例重启过)
        ok = None
    return d if ok else None


def attach(name: str = DEFAULT_HOST, quiet: bool = False) -> Dict:
    say = (lambda *a: None) if quiet else (lambda *a: print(*a, flush=True))
    ssh = connect(name)
    diff = parity(ssh)
    assert not diff, ('云端代码与本地不一致,先部署再 attach(tar + scp_upload;改了 C 源须就地重编):\n  '
                      + '\n  '.join(diff))
    push_combo(ssh)                                 # 在跑的服务按 mtime 重载组合库,不必重启
    d = _alive(name, ssh)
    if d is None:
        if not _sh(ssh, 'pgrep -f "[d]aemon.py serve" || true').strip():
            say(f'[remote] {name} 上服务未运行,正在拉起(装载面板 + CUDA context,约 20-60s)…')
            # `cd X && cmd &` 会让 bash 留一个持有本通道 stdout 的子 shell 等 cmd ⇒ read() 永远等不到 EOF
            # (2026-09-19 实测卡满 60s)。用 `;`:被后台化的只有最后那条简单命令,重定向直接落在它身上。
            _sh(ssh, f'rm -f {FUNNEL}/.q25d.json; cd {ROOT}; setsid nohup {PY} -X utf8 '
                     f'tmp/q25_funnel/daemon.py serve > {FUNNEL}/.q25d.out 2>&1 < /dev/null &')
        t0 = time.time()
        while d is None:
            assert time.time() - t0 < BOOT_BUDGET, \
                f'{BOOT_BUDGET:.0f}s 内云端服务未就绪,看 {name}:{FUNNEL}/.q25d.log 与 .q25d.out'
            time.sleep(1.0)
            d = _alive(name, ssh)
        say(f'[remote] 就绪({time.time() - t0:.1f}s)')
    PORTFILE.write_text(json.dumps(dict(host=name, port=d['port'], pid=d['pid'],
                                        started=d['started'], python=d['python']),
                                   ensure_ascii=False), encoding='utf-8')
    say(f'[remote] 已接到 {name}(pid {d["pid"]} · 云端 :{d["port"]})。此后 run.py / dash 全走云端。')
    return d


def push_combo(ssh: paramiko.SSHClient) -> None:
    """组合库本地为准(门以它为参照,两边不同则判定不同):上传库文件 + Python 成员的源文件;
    本地没有库(组合为空)⇒ 删掉云端那份。"""
    from combolib import STORE
    sftp = ssh.open_sftp()
    if STORE.exists():
        sftp.put(str(STORE), f'{FUNNEL}/{STORE.name}')
        for m in json.loads(STORE.read_text(encoding='utf-8'))['members']:
            if m['lane'] == 'python':
                rel = m['expr'].rsplit(':', 1)[0]
                _sh(ssh, f'mkdir -p {FUNNEL}/{Path(rel).parent.as_posix()}')
                sftp.put(str(HERE / rel), f'{FUNNEL}/{rel}')
    else:
        _sh(ssh, f'rm -f {FUNNEL}/{STORE.name}')
    sftp.close()


def pull_combo(name: str) -> None:
    """`--accept` 改的是云端那份组合库 ⇒ 拉回本地,保持本地为唯一真本。"""
    from combolib import STORE
    sftp = connect(name).open_sftp()
    sftp.get(f'{FUNNEL}/{STORE.name}', str(STORE))
    for m in json.loads(STORE.read_text(encoding='utf-8'))['members']:
        if m['lane'] == 'python':
            rel = m['expr'].rsplit(':',1)[0]
            target = (HERE / rel).resolve()
            assert target.is_relative_to(HERE.resolve()), 'Python member path leaves funnel directory'
            target.parent.mkdir(parents=True,exist_ok=True)
            sftp.get(f'{FUNNEL}/{rel}',str(target))
    sftp.close()
