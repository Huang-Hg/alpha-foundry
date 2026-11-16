"""Owned TCP/SSH transports, remote sessions and explicit deployment attachment."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import shlex
import time
from typing import TYPE_CHECKING
import uuid

import numpy as np

from foundry.config import ROOT
from foundry.rpc import schema
from foundry.rpc.codec import encode_bundle, from_json, json_value
from foundry.study.store import publish
from foundry.types import (
    Array, ArrayBatch, Candidate, CandidateGrid, CommitReport, ContextConflict, DataGrid, Diagnostics, EvaluationReport,
    FamilyResult, FamilySeed, PortfolioReport, RegisteredBatch, ResearchContext,
    ReweightResult, WeightCommit, WeightReport,
    ProgramPlan, ProgramSimulation, StrategyProgram,
)

if TYPE_CHECKING:
    import paramiko
    import socket

REMOTE_ROOT = '/root/quant'
PY = '/root/miniconda3/bin/python'
HOSTS = ROOT / 'mcp-ssh' / 'hosts.json'
DEFAULT_HOST = 'quant-server'
BOOT_BUDGET = 240.
SOURCES = ('*.py', '*.cu', '*.cuh', '*.cpp', '*.hpp', '*.h')
_LOGIN = re.compile(r'^\s*ssh\s+(?:-p\s*(\d+)\s+)?([^@\s]+)@(\S+?)(?:\s+-p\s*(\d+))?\s*$')


@dataclass(frozen=True)
class Endpoint:
    port: int
    address: str = schema.HOST
    host: str | None = None
    pid: int | None = None
    started: float | None = None

    def __post_init__(self) -> None:
        if type(self.port) is not int or not 1 <= self.port <= 65535:
            raise ValueError('endpoint port must be an integer in [1,65535]')
        if not self.address or self.host == '':
            raise ValueError('endpoint addresses must be nonempty')


class SSHConnection:
    """One explicitly owned SSH client; no process-global connection cache."""

    def __init__(self, name: str) -> None:
        import paramiko

        host = json.loads(HOSTS.read_text(encoding='utf-8'))[name]
        login = _LOGIN.fullmatch(host['login'])
        if login is None:
            raise ValueError(f'hosts.json[{name}].login must be an ssh command')
        with ExitStack() as cleanup:
            client = paramiko.SSHClient()
            cleanup.callback(client.close)
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(login.group(3), port=int(login.group(1) or login.group(4) or 22),
                username=login.group(2), pkey=paramiko.PKey.from_path(host['key']),
                look_for_keys=False, allow_agent=False, timeout=15.)
            client.get_transport().set_keepalive(30)
            self.client = client
            cleanup.pop_all()

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> SSHConnection:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def request(self, endpoint: Endpoint, message: dict, timeout: float, payload: bytes) -> dict:
        channel = self.client.get_transport().open_channel('direct-tcpip',
            (schema.HOST, endpoint.port), (schema.HOST, 0), timeout=15.)
        try:
            channel.settimeout(timeout)
            return exchange(channel, message, payload)
        finally:
            channel.close()

    def command(self, command: str) -> str:
        stdin, stdout, stderr = self.client.exec_command(command, timeout=60.)
        try:
            output, error = stdout.read().decode('utf-8'), stderr.read().decode('utf-8')
            if stdout.channel.recv_exit_status() != 0:
                raise RuntimeError(f'remote command failed: {command}\n{error}')
            return output
        finally:
            stdin.close()
            stdout.close()
            stderr.close()
            stdout.channel.close()


class Transport:
    """A remote backend owns this transport until its explicit close boundary."""

    def __init__(self, endpoint: Mapping) -> None:
        self.endpoint = Endpoint(**endpoint)
        self.ssh: SSHConnection | None = None
        self.closed = False

    def request(self, message: dict, timeout: float = 900., payload: bytes = b'') -> dict:
        if self.closed:
            raise RuntimeError('transport is closed')
        message = json_value(message)
        schema.parse(message)
        expected = message['params']['bytes'] if message['method'] == 'research.register_batch' else 0
        if len(payload) != expected:
            raise ValueError('binary payload length differs from the request')
        if self.endpoint.host is not None:
            if self.ssh is None:
                self.ssh = SSHConnection(self.endpoint.host)
            return self.ssh.request(self.endpoint, message, timeout, payload)
        import socket

        with socket.create_connection((self.endpoint.address, self.endpoint.port), timeout=10.) as connection:
            connection.settimeout(timeout)
            return exchange(connection, message, payload)

    def call(self, method: str, parameters: dict, timeout: float = 900., payload: bytes = b''):
        message = dict(jsonrpc='2.0', id=uuid.uuid4().hex, method=method, params=parameters)
        response = self.request(message, timeout, payload)
        value = response_value(response)
        schema.validate_response(method, value)
        return value

    def close(self) -> None:
        if self.ssh is not None:
            self.ssh.close()
            self.ssh = None
        self.closed = True

    def __enter__(self) -> Transport:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def exchange(connection: socket.socket | paramiko.Channel, message: dict, payload: bytes) -> dict:
    encoded = json.dumps(message, ensure_ascii=False, allow_nan=False).encode() + b'\n'
    if len(encoded) > schema.MAX_REQUEST_BYTES:
        raise ValueError('request exceeds protocol byte limit')
    connection.sendall(encoded)
    connection.sendall(payload)
    with connection.makefile('rb') as reader:
        line = reader.readline(schema.MAX_RESPONSE_BYTES + 1)
    if not line.endswith(b'\n') or len(line) > schema.MAX_RESPONSE_BYTES:
        raise ValueError('response exceeds its newline-delimited protocol limit')
    response = json.loads(line)
    if not isinstance(response, dict) or response.get('jsonrpc') != '2.0':
        raise ValueError('response identity differs from its request')
    if ('result' in response) == ('error' in response):
        raise ValueError('response must contain exactly one result or error')
    if response.get('id') != message['id']:
        error = response.get('error')
        if response.get('id') is not None or not isinstance(error, dict) or error.get('code') != -32001:
            raise ValueError('response identity differs from its request')
    return response


def response_value(response: dict):
    if 'error' not in response:
        return response['result']
    error = response['error']
    code, message = error['code'], error['message']
    if code == -32009:
        raise ContextConflict(message)
    if code in (-32602, -32700):
        if error['data']['type'] == 'TypeError':
            raise TypeError(message)
        raise ValueError(message)
    if code == -32002:
        raise TimeoutError(message)
    if error['data']['type'] in ('OSError', 'FileNotFoundError', 'PermissionError'):
        raise OSError(message)
    raise RuntimeError(message)


def info() -> dict:
    record = json.loads(schema.PORTFILE.read_text(encoding='utf-8'))
    Endpoint(**record)
    return record


def invoke(message: dict, timeout: float = 900., *, endpoint: Mapping | None = None, payload: bytes = b'') -> dict:
    with Transport(info() if endpoint is None else endpoint) as transport:
        return transport.request(message, timeout, payload)


def call(method: str, params: dict, timeout: float = 900., *, endpoint: Mapping | None = None, payload: bytes = b''):
    with Transport(info() if endpoint is None else endpoint) as transport:
        return transport.call(method, params, timeout, payload)


def stop():
    return call('system.stop', {}, 10.)


class RemoteBackend:
    def __init__(self, address: tuple[str, int] | None = None) -> None:
        self.endpoint = info() if address is None else dict(address=address[0], port=address[1])
        self.transport = Transport(self.endpoint)

    def _call(self, method: str, parameters: dict, kind, payload: bytes = b''):
        return from_json(kind, self.transport.call(method, parameters, payload=payload))

    def context(self) -> ResearchContext:
        self.description = self.describe()
        return from_json(ResearchContext, self.description['context'])

    def refresh(self) -> ResearchContext:
        context = self._call('research.refresh', {}, ResearchContext)
        self.description = self.describe()
        return context

    def grid(self) -> DataGrid:
        return from_json(DataGrid, self.description['axes'])

    def describe(self) -> dict:
        description = self.transport.call('research.describe', {})
        if description.get('protocol') != schema.PROTOCOL:
            raise ValueError('server protocol differs; update the client and server together')
        return description

    def evaluate(self, candidates: tuple[Candidate, ...], context: str) -> EvaluationReport:
        return self._call('research.evaluate', dict(context=context, candidates=[asdict(c) for c in candidates]), EvaluationReport)

    def plan_program(self, program: StrategyProgram, context: str) -> ProgramPlan:
        raise ValueError('Program instances require an in-process session; remote programs require explicit versioned artifact deployment')

    def simulate(self, program: StrategyProgram, context: str) -> ProgramSimulation:
        raise ValueError('Program instances require an in-process session; remote programs require explicit versioned artifact deployment')

    def resident(self):
        raise ValueError('resident views require an in-process session')

    def evaluate_device(self, candidates: tuple[Candidate, ...], context: str):
        raise ValueError('resident device results require an in-process session')

    def commit(self, candidates: tuple[Candidate, ...], context: str) -> CommitReport:
        return self._call('research.commit', dict(context=context, candidates=[asdict(c) for c in candidates]), CommitReport)

    def register(self, batch: ArrayBatch, context: str) -> RegisteredBatch:
        payload = encode_bundle(batch)
        return self._call('research.register_batch', dict(context=context, bytes=len(payload)), RegisteredBatch, payload)

    def retain(self, candidates: tuple[Candidate, ...], context: str) -> RegisteredBatch:
        return self._call('research.retain', dict(context=context, candidates=[asdict(c) for c in candidates]), RegisteredBatch)

    def release(self, lease: str) -> None:
        self._call('research.release_batch', dict(lease=lease), type(None))

    def diagnose(self, context: str) -> Diagnostics:
        return self._call('research.diagnose', dict(context=context), Diagnostics)

    def judge_family(self, families: Sequence[FamilySeed | CandidateGrid], context: str) -> FamilyResult:
        return self._call('research.judge_family', dict(context=context, families=[asdict(f) for f in families]), FamilyResult)

    def portfolio(self, context: str) -> PortfolioReport:
        return self._call('research.portfolio', dict(context=context), PortfolioReport)

    def trial_weights(self, weights: Array, context: str) -> WeightReport:
        return self._call('research.trial_weights', dict(context=context, weights=np.asarray(weights).tolist()), WeightReport)

    def commit_weights(self, weights: tuple[float, ...], context: str) -> WeightCommit:
        return self._call('research.commit_weights', dict(context=context, weights=list(weights)), WeightCommit)

    def reweight(self, context: str, iterations: int, count: int, seed: int) -> ReweightResult:
        return self._call('research.reweight', dict(context=context, iterations=iterations, count=count, seed=seed), ReweightResult)

    def close(self) -> None:
        self.transport.close()


def parity_files() -> list[str]:
    files = sorted({path for pattern in SOURCES for path in (ROOT / 'foundry').rglob(pattern)})
    return [path.relative_to(ROOT).as_posix() for path in files]


def parity(ssh: SSHConnection) -> list[str]:
    files = parity_files()
    command = ' '.join(shlex.quote(name) for name in files)
    far = dict(reversed(line.split(None, 1)) for line in
               ssh.command(f'cd {shlex.quote(REMOTE_ROOT)} && md5sum {command} 2>/dev/null; true').splitlines())
    return [name for name in files if far.get(name) != hashlib.md5((ROOT / name).read_bytes()).hexdigest()]


def _alive(ssh: SSHConnection) -> dict | None:
    import paramiko

    with ssh.client.open_sftp() as sftp:
        try:
            with sftp.open(f'{REMOTE_ROOT}/{schema.PORTFILE.relative_to(ROOT).as_posix()}') as stream:
                endpoint = json.loads(stream.read().decode('utf-8'))
        except FileNotFoundError:
            return None
    try:
        response = ssh.request(Endpoint(**endpoint),
            dict(jsonrpc='2.0', id='status', method='system.status', params={}), 10., b'')
        response_value(response)
    except paramiko.ChannelException:
        return None
    return endpoint


def attach(name: str = DEFAULT_HOST, quiet: bool = False) -> dict:
    with SSHConnection(name) as ssh:
        different = parity(ssh)
        if different:
            raise ValueError('cloud code differs from this workspace; deploy and rebuild first:\n  ' + '\n  '.join(different))
        endpoint = _alive(ssh)
        if endpoint is None:
            if not ssh.command('pgrep -f "[f]oundry serve" || true').strip():
                if not quiet:
                    print(f'[remote] Starting {name} service', flush=True)
                endpoint_path = f'{REMOTE_ROOT}/{schema.PORTFILE.relative_to(ROOT).as_posix()}'
                ssh.command(f'rm -f {shlex.quote(endpoint_path)}; cd {shlex.quote(REMOTE_ROOT)}; mkdir -p var/service; '
                    f'setsid nohup {shlex.quote(PY)} -X utf8 -m foundry serve > var/service/out.log 2>&1 < /dev/null &')
            started = time.monotonic()
            while endpoint is None:
                if time.monotonic() - started >= BOOT_BUDGET:
                    raise TimeoutError(f'cloud service not ready within {BOOT_BUDGET:.0f}s')
                time.sleep(1.)
                endpoint = _alive(ssh)
        record = dict(host=name, port=endpoint['port'], pid=endpoint['pid'], started=endpoint['started'])
        schema.PORTFILE.parent.mkdir(parents=True, exist_ok=True)
        publish(schema.PORTFILE, json.dumps(record, ensure_ascii=False).encode())
        if not quiet:
            print(f'[remote] Attached to {name} on port {endpoint["port"]}', flush=True)
        return record


def mirror(host: str, source: str, target: Path, push_seconds: float) -> None:
    """Own one SSH watcher until its terminal snapshot or an explicit interruption."""
    if push_seconds <= 0:
        raise ValueError('push_seconds must be positive')
    watcher = """
import json, os, sys, time
path, gap = sys.argv[1], float(sys.argv[2])
seen, sent = None, -gap
while True:
    stamp = os.stat(path).st_mtime_ns if os.path.exists(path) else None
    if stamp is not None and stamp != seen:
        seen = stamp
        with open(path, encoding='utf-8') as stream:
            state = json.load(stream)
        ended = state.get('phase') in ('complete', 'failed', 'stopped', 'interrupted')
        if ended or time.monotonic() - sent >= gap:
            sent = time.monotonic()
            print(json.dumps(state, ensure_ascii=True, separators=(',', ':')), flush=True)
            if ended:
                break
        else:
            seen = None
    time.sleep(.25)
"""
    with SSHConnection(host) as ssh:
        stdin, stdout, stderr = ssh.client.exec_command(
            f'{shlex.quote(PY)} -u -c {shlex.quote(watcher)} {shlex.quote(source)} {push_seconds}')
        try:
            for line in stdout:
                state = json.loads(line)
                publish(target, json.dumps(state, ensure_ascii=False).encode())
                print(time.strftime('%H:%M:%S'), state.get('phase'), state.get('epoch'), flush=True)
            error = stderr.read().decode()
            if stdout.channel.recv_exit_status() != 0:
                raise RuntimeError(f'remote monitor failed: {error}')
        finally:
            stdin.close()
            stdout.close()
            stderr.close()
            stdout.channel.close()
