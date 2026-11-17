"""Owned TCP transports and remote sessions for explicitly selected services."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import json
from typing import TYPE_CHECKING
import uuid

import numpy as np

from foundry.rpc import schema
from foundry.rpc.codec import encode_bundle, from_json, json_value
from foundry.study.store import publish
from foundry.types import (
    Array, ArrayBatch, Candidate, CandidateGrid, CommitReport, ContextConflict, DataGrid, Diagnostics, EvaluationReport,
    FamilyResult, FamilySeed, PortfolioReport, RegisteredBatch, ResearchContext,
    ReweightResult, WeightCommit, WeightReport,
    ProgramPlan, ProgramSimulation, StrategyProgram, ProgramReference, ProgramCheckpoint,
)

if TYPE_CHECKING:
    import socket


@dataclass(frozen=True)
class Endpoint:
    port: int
    address: str = schema.HOST
    pid: int | None = None
    started: float | None = None

    def __post_init__(self) -> None:
        if type(self.port) is not int or not 1 <= self.port <= 65535:
            raise ValueError('endpoint port must be an integer in [1,65535]')
        if not isinstance(self.address, str) or not self.address.strip():
            raise ValueError('endpoint addresses must be nonempty')


class Transport:
    """A remote backend owns this transport until its explicit close boundary."""

    def __init__(self, endpoint: Mapping) -> None:
        self.endpoint = Endpoint(**endpoint)
        self.closed = False

    def request(self, message: dict, timeout: float = 900., payload: bytes = b'') -> dict:
        if self.closed:
            raise RuntimeError('transport is closed')
        message = json_value(message)
        schema.parse(message)
        expected = message['params']['bytes'] if message['method'] == 'research.register_batch' else 0
        if len(payload) != expected:
            raise ValueError('binary payload length differs from the request')
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
        self.closed = True

    def __enter__(self) -> Transport:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def exchange(connection: socket.socket, message: dict, payload: bytes) -> dict:
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

    def plan_program(self, program: StrategyProgram | ProgramReference, context: str) -> ProgramPlan:
        if not isinstance(program, ProgramReference):
            raise TypeError('Remote programs require an explicit deployed ProgramReference')
        return self._call('research.plan_program', dict(context=context, program=asdict(program)), ProgramPlan)

    def simulate(self, program: StrategyProgram | ProgramReference, context: str, *, until: int | None = None,
                 checkpoint: ProgramCheckpoint | None = None) -> ProgramSimulation:
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

    def register_program(self, program: StrategyProgram | ProgramReference, context: str) -> RegisteredBatch:
        if not isinstance(program, ProgramReference):
            raise TypeError('Remote registration requires an explicit deployed ProgramReference')
        return self._call('research.register_program', dict(context=context, program=asdict(program)), RegisteredBatch)

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


def attach(address: str, port: int) -> dict:
    """Select an existing TCP service; SSH tunnels are managed outside Foundry."""
    endpoint = Endpoint(port=port, address=address)
    record = asdict(endpoint)
    with Transport(record) as transport:
        transport.call('system.status', {}, timeout=10.)
    schema.PORTFILE.parent.mkdir(parents=True, exist_ok=True)
    publish(schema.PORTFILE, json.dumps(record, ensure_ascii=False).encode('utf-8'))
    return record
