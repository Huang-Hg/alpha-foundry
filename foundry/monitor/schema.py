"""Wire contracts derived from typed mining telemetry and the service publisher.

The monitor consumes current nested records only. Dot paths identify numeric
readings that glide between snapshots without rewriting producer payloads.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import is_dataclass
from types import UnionType
from typing import (
    Literal,
    TypedDict,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from foundry.types import DiscoveryRow, MiningStatus, SingleGate

Ended = Literal['complete', 'failed', 'stopped', 'interrupted']
ENDED = get_args(Ended)


class Source(TypedDict, total=False):
    """Local reader metadata; never part of a producer's numerical report."""
    _source_mtime: float
    _source_name: str


_RECORDS: dict[type, type] = {}


def _wire_type(hint):
    origin, args = get_origin(hint), get_args(hint)
    if is_dataclass(hint):
        return _record(hint)
    if origin in (Union, UnionType):
        return Union[tuple(_wire_type(arg) for arg in args)]
    if origin in (tuple, list, Sequence):
        return list[_wire_type(args[0])]
    if origin in (dict, Mapping):
        return dict[args[0], _wire_type(args[1])]
    return hint


def _record(model: type) -> type:
    if model not in _RECORDS:
        hints = {name: _wire_type(hint) for name, hint in get_type_hints(model).items()}
        _RECORDS[model] = TypedDict(model.__name__ + 'Record', hints)
    return _RECORDS[model]


CandidateReport = _record(DiscoveryRow)
Gate = _record(SingleGate)
SearchStatus = TypedDict('SearchStatus', {
    **get_type_hints(_record(MiningStatus)), **get_type_hints(Source),
}, total=False)


def reading(state, path):
    """Read one declared nested telemetry path; absent snapshots have no value."""
    value = state
    for part in path.split('.'):
        value = value.get(part)
        if value is None:
            return None
    return value


def market_name(state) -> str | None:
    return state.get('market') if state.get('kind') == 'service' else reading(state, 'portfolio.context.snapshot.study.market')


def candidate_name(report) -> str:
    candidate = report['candidate']
    source = candidate['source']
    if source['kind'] == 'formula':
        return source['expression']
    return candidate['id'] + ' · ' + source['kind']


def validate_status(state):
    """Reject obsolete mining layouts at the file boundary; waiting has no record."""
    if state and state.get('kind') == 'service':
        phases = get_args(get_type_hints(ServiceStatus)['phase'])
        if state.get('phase') not in phases:
            raise ValueError('service status contains an unknown lifecycle phase')
        if state['phase'] == 'failed' and not isinstance(state.get('error'), str):
            raise ValueError('failed service status requires its error message')
    elif state:
        required = set(get_type_hints(_record(MiningStatus)))
        if not required.issubset(state):
            raise ValueError('mining status does not match the current nested telemetry contract')
    return state


# ------------------------------------------------------------------ evaluation service

Op = str


class PortfolioSummary(TypedDict, total=False):
    """The portfolio reference the service admits against."""
    n: int
    sharpe: float
    cagr: float
    maxdd: float


class CurrentJob(TypedDict, total=False):
    """The job on the GPU (several coalesced evaluation requests count as one)."""
    op: Op
    id: str
    state: str
    clients: int
    t0: float                                # service clock
    source: str | None                    # submitted file; its directory names the submitter
    n: int
    # From the first progress tick on:
    stage: str
    stage_t0: float
    done: int
    total: int


class JobRecord(TypedDict, total=False):
    """A finished job; the service keeps the last 32."""
    t: float                                 # service clock
    op: Op
    id: str
    state: str
    source: str | None
    dt: float
    ok: bool
    n: int
    variants: int
    verdicts: dict[str, int]


class SubmitterTotals(TypedDict, total=False):
    """Every round one submitting directory has sent since the service started."""
    rounds: int
    formulas: int
    variants: int
    verdicts: dict[str, int]
    last: float                              # service clock


class ServiceStatus(Source, total=False):
    protocol: str
    kind: Literal['service']
    market: str
    phase: Literal['warm', 'serve', 'stopping', 'stopped', 'failed']
    error: str | None
    warm: str                                # what warm-up is doing
    pid: int
    port: int
    now: float                               # service clock when written; only differences to it are shown
    elapsed_seconds: float
    busy_seconds: float
    device: str | None                    # None while warming up
    vram: str
    queue: int
    clients: int
    served: int
    exprs: int
    batches: int
    portfolio: PortfolioSummary | None
    cur: CurrentJob | None
    jobs: list[JobRecord]
    sources: dict[str, SubmitterTotals]


Status = Union[SearchStatus, ServiceStatus]

GLIDE = ('epoch', 'accepted', 'counters.requested', 'counters.unique_evaluations',
         'counters.g1', 'counters.g2', 'counters.g3', 'counters.low_corr',
         'counters.behavior_duplicates', 'counters.weight_trials',
         'counters.probe_evaluations', 'counters.reweight_evaluations',
         'portfolio.metrics.sharpe', 'portfolio.metrics.cagr', 'used_mib', 'exprs')
RESTART = ('epoch', 'counters.unique_evaluations', 'exprs')
