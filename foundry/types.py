"""Pure value contracts; device arrays remain borrowed on their owning runtime.

Array axes are batch B, decision D, execution event E and asset S. Host
arrays use NumPy; device arrays expose the CUDA array interface without loading
a device library. Field documentation defines dtype, units and ownership.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from contextlib import AbstractContextManager
from datetime import datetime
import hashlib
import json
import re
from types import ModuleType
from typing import Callable, Generic, Literal, Mapping, Protocol, Sequence, TypeAlias, TypeVar, runtime_checkable
from enum import IntEnum
from functools import cached_property

import numpy as np
from numpy.typing import NDArray

MarketValue = TypeVar('MarketValue')
PanelValue = TypeVar('PanelValue')
SpecValue = TypeVar('SpecValue')
LibraryValue = TypeVar('LibraryValue')
ResidentValue = TypeVar('ResidentValue', covariant=True)


class DeviceArray(Protocol):
    """Borrowed device tensor; shape and dtype match the host-array contract."""
    shape: tuple[int, ...]
    ndim: int
    size: int
    dtype: np.dtype

    @property
    def __cuda_array_interface__(self) -> Mapping[str, int | str | tuple | list | None]: ...


Array: TypeAlias = NDArray[np.generic] | DeviceArray
HostArray: TypeAlias = NDArray[np.generic]
FloatArray: TypeAlias = NDArray[np.floating] | DeviceArray
IntArray: TypeAlias = NDArray[np.integer] | DeviceArray
BoolArray: TypeAlias = NDArray[np.bool_] | DeviceArray

NO_INFORMATION = -9223372036854775808
ArrayKind: TypeAlias = Literal['scores', 'targets']
ArrayUnit: TypeAlias = Literal['score', 'equity_fraction', 'quantity', 'notional']


def utc_nanoseconds(value: str) -> int:
    """Parse an explicit UTC ISO timestamp without timezone-dependent conversion."""
    if not isinstance(value, str) or not re.fullmatch(
            r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?Z', value):
        raise ValueError('Decision timestamps require ISO UTC seconds and a Z suffix')
    whole, _, fraction = value[:-1].partition('.')
    elapsed = datetime.fromisoformat(whole) - datetime(1970, 1, 1)
    result = ((elapsed.days * 86400 + elapsed.seconds) * 1000000000
              + int(fraction.ljust(9, '0') or '0'))
    if not NO_INFORMATION < result <= 9223372036854775807:
        raise ValueError('Timestamp is outside the supported int64 nanosecond range')
    return result


def utc_timestamps(values: NDArray[np.int64]) -> tuple[str, ...]:
    """Format UTC nanoseconds as canonical JSON-safe timestamps."""
    return tuple(np.datetime_as_string(np.datetime64(int(value), 'ns'), unit='ns') + 'Z'
                 for value in values)


class ParsedFormula(Protocol):
    """Read-only parsed expression metadata, independent of its evaluator."""
    @property
    def hash(self) -> int: ...
    def cost(self) -> int: ...
    def fields(self) -> list[str]: ...
    def pretty(self) -> str: ...


@dataclass(frozen=True)
class Study:
    market: str
    start: str
    asof: str


@dataclass(frozen=True)
class Allocation:
    """Gross notional/equity budget and whether signed exposures are allowed."""
    max_gross: float
    allow_short: bool


@dataclass(frozen=True)
class PrecisionPlan:
    """The requested policy and explicit capability-based execution decision."""
    requested: str
    effective: str
    reason: str
    supported: tuple[str, ...]


@dataclass(frozen=True)
class ExecutionRiskBatch:
    """Derivative risk arrays owned by the runtime, on the full input grid.

    Cashflows/counts are (B,T). Cashflows use initial NAV=1; negative funding
    means payment. Isolated-loss caps record the liquidation adjustment that
    prevents negative isolated buckets from consuming free cash. Events are
    uint8 (B,E,S), with open/funding/rebalance/intrabar stage bits 1/2/4/8.
    Remaining arrays are (B,); UTC timestamps are int64 nanoseconds, with -1
    meaning no liquidation. Max gross leverage may be infinite at nonpositive
    equity before liquidation; the JSON boundary represents nonfinite values
    as null while the device result preserves the value.
    """
    funding: FloatArray
    trading_fees: FloatArray
    liquidation_fees: FloatArray
    isolated_loss_cap: FloatArray
    liquidations: IntArray
    liquidation_events: IntArray
    first_liquidation_ns: IntArray
    bankrupt: BoolArray
    ending_equity: FloatArray
    max_gross_leverage: FloatArray
    model: str
    gross_sampling: str = 'account_event_samples'


@dataclass(frozen=True)
class SimulationResult:
    """Simulate returns full (B,T); blend returns (B,T-start). Risk is always full-grid."""
    returns: FloatArray
    statistics: FloatArray
    risk: ExecutionRiskBatch | None


@dataclass(frozen=True)
class ReturnBatch:
    """Books on the full grid, returns on the research window; risk remains full-grid."""
    books: FloatArray
    returns: FloatArray
    valid: BoolArray
    maxdd: FloatArray
    risk: ExecutionRiskBatch | None


@dataclass(frozen=True)
class ReturnMoments:
    """Per-row arithmetic Sharpe, geometric growth and recency-weighted moments.

    Arrays stay on the backend and have the input's shape excluding its time axis.
    Population variance, zero variance -> Sharpe 0; weights sum to one. Growth
    uses log1p of each net return. Nonfinite/ruined rows remain domain-invalid.
    """
    sharpe: FloatArray
    cagr: FloatArray
    weighted_sharpe: FloatArray
    recent_growth: FloatArray


ReturnMeasure = Callable[[FloatArray, FloatArray, float], ReturnMoments]


@dataclass(frozen=True)
class Backend:
    name: str
    xp: ModuleType
    impl: Mapping[str, Callable[..., FloatArray]]
    unsupported: Callable[[ParsedFormula], tuple[str, ...]]
    max_window: int
    max_pair_window: int
    to_host: Callable[[Array], HostArray]
    measure: ReturnMeasure
    close: Callable[[], None]


@dataclass(frozen=True)
class DataContract:
    schema: str
    source: str
    retrieved_at: str
    features: str
    execution_prices: str
    publication: str
    membership: str


@dataclass(frozen=True)
class MarketData:
    """Host daily arrays (T,S), float64 prices/volume; boolean membership.

    The reader owns these arrays until compilation; a compiled study exposes
    immutable source arrays and separate execution buffers.
    """
    dates: NDArray[np.datetime64]
    assets: tuple[str, ...]
    values: Mapping[str, HostArray]
    members: NDArray[np.bool_]
    contract: DataContract


@dataclass(frozen=True)
class ClockGrid:
    """Owned read-only decision UTC ns (D,), with calendar and provenance identity."""
    decision_ns: NDArray[np.int64]
    timezone: str
    calendar: str
    provenance: str

    def __post_init__(self) -> None:
        values = np.asarray(self.decision_ns)
        if (values.dtype != np.dtype('int64') or values.ndim != 1 or not values.size
                or np.any(values == NO_INFORMATION) or np.any(values[1:] <= values[:-1])):
            raise ValueError('Decision times must be nonempty, strictly increasing int64 UTC ns')
        owned = np.array(values, copy=True, order='C')
        owned.flags.writeable = False
        object.__setattr__(self, 'decision_ns', owned)

    @property
    def decision_times(self) -> tuple[str, ...]:
        return utc_timestamps(self.decision_ns)


@dataclass(frozen=True)
class Window:
    """Separate decision and measurement axes with explicit evaluation offsets."""
    clock: ClockGrid
    decision_start: int
    measurement_times: NDArray[np.datetime64]
    measurement_start: int
    periods_per_year: float

    def __post_init__(self) -> None:
        times = np.asarray(self.measurement_times)
        if (times.ndim != 1 or times.dtype.kind != 'M' or not len(times)
                or np.isnat(times).any() or np.any(times[1:] <= times[:-1])):
            raise ValueError('Measurement labels must be ordered nonempty datetime64 values')
        if not (0 <= self.decision_start < len(self.clock.decision_ns)
                and 0 <= self.measurement_start < len(times)
                and np.isfinite(self.periods_per_year) and self.periods_per_year > 0):
            raise ValueError('Research offsets and annualization must match their declared axes')
        owned = np.array(times, copy=True)
        owned.flags.writeable = False
        object.__setattr__(self, 'measurement_times', owned)


@dataclass(frozen=True)
class LoadedStudy(Generic[MarketValue, PanelValue, SpecValue]):
    study: Study
    market: MarketValue
    panel: PanelValue
    host_spec: SpecValue
    window: Window
    data_contract: DataContract
    program_spec: BarExecution | None = None


@dataclass(frozen=True)
class StudySnapshot:
    id: str
    study: Study
    data_snapshot: str
    execution_snapshot: str
    implementation: str
    configuration: str
    data_contract: DataContract
    evaluation: str
    decision: str
    execution: str
    allocation: Allocation
    precision: PrecisionPlan

@dataclass(frozen=True)
class Formula:
    expression: str
    kind: Literal['formula'] = 'formula'

    def normalized_complexity(self, ast_cost: int, max_cost: int) -> float:
        return ast_cost / max_cost


@dataclass(frozen=True)
class ArraySource:
    """Versioned content identity; empty scope is durable, otherwise session-resident."""
    kind: ArrayKind
    digest: str
    snapshot: str
    producer: str
    complexity: float = 0.
    scope: str = ''
    unit: ArrayUnit = 'score'

    def normalized_complexity(self, ast_cost: int, max_cost: int) -> float:
        return self.complexity


@dataclass(frozen=True)
class EqualWeight:
    kind: Literal['equal_weight'] = 'equal_weight'

    def normalized_complexity(self, ast_cost: int, max_cost: int) -> float:
        return 0.


@dataclass(frozen=True)
class ProgramReference:
    """Explicit deployed artifact and bounded canonical JSON parameters."""
    artifact_digest: str
    parameters: str = '{}'

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_digest, str) or re.fullmatch('[0-9a-f]{64}', self.artifact_digest) is None:
            raise ValueError('Program artifacts require a lowercase SHA256 digest')
        validate_program_parameters(self.parameters)


def validate_program_parameters(parameters: str) -> None:
    """Validate the immutable parameter payload at its external boundary."""
    if not isinstance(parameters, str) or len(parameters.encode('utf-8')) > 65536:
        raise ValueError('Program parameters require at most 64KiB of canonical JSON')
    value = json.loads(parameters)
    if not isinstance(value, dict) or json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                                                allow_nan=False) != parameters:
        raise ValueError('Program parameters require a canonical finite JSON object')


@dataclass(frozen=True)
class ProgramSource:
    """Replay identity for one original program variant; scope denotes a lease."""
    descriptor: ProgramDescriptor
    variant: int
    snapshot: str
    scope: str = ''
    complexity: float = 0.
    kind: Literal['program'] = 'program'

    def __post_init__(self) -> None:
        if not isinstance(self.descriptor, ProgramDescriptor):
            raise TypeError('Program sources require a typed descriptor')
        if type(self.variant) is not int or not 0 <= self.variant < self.descriptor.batch_size:
            raise ValueError('Program variant must identify an original descriptor row')
        if not isinstance(self.snapshot, str) or not self.snapshot or not isinstance(self.scope, str):
            raise ValueError('Program sources require a snapshot and string scope')
        if not 0 <= self.complexity <= 1 or self.kind != 'program':
            raise ValueError('Program complexity must be in [0,1] and kind must be program')

    def normalized_complexity(self, ast_cost: int, max_cost: int) -> float:
        return self.complexity


Source = Formula | ArraySource | ProgramSource | EqualWeight


@dataclass(frozen=True)
class ArrayBatch:
    """Explicit external-input boundary; registration takes an owned copy.

    values: NumPy/CuPy float32/float64 (B,D,S), on the engine's device or host.
    scores use unit score; NaN is missing and the selected Sizing rule defines
    its treatment. Targets are finite signed values in the declared unit;
    zero explicitly closes a position. Targets bypass score mapping.
    update_mask: optional bool (B,D,S); False emits no new target. Omission
    updates every asset. R1 daily kernels require every entry to be True.
    observed_at / fit_at: int64 UTC ns (B,D), no later than the decision time;
    NO_INFORMATION means the producer used no observations or fitting.
    producer: immutable model/program/version identifier supplied by the caller.
    decision_times: ISO UTC timestamps; assets: exact study asset order.
    quantity and notional units require an execution capability beyond R1.
    All arrays are borrowed until register_batch returns; callers may then
    mutate their buffers without changing the registered candidate.
    CUDA inputs must be ready on the runtime's current stream when submitted;
    producers using other streams establish that dependency before calling.
    complexity: optional B normalized costs in [0,1]; omission means zero.
    """
    kind: ArrayKind
    ids: tuple[str, ...]
    values: FloatArray
    snapshot: str
    producer: str
    decision_times: tuple[str, ...]
    assets: tuple[str, ...]
    observed_at: IntArray
    fit_at: IntArray
    complexity: tuple[float, ...] | None = None
    unit: ArrayUnit = 'score'
    update_mask: BoolArray | None = None


class ArrayStore(Protocol):
    """Persist validated host batches under their registration-computed identities.

    Sources follow batch row order and have empty scopes. The caller owns the
    immutable host buffers until put returns. Reads verify stored integrity.
    """
    def put(self, batch: ArrayBatch, sources: tuple[ArraySource, ...]) -> None: ...
    def read(self, source: ArraySource) -> ArrayBatch: ...

@dataclass(frozen=True)
class Candidate:
    id: str
    source: Source


@dataclass(frozen=True)
class FamilySeed:
    id: str
    expression: str
    hypothesis: str


@dataclass(frozen=True)
class YearMetrics:
    year: int
    periods: int
    sharpe: float
    attend: float
    lg: float


@dataclass(frozen=True)
class Metrics:
    """Common research metrics with an explicit activity measurement basis.

    target_book measures target changes per decision and sampled target exposure.
    executed_fills sums each bar's actual filled absolute notional divided by
    account equity after opening mark/funding/risk and before commands, then
    divides by research decision count. Independent accounts aggregate filled
    notional and opening equity, including unallocated cash, before the ratio.
    Exposure uses actual account positions.
    margin_bps uses the reported turnover denominator. Yearly attendance measures
    exposure-active periods; derivative cashflows are reported separately in risk.
    fills counts actual fill events for executed_fills; target_book leaves it None.
    """
    sharpe: float
    ann_ret: float
    maxdd: float
    cagr: float
    total_ret: float
    turnover: float
    exposure: float
    margin_bps: float
    max_single_weight: float
    concentration: float
    yearly: tuple[YearMetrics, ...]
    span: str
    n_periods: int
    activity_basis: Literal['target_book', 'executed_fills'] = 'target_book'
    fills: int | None = None


@dataclass(frozen=True)
class RiskSummary:
    """Window cashflows/counts; account state and first event use the full run.

    first_liquidation_ns identifies the execution event bar, not an exact tick fill.
    """
    model: str
    funding: float
    trading_fees: float
    liquidation_fees: float
    isolated_loss_cap: float
    liquidations: int
    first_liquidation_ns: int
    bankrupt: bool
    ending_equity: float
    max_gross_leverage: float
    gross_sampling: str = 'account_event_samples'


@dataclass(frozen=True)
class StatsRow:
    book: FloatArray | None
    returns: Array
    metrics: Metrics
    valid_returns: bool
    risk: RiskSummary | None = None


@dataclass
class EvaluationRow:
    source: Source
    ok: bool
    error: str | None
    precision: str
    tree: ParsedFormula | None = None
    book: FloatArray | None = None
    returns: FloatArray | None = None
    valid_returns: bool = False
    maxdd: float | None = None
    metrics: Metrics | None = None
    risk: RiskSummary | None = None
    view: tuple[float, float, float] = (float('nan'), float('nan'), float('nan'))
    behavior_digest: str | None = None


@dataclass(frozen=True)
class ConstraintCheck:
    """Raw metric change and threshold; None means an unattainable +infinity threshold."""

    name: str
    passed: bool
    delta: float
    required: float | None


@dataclass
class Proposal:
    passed: bool
    checks: tuple[ConstraintCheck, ...]
    vetoes: tuple[str, ...]
    weight: float
    trials: int
    backend: str
    reference_version: str
    valid: bool
    wsharpe: float = float('nan')
    sharpe: float = float('nan')
    cagr: float = float('nan')
    recent_growth: float = float('nan')
    maxdd: float = float('nan')
    utility: float = float('nan')


@dataclass(frozen=True)
class Verdict:
    source: Source
    passed: bool
    stopped_at: str | None
    reason: str
    message: str
    proposal: Proposal | None


@dataclass(frozen=True)
class DeviceBatch:
    rows: tuple[EvaluationRow, ...]
    verdicts: tuple[Verdict, ...]


@dataclass(frozen=True)
class ResearchContext:
    id: str
    snapshot: StudySnapshot
    portfolio: str


@dataclass(frozen=True)
class CandidateResult:
    id: str
    row: EvaluationRow
    verdict: Verdict
    accepted: bool = False
    weight: float = 0.


@dataclass(frozen=True)
class EvaluationResult:
    context: ResearchContext
    rows: tuple[CandidateResult, ...]


@dataclass(frozen=True)
class CommitResult:
    before: ResearchContext
    context: ResearchContext
    rows: tuple[CandidateResult, ...]


@dataclass(frozen=True)
class CandidateReport:
    id: str
    source: Source
    ok: bool
    error: str | None
    precision: str
    metrics: Metrics | None
    risk: RiskSummary | None
    decision: Verdict
    accepted: bool
    weight: float


@dataclass(frozen=True)
class EvaluationReport:
    context: ResearchContext
    rows: tuple[CandidateReport, ...]
    lease: str = ''


@dataclass(frozen=True)
class ResidentEvaluation:
    """One query's committable report and borrowed in-process device batch.

    Candidate IDs and context belong to report; batch preserves the same row
    order and exposes resident arrays without repeating evaluation. Direct
    program evaluation owns report.lease until commit, release or session close.
    """
    report: EvaluationReport
    batch: DeviceBatch


@dataclass(frozen=True)
class CommitReport:
    before: ResearchContext
    context: ResearchContext
    rows: tuple[CandidateReport, ...]


@dataclass(frozen=True)
class RegisteredBatch:
    """Candidate references plus an owned resident lease, or empty lease after retain."""
    context: ResearchContext
    candidates: tuple[Candidate, ...]
    lease: str


@dataclass(frozen=True)
class DataGrid:
    """Producer-visible coordinates, bound to one immutable data snapshot."""
    snapshot: str
    decision_times: tuple[str, ...]
    assets: tuple[str, ...]

    def __post_init__(self) -> None:
        values = tuple(utc_nanoseconds(value) for value in self.decision_times)
        if not values or any(right <= left for left, right in zip(values, values[1:])):
            raise ValueError('Decision times must be nonempty and strictly increasing')
        if not self.assets or len(set(self.assets)) != len(self.assets):
            raise ValueError('Grid assets must be nonempty and unique')
        object.__setattr__(self, 'decision_times', utc_timestamps(np.asarray(values, dtype=np.int64)))

    @property
    def time_ns(self) -> NDArray[np.int64]:
        values = np.asarray([utc_nanoseconds(value) for value in self.decision_times], dtype=np.int64)
        values.flags.writeable = False
        return values


@dataclass(frozen=True)
class PortfolioReport:
    context: ResearchContext
    sources: tuple[Source, ...]
    weights: tuple[float, ...]
    metrics: Metrics
    baseline: bool


@dataclass(frozen=True)
class WeightReport:
    context: ResearchContext
    weights: tuple[tuple[float, ...], ...]
    feasible: tuple[bool, ...]
    utility: tuple[float, ...]
    delta: tuple[tuple[float, ...], ...]


class LibraryStore(Protocol[LibraryValue]):
    persistent: bool
    def read(self, study: Study) -> LibraryValue: ...
    def commit(self, expected: LibraryValue, replacement: LibraryValue) -> None: ...


class ContextConflict(ValueError):
    """The supplied study or portfolio version no longer names the current state."""


@dataclass(frozen=True)
class WeightTrials:
    feasible: Array
    utility: Array
    delta: Array


@dataclass(frozen=True)
class WeightMetrics:
    valid: Array
    wsharpe: Array
    sharpe: Array
    cagr: Array
    recent_growth: Array
    maxdd: Array
    delta: Array
    required: Array
    checks: Array
    utility: Array

    @property
    def feasible(self):
        return self.valid & self.checks.all(-1)


@dataclass(frozen=True)
class WeightCommit:
    before: ResearchContext
    context: ResearchContext
    changed: bool
    utility: float

@dataclass(frozen=True)
class FamilyPoint:
    candidate: Candidate
    passed: bool
    sharpe: float | None
    maxdd: float | None
    attend: float | None
    gain: float | None
    d_cagr: float | None
    d_maxdd: float | None


@dataclass(frozen=True)
class Spread:
    n: int
    pass_rate: float
    median: float | None
    median_cagr: float | None
    median_maxdd: float | None
    p25: float | None
    p75: float | None


@dataclass(frozen=True)
class PlateauPoint:
    point: FamilyPoint
    neighbourhood_median: float
    neighbours: int


@dataclass(frozen=True)
class FieldSummary:
    field: str
    substitutes: int
    passed: int
    median_gain: float | None


@dataclass(frozen=True)
class SkippedFamily:
    id: str
    candidate: Candidate
    hypothesis: str
    error: str


@dataclass(frozen=True)
class FamilySummary:
    id: str
    hypothesis: str
    seed: FamilyPoint
    grid: Spread
    twins: Spread
    verdict: str
    seed_rank: float | None
    centre: PlateauPoint | None
    fields: tuple[FieldSummary, ...]


@dataclass(frozen=True)
class FamilyResult:
    context: ResearchContext
    families: tuple[FamilySummary, ...]
    variants: int
    skipped: tuple[SkippedFamily, ...]


class SessionBackend(Protocol[ResidentValue]):
    def context(self) -> ResearchContext: ...
    def refresh(self) -> ResearchContext: ...
    def grid(self) -> DataGrid: ...
    def describe(self) -> dict: ...
    def plan_program(self, program: StrategyProgram | ProgramReference, context: str) -> ProgramPlan: ...
    def simulate(self, program: StrategyProgram | ProgramReference, context: str, *, until: int | None = None,
                 checkpoint: ProgramCheckpoint | None = None) -> ProgramSimulation: ...
    def evaluate(self, candidates: tuple[Candidate, ...], context: str) -> EvaluationReport: ...
    def evaluate_device(self, candidates: tuple[Candidate, ...], context: str) -> DeviceBatch: ...
    def commit(self, candidates: tuple[Candidate, ...], context: str) -> CommitReport: ...
    def register(self, batch: ArrayBatch, context: str) -> RegisteredBatch: ...
    def register_program(self, program: StrategyProgram | ProgramReference, context: str) -> RegisteredBatch: ...
    def retain(self, candidates: tuple[Candidate, ...], context: str) -> RegisteredBatch: ...
    def release(self, lease: str) -> None: ...
    def diagnose(self, context: str) -> Diagnostics: ...
    def judge_family(self, families: Sequence[FamilySeed | CandidateGrid], context: str) -> FamilyResult: ...
    def portfolio(self, context: str) -> PortfolioReport: ...
    def trial_weights(self, weights: Array, context: str) -> WeightReport: ...
    def commit_weights(self, weights: tuple[float, ...], context: str) -> WeightCommit: ...
    def reweight(self, context: str, iterations: int, count: int, seed: int) -> ReweightResult: ...
    def resident(self) -> ResidentValue: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class ExecutionSchedule:
    """Owned event clock and compact target/measurement mappings.

    event_ns is ordered UTC ns (E,). decision_index (E,) is -1 to retain
    actual positions or an index in the target grid (D,). measurement_index
    (E,) assigns every event to one contiguous valuation period in [0,V).
    The compiler guarantees that a target is visible before its event.
    """
    event_ns: NDArray[np.int64]
    decision_index: NDArray[np.int64]
    measurement_index: NDArray[np.int64]
    decision_count: int
    measurement_count: int

    def __post_init__(self) -> None:
        names = ('event_ns', 'decision_index', 'measurement_index')
        arrays = tuple(np.asarray(getattr(self, name)) for name in names)
        times, decisions, measurements = arrays
        if (self.decision_count < 1 or self.measurement_count < 1
                or any(a.dtype != np.dtype('int64') or a.ndim != 1 for a in arrays)
                or not len(times) or any(a.shape != times.shape for a in arrays)):
            raise ValueError('Execution schedule requires aligned int64 event arrays and positive axis sizes')
        if np.any(times == NO_INFORMATION) or np.any(times[1:] <= times[:-1]):
            raise ValueError('Execution events must be strictly increasing UTC nanoseconds')
        updates = decisions[decisions >= 0]
        if (np.any(decisions < -1) or np.any(decisions >= self.decision_count)
                or np.any(updates[1:] <= updates[:-1])):
            raise ValueError('Target updates must be unique ordered decision indices or -1')
        if (measurements[0] != 0 or measurements[-1] != self.measurement_count - 1
                or np.any(np.diff(measurements) < 0) or np.any(np.diff(measurements) > 1)):
            raise ValueError('Every measurement period must have a contiguous nonempty event interval')
        for name, array in zip(names, arrays):
            owned = np.array(array, copy=True, order='C')
            owned.flags.writeable = False
            object.__setattr__(self, name, owned)

    @property
    def snapshot(self) -> str:
        digest = hashlib.sha256(str((self.decision_count, self.measurement_count)).encode())
        for array in (self.event_ns, self.decision_index, self.measurement_index):
            digest.update(array.tobytes())
        return digest.hexdigest()

    def prefix(self, measurements: int, decisions: int) -> ExecutionSchedule:
        if not 0 < measurements <= self.measurement_count or not 0 < decisions <= self.decision_count:
            raise ValueError('Schedule prefix bounds must lie within both declared axes')
        end = int(np.searchsorted(self.measurement_index, measurements))
        return ExecutionSchedule(self.event_ns[:end], self.decision_index[:end],
                                 self.measurement_index[:end], decisions, measurements)


CASH_ARRAY_FIELDS = ('open_raw', 'prev_close', 'open_ff0', 'close_ff0', 'cost_buy',
                     'live', 'cost_sell', 'block', 'carry')


@dataclass(frozen=True)
class ExecutionSpec:
    schedule: ExecutionSchedule
    open_raw: HostArray
    prev_close: HostArray
    open_ff0: HostArray
    close_ff0: HostArray
    cost_buy: HostArray
    live: HostArray
    cost_sell: HostArray | None = None
    block: HostArray | None = None
    carry: HostArray | None = None

    def __post_init__(self):
        shape = np.asarray(self.open_raw).shape
        if len(shape) != 2 or shape[0] != len(self.schedule.event_ns) or shape[1] < 1:
            raise ValueError('Cash execution prices must align with the event and asset axes')
        for name in CASH_ARRAY_FIELDS:
            value = getattr(self, name)
            if value is not None:
                owned = np.array(value, order='C', copy=True)
                if owned.shape != shape:
                    raise ValueError('Cash execution arrays must share the event and asset axes')
                owned.flags.writeable = False
                object.__setattr__(self, name, owned)

    @property
    def shape(self):
        return self.schedule.decision_count, self.open_raw.shape[1]

    @property
    def snapshot(self):
        digest = hashlib.sha256(self.schedule.snapshot.encode())
        for name in CASH_ARRAY_FIELDS:
            value = getattr(self, name)
            digest.update(name.encode())
            if value is not None:
                digest.update(str((value.shape, value.dtype.str)).encode())
                digest.update(value.tobytes())
        return digest.hexdigest()

    @property
    def symmetric(self) -> bool:
        return self.cost_sell is None and self.block is None and self.carry is None

    def prefix(self, measurements: int, decisions: int) -> ExecutionSpec:
        schedule = self.schedule.prefix(measurements, decisions)
        end = len(schedule.event_ns)
        return replace(self, schedule=schedule,
            **{name: getattr(self, name)[:end] for name in CASH_ARRAY_FIELDS if getattr(self, name) is not None})


PERPETUAL_MODEL = 'linear_usdt_isolated_event_mark_touch_v2'
PERPETUAL_ARRAY_FIELDS = ('trade_open', 'mark_open', 'mark_high', 'mark_low', 'mark_close',
                'funding_rate', 'funding_mark', 'maintenance_rate', 'fee_rate', 'liquidation_fee')


@dataclass(frozen=True)
class PerpetualSpec:
    """Owned prices/rates (E,S) and independent decision/measurement mappings.

    Prices use USDT per asset unit. Funding, maintenance and fees are notional
    fractions. Funding is applied before target updates at each event. Intrabar
    liquidation timestamps identify the bar, not a known tick. Leverage controls
    isolated collateral; gross_limit separately bounds target notional/equity.
    """
    schedule: ExecutionSchedule
    trade_open: HostArray
    mark_open: HostArray
    mark_high: HostArray
    mark_low: HostArray
    mark_close: HostArray
    funding_rate: HostArray
    funding_mark: HostArray
    maintenance_rate: HostArray
    fee_rate: HostArray
    liquidation_fee: HostArray
    leverage: float
    gross_limit: float

    def __post_init__(self):
        shape = np.asarray(self.trade_open).shape
        if len(shape) != 2 or shape[0] != len(self.schedule.event_ns) or shape[1] < 1:
            raise ValueError('Perpetual prices must align with the event and asset axes')
        for name in PERPETUAL_ARRAY_FIELDS:
            owned = np.array(getattr(self, name), dtype=np.float64, order='C', copy=True)
            if owned.shape != shape:
                raise ValueError('Perpetual arrays must share the event and asset axes')
            owned.flags.writeable = False
            object.__setattr__(self, name, owned)
        ratio = self.trade_open / self.mark_open
        if not np.all(self.leverage * (np.abs(1 - ratio).max(1)
                                      + (self.fee_rate * ratio).max(1)) < 1):
            raise ValueError('non-monotone rebalance root')

    @property
    def shape(self):
        return self.schedule.decision_count, self.trade_open.shape[1]

    @property
    def snapshot(self):
        digest = hashlib.sha256(PERPETUAL_MODEL.encode())
        digest.update(self.schedule.snapshot.encode())
        digest.update(str((self.leverage, self.gross_limit)).encode())
        for name in PERPETUAL_ARRAY_FIELDS:
            value = getattr(self, name)
            digest.update(name.encode())
            digest.update(str((value.shape, value.dtype.str)).encode())
            digest.update(value.tobytes())
        return digest.hexdigest()

    def prefix(self, measurements: int, decisions: int) -> PerpetualSpec:
        schedule = self.schedule.prefix(measurements, decisions)
        end = len(schedule.event_ns)
        return replace(self, schedule=schedule, **{name: getattr(self, name)[:end] for name in PERPETUAL_ARRAY_FIELDS})


@dataclass(frozen=True)
class ReweightResult:
    before: ResearchContext
    context: ResearchContext
    changed: bool
    evaluations: int
    utility: float | None = None
    weights: tuple[float, ...] = ()


@dataclass(frozen=True)
class PortfolioView:
    """Borrowed read-only books and return statistics for a fixed portfolio."""
    U: FloatArray | None
    W: FloatArray | None
    rr: FloatArray
    p: FloatArray
    metrics: Metrics
    sharpe: float
    cagr: float
    maxdd: float
    wsharpe: float
    recent_growth: float
    n: int
    seed: bool
    keys: tuple[Source, ...]
    version: str
    mean_exposure: float


@dataclass(frozen=True)
class ContextFeature:
    name: str
    value: float


@dataclass(frozen=True)
class DrawdownWindow:
    peak: str | None
    trough: str
    recovery: str | None
    drawdown: float


@dataclass(frozen=True)
class SingleGate:
    prior_mean: float
    shrinkage: float
    reference_sharpe: float
    required_appraisal: float
    anchor_correlation: float
    anchor_sharpe: float
    pool_size: int
    pool_target: float
    recent_years: int
    recent_year_sharpe: float


@dataclass(frozen=True)
class Diagnostics:
    context: ResearchContext
    features: tuple[ContextFeature, ...]
    drawdown_window: DrawdownWindow
    member_correlations: tuple[tuple[float, ...], ...]
    capacity_remaining: int
    gate: SingleGate | None = None


@dataclass(frozen=True)
class DiscoveryRow:
    candidate: Candidate
    log_reward: float
    passed: bool
    g1: bool
    g2: bool
    decision: Verdict
    rho: float | None = None
    utility: float | None = None
    error: str | None = None
    co_loss: float = 0.
    novelty: float = 0.
    duplicate: bool = False
    weight: float = 0.
    delta: tuple[ConstraintCheck, ...] = ()
    backend: str = ''
    g3: bool = False
    sharpe: float | None = None
    cagr: float | None = None
    maxdd: float | None = None
    rho_portfolio: float | None = None
    rho_members: float | None = None
    appraisal: float | None = None
    required_sharpe: float | None = None
    recent_years: tuple[float, ...] = ()


@dataclass(frozen=True)
class DiscoveryReport:
    context: ResearchContext
    objective: str
    rows: tuple[DiscoveryRow, ...]
    delayed: bool = False


@dataclass(frozen=True)
class CandidateAcceptance:
    candidate: Candidate
    weight: float
    utility: float



@dataclass(frozen=True)
class ComputeDevice:
    kind: Literal['cpu', 'cuda']
    index: int = 0


class ResidentVector(Protocol):
    """Owned finite FP32 (B,) feedback, borrowed read-only until scope exit.

    CUDA consumers name their stream; the borrow orders production before
    consumption. Close waits for consumers and releases the allocation.
    """
    device: ComputeDevice
    shape: tuple[int, ...]

    def borrow(self, stream: int | None = None) -> AbstractContextManager[FloatArray]: ...
    def to_host(self) -> HostArray: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class ProposalRequest:
    context: ResearchContext
    count: int
    features: tuple[float, ...]


@dataclass(frozen=True)
class ProposalBatch:
    """One producer-owned, ordered batch; only its exact handle may be learned."""
    owner: str
    sequence: int
    context: ResearchContext
    ordinary: int
    candidates: tuple[Candidate, ...]

    @property
    def auxiliary(self) -> int:
        return len(self.candidates) - self.ordinary


@dataclass(frozen=True)
class LearningFeedback:
    batch: ProposalBatch
    rewards: ResidentVector

    def __post_init__(self) -> None:
        if self.rewards.shape != (len(self.batch.candidates),):
            raise ValueError('feedback must preserve the proposal batch axis')


@dataclass(frozen=True)
class TrainingResult:
    metrics: Mapping[str, float] = field(default_factory=dict)


class ProposalProducer(Protocol):
    """One run owns its resources and at most one unreleased proposal batch."""
    name: str

    def propose(self, request: ProposalRequest) -> ProposalBatch | None: ...
    def learn(self, feedback: LearningFeedback) -> TrainingResult: ...
    def release(self, batch: ProposalBatch) -> None: ...
    def dump(self) -> bytes: ...
    def load(self, payload: bytes) -> None: ...


class Proposer(Protocol):
    """Resource-free specification; mine owns each opened producer's lifetime."""
    name: str

    def open(self, device: ComputeDevice) -> AbstractContextManager[ProposalProducer]: ...


@dataclass(frozen=True)
class ProposerState:
    name: str
    payload: bytes


@dataclass(frozen=True)
class MiningTimings:
    sampling_seconds: float = 0.
    evaluation_seconds: float = 0.
    training_seconds: float = 0.
    commit_seconds: float = 0.
    reweight_seconds: float = 0.
    retention_seconds: float = 0.
    checkpoint_seconds: float = 0.


@dataclass(frozen=True)
class MiningCheckpoint:
    signature: str
    epoch: int
    portfolio: PortfolioReport
    archive: tuple[Candidate, ...]
    counters: Mapping[str, int]
    accepted: int
    elapsed_seconds: float
    proposers: tuple[ProposerState, ...]


@dataclass(frozen=True)
class MiningStatus:
    phase: str
    epoch: int
    epochs: int
    accepted: int
    counters: Mapping[str, int]
    elapsed_seconds: float
    portfolio: PortfolioReport
    timings: MiningTimings = field(default_factory=MiningTimings)
    recent: tuple[DiscoveryRow, ...] = ()
    objective: str = 'portfolio'
    gate: SingleGate | None = None
    proposer: str | None = None
    noise_control: bool = False
    training: TrainingResult | None = None
    device: str = ''
    used_mib: int = 0


@dataclass(frozen=True)
class MiningEpoch:
    epoch: int
    signature: str
    proposer: str
    archive_version: int
    report: DiscoveryReport
    accepted: Candidate | None
    reweight: ReweightResult | None
    training: TrainingResult
    timings: MiningTimings
    counters: Mapping[str, int]
    checkpoint_candidates: tuple[Candidate, ...] = ()


class CheckpointStore(Protocol):
    def read(self) -> MiningCheckpoint | None: ...
    def publish(self, checkpoint: MiningCheckpoint) -> None: ...


class TelemetrySink(Protocol):
    def stopping(self) -> bool: ...
    def publish(self, status: MiningStatus) -> None: ...
    def record(self, epoch: MiningEpoch) -> None: ...




@dataclass(frozen=True)
class CandidateGrid:
    """Explicit candidate lattice; variants include seed exactly once.

    Each coordinate has the same positive dimension and contains integer rung
    indices. Adjacent points differ by one rung along one parameter axis.
    """
    seed: Candidate
    variants: tuple[Candidate, ...]
    parameter_coords: tuple[tuple[int, ...], ...]
    hypothesis: str = ''


@dataclass(frozen=True)
class Fold:
    """Half-open fit and prediction intervals, expressed as UTC int64 ns."""
    fit_start: int
    fit_stop: int
    predict_start: int
    predict_stop: int


@dataclass(frozen=True)
class FoldView:
    """Owned causal feature view (D,S); host arrays are read-only.

    start/stop are UTC ns interval bounds with an exclusive stop. Arrays
    retain their host/device dtype and cannot expose later rows through a base.
    """
    start: int
    stop: int
    decision_times: tuple[str, ...]
    assets: tuple[str, ...]
    features: Mapping[str, FloatArray]
    eligibility: BoolArray


@dataclass(frozen=True)
class FoldResult:
    """Realized net-return signs in a half-open UTC ns measurement interval.

    Bounds are statistics period boundaries, not decisions or fills. The
    terminal stop is the final period start plus its declared duration.
    Mean returns/signs follow registered candidate order. Invalid returns have
    a missing mean and sign zero; valid zero means also have sign zero.
    """
    fold: Fold
    measurement_start: int
    measurement_stop: int
    mean_returns: tuple[float | None, ...]
    signs: tuple[int, ...]


@dataclass(frozen=True)
class WalkForwardResult:
    """Registered outputs and realized cross-fold directional stability.

    sign_consistency is the larger positive/negative fold count divided by all
    folds for each candidate. Zero/invalid folds contribute to neither count.
    A consistently negative candidate can score 1; inspect fold signs as well.
    """
    registered: RegisteredBatch
    folds: tuple[FoldResult, ...]
    sign_consistency: tuple[float, ...]


@dataclass(frozen=True)
class ResourceStatus:
    device: str
    used_mib: int


class OrderAction(IntEnum):
    NONE = 0
    NEW = 1
    CANCEL = 2
    AMEND = 3


class OrderType(IntEnum):
    MARKET = 1
    LIMIT = 2
    STOP = 3


class LiquidityRole(IntEnum):
    NONE = 0
    MAKER = 1
    TAKER = 2


class OrderStatus(IntEnum):
    NONE = 0
    ACCEPTED = 1
    PARTIAL = 2
    FILLED = 3
    CANCELLED = 4
    REJECTED = 5
    EXPIRED = 6
    LIQUIDATED = 7


class OrderReason(IntEnum):
    NONE = 0
    CAPACITY = 1
    INSUFFICIENT_FUNDS = 2
    NOT_TRADABLE = 3
    UNKNOWN_ORDER = 4
    INVALID_ORDER = 5
    GROSS_LIMIT = 6
    REDUCE_ONLY = 7
    EXPIRED = 8
    LIQUIDATION = 9
    UNSETTLED = 10
    CORPORATE_ACTION = 11


class OrderStage(IntEnum):
    COMMAND = 1
    OPEN = 2
    PATH_FIRST = 3
    PATH_SECOND = 4
    CLOSE = 5
    FUNDING = 6
    LIQUIDATION = 7


@dataclass(frozen=True)
class OrderCommands:
    """Owned command columns (B,K), resident on the selected execution device.

    Action zero pads unused slots. Positive new IDs increase strictly within
    each assigned owner namespace, and (owner,id) identifies an order. Internal
    negative target-child IDs cannot be cancelled by programs. Amend replaces its remaining
    quantity and conditions and resets priority. Prices are quote currency per
    asset unit; reservation_price values funding requirements, not a fill cap.
    valid_until is inclusive UTC ns aligned to a compiled base-bar open, or
    -1 for good until cancelled. Expiry between base opens is unsupported.
    """
    action: IntArray
    owner: IntArray
    id: IntArray
    asset: IntArray
    side: IntArray
    quantity: FloatArray
    type: IntArray
    price: FloatArray
    reservation_price: FloatArray
    valid_until: IntArray
    reduce_only: BoolArray


@dataclass(frozen=True)
class OrderSnapshot:
    """Detached active-order columns (B,Q); action zero denotes an empty slot."""
    action: IntArray
    owner: IntArray
    id: IntArray
    asset: IntArray
    side: IntArray
    quantity: FloatArray
    type: IntArray
    price: FloatArray
    reservation_price: FloatArray
    valid_until: IntArray
    reduce_only: BoolArray
    remaining: FloatArray
    reserved: FloatArray
    status: IntArray


@dataclass(frozen=True)
class AccountSnapshot:
    """Detached actual ledger: quote cash/equity (B,), asset quantities (B,S).

    reserved_cash (B,) and margin (B,S) use quote currency; cross margin is
    marked initial collateral, while isolated margin includes leg PnL.
    available_cash deducts unsettled proceeds, collateral and active reserves.
    sellable_quantity excludes T+1 locks but precedes active sell reservations.
    entry_price (B,S) uses quote currency per unit. Mutating a device snapshot
    cannot alter the execution ledger. CPU snapshots are also read-only.
    """
    cash: FloatArray
    equity: FloatArray
    reserved_cash: FloatArray
    quantity: FloatArray
    margin: FloatArray
    entry_price: FloatArray
    orders: OrderSnapshot
    available_cash: FloatArray
    sellable_quantity: FloatArray


@dataclass(frozen=True)
class ExecutionFeedback:
    """Detached bounded audit columns (B,F), valid entries counted by (B,).

    Each base event is delivered once. F is bounded by
    K + M*S + 4*Q + (M+1)*(Q+S), where M is owner count, independent of
    history length. Sequence is stable within each account; quantities are
    unsigned fills except owner-attributed LIQUIDATED rows, whose quantities
    are signed owner contributions. A global liquidation row has owner=-1,
    unsigned aggregate quantity and the global fee; owner rows allocate that
    fee by absolute contribution. Do not sum global and owner fees twice.
    Prices are quote/unit and fees are quote currency; liquidity names each
    fill role, with NONE for non-fill events. Maker rebates are negative fees. Account
    reflects all event stages, including its final close valuation.
    """
    account: AccountSnapshot
    count: IntArray
    status: IntArray
    reason: IntArray
    stage: IntArray
    sequence: IntArray
    owner: IntArray
    id: IntArray
    asset: IntArray
    time: IntArray
    quantity: FloatArray
    price: FloatArray
    fee: FloatArray
    remaining: FloatArray
    event_ns: int
    available_ns: int
    liquidity: IntArray


@dataclass(frozen=True)
class AccountCheckpoint:
    """Detached host account state at a completed base-bar boundary.

    Asset/owner/order state is bounded by B,S,M,Q. Measurement arrays hold only
    completed statistics columns; commands, audit buffers and market history
    are not checkpointed. restore validates shapes against the opened executor.
    """
    snapshot: str
    cursor: int
    assets: HostArray
    cash: HostArray
    meta: HostArray
    orders: HostArray
    order_values: HostArray
    owners: HostArray
    owner_locked: HostArray
    highwater: HostArray
    targets: HostArray
    units: HostArray
    coefficients: HostArray
    pending_targets: HostArray
    target_dirty: HostArray
    activity: HostArray
    digest: HostArray
    returns: HostArray
    equity: HostArray
    active: HostArray
    risk_flows: HostArray
    risk_counts: HostArray
    risk_previous: HostArray
    format: str = 'order-account-v2'


class AccountExecutionPort(Protocol):
    """Owned native account replacement point for bounded chronological replay.

    Arrays reside on the selected device. Commands/targets use batch B, owner M
    and asset S axes; snapshots and measured results detach their buffers.
    """
    batch: int
    feedback_capacity: int

    def configure_targets(self, units: tuple[str, ...], coefficients: FloatArray) -> None: ...
    def advance(self, event: int, commands: OrderCommands, *, targets: FloatArray | None = None,
                update_mask: Array | None = None) -> ExecutionFeedback: ...
    def snapshot(self) -> AccountSnapshot: ...
    def owner_quantities(self, owner: int | None = None) -> FloatArray: ...
    def activity(self) -> FloatArray: ...
    def bar_activity(self) -> FloatArray: ...
    def risk(self, normalization: float) -> ExecutionRiskBatch: ...
    def returns(self) -> FloatArray: ...
    def equity(self) -> FloatArray: ...
    def active(self) -> Array: ...
    def behavior_digests(self) -> tuple[str, ...]: ...
    def checkpoint(self) -> AccountCheckpoint: ...
    def restore(self, state: AccountCheckpoint) -> None: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class BarExecution:
    """Owned host Bar execution inputs; numeric/bool panels have shape (E,S).

    Fees, funding and margin rates are notional fractions. Volume and qty_step
    use asset units; zero step/tick explicitly declares continuous quantities
    or prices. event_close_ns (E,) declares when the complete Bar is visible.
    Maker/taker fees describe liquidity roles; fee_buy/sell add directional
    charges. Splits and payable dividends are explicit pre-open events.
    Equity sessions unlock prior-day inventory/cash and expire day orders.
    """
    schedule: ExecutionSchedule
    event_close_ns: NDArray[np.int64]
    trade_open: HostArray
    trade_high: HostArray
    trade_low: HostArray
    trade_close: HostArray
    volume: HostArray
    mark_open: HostArray
    mark_high: HostArray
    mark_low: HostArray
    mark_close: HostArray
    fee_buy: HostArray
    fee_sell: HostArray
    funding_rate: HostArray
    funding_mark: HostArray
    maintenance_rate: HostArray
    liquidation_fee: HostArray
    buy_allowed: HostArray
    sell_allowed: HostArray
    quote_valid: HostArray
    qty_step: HostArray
    price_tick: HostArray
    market: str
    maker_fee: HostArray
    taker_fee: HostArray
    split_factor: HostArray
    cash_dividend: HostArray
    min_quantity: HostArray
    limit_up: HostArray
    limit_down: HostArray
    session_start: HostArray
    session_end: HostArray
    inventory_t1: HostArray
    cash_t1: HostArray
    odd_lot_sell: HostArray
    fill_step: HostArray

    def __post_init__(self) -> None:
        shape = np.asarray(self.trade_open).shape
        if len(shape) != 2 or shape[0] != len(self.schedule.event_ns) or shape[1] < 1:
            raise ValueError('Bar execution panels must align with the execution event and asset axes')
        close = np.asarray(self.event_close_ns)
        if (close.dtype != np.dtype('int64') or close.shape != self.schedule.event_ns.shape
                or np.any(close <= self.schedule.event_ns)
                or np.any(close[:-1] > self.schedule.event_ns[1:])):
            raise ValueError('Bar close timestamps must follow their opens and cannot overlap later bars')
        if self.market not in ('crypto_spot', 'crypto_perpetual', 'us_equity', 'a_share'):
            raise ValueError('Unknown Bar execution market')
        close = np.array(close, copy=True, order='C')
        close.flags.writeable = False
        object.__setattr__(self, 'event_close_ns', close)
        boolean = {'buy_allowed', 'sell_allowed', 'quote_valid', 'session_start',
                   'session_end', 'inventory_t1', 'cash_t1', 'odd_lot_sell'}
        for item in fields(self):
            if item.name in ('schedule', 'event_close_ns', 'market'):
                continue
            value = np.asarray(getattr(self, item.name))
            if value.shape != shape:
                raise ValueError('Bar execution panels must share the event and asset axes')
            if item.name in boolean:
                if value.dtype != np.dtype('bool'):
                    raise TypeError('Bar execution eligibility columns require bool dtype')
                owned = np.array(value, dtype=np.bool_, copy=True, order='C')
            else:
                owned = np.array(value, dtype=np.float64, copy=True, order='C')
            owned.flags.writeable = False
            object.__setattr__(self, item.name, owned)
        if (np.any(~np.isfinite(self.volume)) or np.any(self.volume < 0)
                or np.any(~np.isfinite(self.qty_step)) or np.any(self.qty_step < 0)
                or np.any(~np.isfinite(self.fill_step)) or np.any(self.fill_step < 0)
                or np.any(~np.isfinite(self.price_tick)) or np.any(self.price_tick < 0)):
            raise ValueError('Bar volume and quantity/price steps must be finite and nonnegative')
        if np.any((self.buy_allowed | self.sell_allowed) & ~self.quote_valid):
            raise ValueError('Executable Bar directions require valid quotes')
        for prefix in ('trade', 'mark'):
            o, h, l, c = (getattr(self, prefix + '_' + suffix)
                          for suffix in ('open', 'high', 'low', 'close'))
            prices = np.stack((o, h, l, c))
            if (np.any(self.quote_valid & (~np.isfinite(prices).all(0) | (prices <= 0).any(0)))
                    or np.any(self.quote_valid & ((h < np.maximum(o, c))
                                                 | (l > np.minimum(o, c)) | (h < l)))):
                raise ValueError('Valid Bar quotes require positive finite and consistent OHLC prices')
            if prefix == 'mark':
                if (np.any(~np.isfinite(prices)) or np.any(prices < 0)
                        or np.any((h < np.maximum(o, c)) | (l > np.minimum(o, c)) | (h < l))):
                    raise ValueError('Valuation marks must be finite nonnegative OHLC, including unquoted assets')
                known = np.maximum.accumulate(self.quote_valid, axis=0)
                if np.any(known & (prices <= 0).any(0)):
                    raise ValueError('Valuation marks must stay positive after an asset first has a valid quote')
        for name in ('fee_buy', 'fee_sell', 'taker_fee', 'maintenance_rate', 'liquidation_fee'):
            value = getattr(self, name)
            if np.any(~np.isfinite(value)) or np.any((value < 0) | (value >= 1)):
                raise ValueError('Bar fee and maintenance rates require finite fractions in [0,1)')
        if np.any(~np.isfinite(self.maker_fee) | (self.maker_fee <= -1) | (self.maker_fee >= 1)):
            raise ValueError('Maker rates require finite fractions in (-1,1), including rebates')
        if np.any(~np.isfinite(self.split_factor) | (self.split_factor <= 0)):
            raise ValueError('Split factors must be positive finite post/pre share ratios')
        for name in ('cash_dividend', 'min_quantity', 'limit_up', 'limit_down'):
            value = getattr(self, name)
            if np.any(~np.isfinite(value) | (value < 0)):
                raise ValueError(f'{name} must contain finite nonnegative values')
        if np.any((self.limit_up > 0) & (self.limit_down > self.limit_up)):
            raise ValueError('Price limits must be ordered; zero means no limit')
        if np.any(self.cash_t1 != self.cash_t1[:, :1]):
            raise ValueError('Cash settlement policy must be shared across an account')
        if np.any(self.cash_t1 & (self.session_start != self.session_start[:, :1])):
            raise ValueError('Cash settlement requires a common account session boundary')
        if (np.any(~np.isfinite(self.funding_rate))
                or np.any(~np.isfinite(self.funding_mark)) or np.any(self.funding_mark < 0)
                or np.any((self.funding_rate != 0)
                          & (~np.isfinite(self.funding_mark) | (self.funding_mark <= 0)))):
            raise ValueError('Funding requires finite rates and positive settlement marks')

    @cached_property
    def snapshot(self) -> str:
        digest = hashlib.sha256((self.schedule.snapshot + self.market).encode())
        for item in fields(self):
            if item.name in ('schedule', 'market'):
                continue
            value = getattr(self, item.name)
            digest.update(item.name.encode())
            digest.update(str((value.shape, value.dtype.str)).encode())
            digest.update(value.tobytes())
        return digest.hexdigest()


@dataclass(frozen=True)
class TensorState:
    """Declared private tensor name, dtype and shape, excluding the batch axis."""
    name: str
    dtype: Literal['float32', 'float64', 'int32', 'int64', 'uint8', 'bool']
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if (not isinstance(self.name, str) or not self.name
                or self.dtype not in ('float32', 'float64', 'int32', 'int64', 'uint8', 'bool')):
            raise ValueError('Tensor state requires a name and a supported explicit dtype')
        if not isinstance(self.shape, tuple) or any(type(size) is not int or size < 1 for size in self.shape):
            raise ValueError('Tensor state dimensions must be positive integers')


@dataclass(frozen=True)
class ProgramDescriptor:
    """Immutable deployed identity and declared bounded program capabilities.

    artifact_digest and parameter_digest are SHA256 identities, not source or
    pickle payloads. Programs create fresh private state at every open; shared
    model parameters remain immutable. Arbitrary Python callbacks are not
    automatically compiled for a device or certified as market independent.
    """
    id: str
    artifact_digest: str
    parameters: str = '{}'
    state_schema: tuple[TensorState, ...] = ()
    features: tuple[str, ...] = ()
    history: int = 1
    batch_size: int = 1
    devices: tuple[Literal['cpu', 'cuda'], ...] = ('cpu',)
    seed: int = 0
    fit_at: int = NO_INFORMATION
    checkpointable: bool = False
    protocol: int = 3

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError('Programs require a nonempty stable ID')
        ProgramReference(self.artifact_digest, self.parameters)
        if self.protocol != 3:
            raise ValueError('Unsupported program protocol')
        if type(self.checkpointable) is not bool:
            raise TypeError('Program checkpoint capability must be explicit bool')
        if any(not isinstance(value, tuple) for value in (self.state_schema, self.features, self.devices)):
            raise TypeError('Program schemas, features and devices require immutable tuples')
        if any(type(value) is not int or value < 1 for value in (self.history, self.batch_size)):
            raise ValueError('Program history and batch_size must be positive integers')
        if type(self.seed) is not int or self.seed < 0 or type(self.fit_at) is not int:
            raise ValueError('Program seed must be nonnegative and fit_at must be integer UTC ns')
        if (not self.devices or len(set(self.devices)) != len(self.devices)
                or any(device not in ('cpu', 'cuda') for device in self.devices)):
            raise ValueError('Programs must declare unique supported devices')
        if len(set(self.features)) != len(self.features) or any(not isinstance(name, str) or not name for name in self.features):
            raise ValueError('Program features must be unique nonempty field names')
        if (any(not isinstance(item, TensorState) for item in self.state_schema)
                or len({item.name for item in self.state_schema}) != len(self.state_schema)):
            raise ValueError('Program state schema requires unique typed tensor names')

    @property
    def parameter_digest(self) -> str:
        return hashlib.sha256(self.parameters.encode('utf-8')).hexdigest()


@dataclass(frozen=True)
class ProgramPlan:
    """Declared simulation capabilities before opening private program state."""
    device: ComputeDevice
    reason: str
    precision: Literal['mixed', 'paired']
    market: str
    model: str
    bar_path: str
    participation: float
    batch_size: int
    history: int
    order_capacity: int
    command_capacity: int
    state_bytes: int
    feature_window_bytes: int
    feedback_capacity: int
    order_types: tuple[str, ...]
    decision_latency: str
    quote_policy: str
    snapshot: str
    checkpointable: bool = False


@dataclass(frozen=True)
class ProgramContext:
    """Run initialization metadata; no future timeline, panel or ledger handle.

    CUDA callbacks must order their work on stream, including framework
    interop such as a PyTorch ExternalStream. Private state stays on device.
    batch_size is the effective run count; descriptor.batch_size describes the
    original candidate batch. variant_indices is resident int64 (B,), while
    initial_cash and intent_coefficients are resident floating (B,) arrays.
    Shared accounts scale emitted order quantities once as unscaled intents.
    To close an actual owner quantity in full, emit that quantity divided by
    intent_coefficients; emitting the actual quantity without this conversion
    requests a scaled partial close. Independent accounts scale initial funding
    and expose intent_coefficients=1, so their order quantities are literal.
    owner is the fixed command namespace for this member.
    """
    descriptor: ProgramDescriptor
    assets: tuple[str, ...]
    plan: ProgramPlan
    xp: ModuleType
    stream: int | None
    initial_cash: FloatArray
    batch_size: int
    variant_indices: IntArray
    intent_coefficients: FloatArray
    owner: int


@dataclass(frozen=True)
class ProgramEvent:
    """One decision with copied causal (H_visible,S) fields and actual account.

    history_times is host int64 UTC ns and contains no future coordinates.
    Feature windows own their buffers so array bases cannot expose later
    observations. eligibility has the same bounded history and asset axes.
    owner_quantity is resident signed (B,S) attribution in the actual account;
    it does not replace the aggregate actual account or confer separate cash.
    """
    decision_ns: int
    history_times: NDArray[np.int64]
    features: Mapping[str, FloatArray]
    eligibility: BoolArray
    account: AccountSnapshot
    owner_quantity: FloatArray


@runtime_checkable
class ProgramRun(Protocol):
    """Fresh causal state, closed once even when a callback fails.

    Callbacks start at the first executable research decision. Earlier flat
    account events are skipped; the bounded feature window supplies declared
    historical context. Feedback cannot submit orders between decisions.
    """
    def on_event(self, event: ProgramEvent) -> OrderCommands: ...
    def on_feedback(self, feedback: ExecutionFeedback) -> None: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class StateTensor:
    """A named private (B,*shape) state tensor, including declared RNG state."""
    name: str
    values: Array


@runtime_checkable
class CheckpointableRun(ProgramRun, Protocol):
    """Explicit complete private-state transfer; hidden mutable state is invalid."""
    def snapshot(self) -> tuple[StateTensor, ...]: ...
    def restore(self, state: tuple[StateTensor, ...]) -> None: ...


@runtime_checkable
class StrategyProgram(Protocol):
    @property
    def descriptor(self) -> ProgramDescriptor: ...
    def open(self, context: ProgramContext) -> ProgramRun: ...


@dataclass(frozen=True)
class ProgramArtifact:
    """Trusted deployed loader over integrity-checked bytes and explicit semantics.

    The semantic version identifies the preinstalled implementation. The loader
    interprets the payload and parameters; it must never execute supplied source
    or deserialize Python objects. Its returned descriptor must bind this digest.
    """
    payload: bytes
    semantic_version: str
    build: Callable[[bytes, str, str], StrategyProgram]

    def __post_init__(self) -> None:
        if not isinstance(self.payload, bytes) or not self.payload:
            raise ValueError('Program artifacts require nonempty immutable bytes')
        if not isinstance(self.semantic_version, str) or not self.semantic_version or not callable(self.build):
            raise ValueError('Program artifacts require explicit implementation semantics and a trusted loader')

    @property
    def digest(self) -> str:
        version = self.semantic_version.encode('utf-8')
        return hashlib.sha256(len(version).to_bytes(8, 'big') + version + self.payload).hexdigest()


class ProgramCatalog(Protocol):
    """Trusted deployment control plane; no runtime state is persisted.

    resolve verifies version-bound artifact bytes and canonical parameters before
    invoking the preinstalled loader. DeployedPrograms implements this contract.
    """
    def resolve(self, reference: ProgramReference) -> StrategyProgram: ...


@dataclass(frozen=True)
class ProgramCheckpoint:
    """An explicit host checkpoint; no market panels or event histories are stored."""
    context: ResearchContext
    descriptor: ProgramDescriptor
    plan: ProgramPlan
    account: AccountCheckpoint
    state: tuple[StateTensor, ...]
    format: str = 'program-checkpoint-v2'


@dataclass(frozen=True)
class ProgramSimulation:
    """Standalone actual-account audit; admission requires portfolio replay.

    Returns/equity are owned resident (B,V_research) arrays, and measurement_ns
    contains the matching period starts. No complete event or fill trace is
    retained. on_feedback may stream a user-owned audit during the run.
    """
    context: ResearchContext
    descriptor: ProgramDescriptor
    plan: ProgramPlan
    measurement_ns: NDArray[np.int64]
    returns: FloatArray
    equity: FloatArray
    final_account: AccountSnapshot
    checkpoint: ProgramCheckpoint | None = None
