"""Pure value contracts; device arrays remain borrowed on their owning runtime.

Array axes are batch B, decision day T, execution event E and asset S. Host
arrays use NumPy; device arrays expose the CUDA array interface without loading
a device library. Field documentation defines dtype, units and ownership.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
import hashlib
from types import ModuleType
from typing import Callable, Generic, Literal, Mapping, Protocol, Sequence, TypeAlias, TypeVar

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
    Remaining arrays are (B,); UTC timestamps are int64 milliseconds, with -1
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
    first_liquidation_ms: IntArray
    bankrupt: BoolArray
    ending_equity: FloatArray
    max_gross_leverage: FloatArray
    model: str


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
class Window:
    dates: NDArray[np.datetime64]
    start: int

    @property
    def stat_dates(self):
        return self.dates[self.start:]


@dataclass(frozen=True)
class LoadedStudy(Generic[MarketValue, PanelValue, SpecValue]):
    study: Study
    market: MarketValue
    panel: PanelValue
    host_spec: SpecValue
    window: Window
    data_contract: DataContract


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
    kind: Literal['signals', 'target_weights', 'target_exposure']
    digest: str
    snapshot: str
    producer: str
    complexity: float = 0.

    def normalized_complexity(self, ast_cost: int, max_cost: int) -> float:
        return self.complexity


@dataclass(frozen=True)
class EqualWeight:
    kind: Literal['equal_weight'] = 'equal_weight'

    def normalized_complexity(self, ast_cost: int, max_cost: int) -> float:
        return 0.


Source = Formula | ArraySource | EqualWeight


@dataclass(frozen=True)
class ArrayBatch:
    """Explicit external-input boundary; registration takes an owned copy.

    values: NumPy/CuPy float32/float64 (B,T,S), on the engine's device or host.
    signals: NaN denotes missing; cash uses clip(0,1)/eligible_count;
    short-enabled markets use clip(-1,1)/eligible_count (at most 1x gross).
    target_weights: finite long-only fractions, zero for ineligible assets,
    each day's sum <= 1 (the balance is cash); excess within the configured
    portfolio.weight_sum_atol is normalized to 1. No leverage or shorts implied.
    target_exposure: signed notional/equity ratios, zero for ineligible assets;
    sum(abs(exposure)) <= the market allocation budget. Cash allocation still
    forbids shorts. Derivatives require this kind instead of target_weights.
    observed_through / fit_through: integer (B,T), from -1 through row t.
    producer: immutable model/program/version identifier supplied by the caller.
    dates: ISO trading dates; assets: identifiers in the exact study order.
    All arrays are borrowed until register_batch returns; callers may then
    mutate their buffers without changing the registered candidate.
    CUDA inputs must be ready on the runtime's current stream when submitted;
    producers using other streams establish that dependency before calling.
    complexity: optional B normalized costs in [0,1]; omission means zero.
    """
    kind: Literal['signals', 'target_weights', 'target_exposure']
    ids: tuple[str, ...]
    values: FloatArray
    snapshot: str
    producer: str
    dates: tuple[str, ...]
    assets: tuple[str, ...]
    observed_through: IntArray
    fit_through: IntArray
    complexity: tuple[float, ...] | None = None


class ArrayStore(Protocol):
    """Immutable, content-addressed artifacts; writes only on registration."""
    def put(self, batch: ArrayBatch) -> tuple[ArraySource, ...]: ...
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
    days: int
    sharpe: float
    attend: float
    lg: float


@dataclass(frozen=True)
class Metrics:
    """Common research metrics; turnover/exposure/max_single_weight use targets.

    Turnover is mean absolute change of the daily target book, not executed
    notional. margin_bps uses this same target turnover denominator. Derivative
    realized fees and actual account leverage are reported separately in risk.
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
    n_days: int


@dataclass(frozen=True)
class RiskSummary:
    """Window cashflows/counts; account state and first event use the full run.

    first_liquidation_ms identifies the hourly event bar, not an exact tick fill.
    """
    model: str
    funding: float
    trading_fees: float
    liquidation_fees: float
    isolated_loss_cap: float
    liquidations: int
    first_liquidation_ms: int
    bankrupt: bool
    ending_equity: float
    max_gross_leverage: float


@dataclass(frozen=True)
class StatsRow:
    book: Array
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


@dataclass(frozen=True)
class ConstraintCheck:
    name: str
    passed: bool
    delta: float


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
    violation: float = float('inf')


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


@dataclass(frozen=True)
class CommitReport:
    before: ResearchContext
    context: ResearchContext
    rows: tuple[CandidateReport, ...]


@dataclass(frozen=True)
class RegisteredBatch:
    context: ResearchContext
    candidates: tuple[Candidate, ...]


@dataclass(frozen=True)
class DataGrid:
    """Producer-visible coordinates, bound to one immutable data snapshot."""
    snapshot: str
    dates: tuple[str, ...]
    assets: tuple[str, ...]


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
    def evaluate(self, candidates: tuple[Candidate, ...], context: str) -> EvaluationReport: ...
    def evaluate_device(self, candidates: tuple[Candidate, ...], context: str) -> DeviceBatch: ...
    def commit(self, candidates: tuple[Candidate, ...], context: str) -> CommitReport: ...
    def register(self, batch: ArrayBatch, context: str) -> RegisteredBatch: ...
    def diagnose(self, context: str) -> Diagnostics: ...
    def judge_family(self, families: Sequence[FamilySeed | CandidateGrid], context: str) -> FamilyResult: ...
    def portfolio(self, context: str) -> PortfolioReport: ...
    def trial_weights(self, weights: Array, context: str) -> WeightReport: ...
    def commit_weights(self, weights: tuple[float, ...], context: str) -> WeightCommit: ...
    def reweight(self, context: str, iterations: int, count: int, seed: int) -> ReweightResult: ...
    def resident(self) -> ResidentValue: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class ExecutionSpec:
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
        for field in fields(self):
            value = getattr(self, field.name)
            if value is not None:
                owned = np.array(value, order='C', copy=True)
                owned.flags.writeable = False
                object.__setattr__(self, field.name, owned)


    @property
    def snapshot(self):
        digest=hashlib.sha256()
        for field in fields(self):
            value=getattr(self,field.name)
            digest.update(field.name.encode())
            if value is not None:
                digest.update(str((value.shape, value.dtype.str)).encode())
                digest.update(value.tobytes())
        return digest.hexdigest()

    @property
    def symmetric(self) -> bool:
        return self.cost_sell is None and self.block is None and self.carry is None

    def rows(self, end: int) -> 'ExecutionSpec':
        return replace(self, **{f.name: getattr(self, f.name)[:end] for f in fields(self) if getattr(self, f.name) is not None})


PERPETUAL_MODEL = 'linear_usdt_isolated_daily_collateral_hourly_mark_touch_v1'
PERPETUAL_EVENT_FIELDS = ('event_times_ms', 'day_index', 'rebalance')
PERPETUAL_PANEL_FIELDS = ('trade_open', 'mark_open', 'mark_high', 'mark_low', 'mark_close',
                'funding_rate', 'funding_mark', 'maintenance_rate', 'fee_rate', 'liquidation_fee')
PERPETUAL_ARRAY_FIELDS = PERPETUAL_EVENT_FIELDS + PERPETUAL_PANEL_FIELDS

@dataclass(frozen=True)
class PerpetualSpec:
    """Owned read-only host arrays for T daily targets and E hourly events.

    event_times_ms/day_index are int64 (E,), UTC bar-start milliseconds and
    indices in [0, days); rebalance is bool (E,). Price and rate arrays are
    float64 (E,S): prices use USDT per asset unit, funding_rate is the signed
    settlement fraction, and maintenance/fee/liquidation_fee are fractions of
    mark/traded notional. The loader validates a complete, ordered event grid.
    Intrabar first-liquidation timestamps identify the bar, not a known tick.
    """
    event_times_ms: HostArray
    day_index: HostArray
    rebalance: HostArray
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
    days: int

    def __post_init__(self):
        for name in PERPETUAL_ARRAY_FIELDS:
            dtype = np.int64 if name in PERPETUAL_EVENT_FIELDS[:2] else np.bool_ if name == 'rebalance' else np.float64
            owned = np.array(getattr(self, name), dtype=dtype, order='C', copy=True)
            owned.flags.writeable = False
            object.__setattr__(self, name, owned)
        ratio = self.trade_open / self.mark_open
        if not np.all(self.leverage * (np.abs(1 - ratio).max(1)
                                      + (self.fee_rate * ratio).max(1)) < 1):
            raise ValueError('non-monotone rebalance root')

    @property
    def shape(self):
        return self.days, self.trade_open.shape[1]

    @property
    def snapshot(self):
        digest = hashlib.sha256(PERPETUAL_MODEL.encode())
        for definition in fields(self):
            value = getattr(self, definition.name)
            digest.update(definition.name.encode())
            if definition.name in PERPETUAL_ARRAY_FIELDS:
                digest.update(str((value.shape, value.dtype.str)).encode())
                digest.update(value.tobytes())
            else:
                digest.update(str(value).encode())
        return digest.hexdigest()

    def rows(self, end):
        stop = int(np.searchsorted(self.day_index, end))
        return replace(self, **{name: getattr(self, name)[:stop] for name in PERPETUAL_ARRAY_FIELDS}, days=end)


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
    U: FloatArray
    W: FloatArray
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
class ProposalBatch:
    epoch: int
    ordinary: int
    probe_count: int
    candidates: tuple[Candidate, ...]


@dataclass(frozen=True)
class TrainingResult:
    loss_before: float | None = None
    loss_after: float | None = None
    teachers: int = 0
    logz: float | None = None


class Proposer(Protocol):
    name: str

    def sample(self, count: int, probes: int, context: Sequence[float]) -> ProposalBatch | None: ...
    def train(self, batch: ProposalBatch, rewards: Sequence[float], *, steps: int,
              learning_rate: float, opd_strength: float) -> TrainingResult: ...
    def dump(self) -> bytes: ...
    def load(self, payload: bytes) -> None: ...


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
    """Fit on [fit_start, fit_through], predict on [predict_start, predict_stop)."""
    fit_start: int
    fit_through: int
    predict_start: int
    predict_stop: int


@dataclass(frozen=True)
class FoldView:
    """Owned feature prefix (T,S); host arrays are read-only.

    start/stop are absolute study row indices with an exclusive stop. Arrays
    retain their host/device dtype and cannot expose later rows through a base.
    """
    start: int
    stop: int
    dates: tuple[str, ...]
    assets: tuple[str, ...]
    features: Mapping[str, FloatArray]
    eligibility: BoolArray


@dataclass(frozen=True)
class FoldResult:
    """Realized next-bar net-return signs in [execution_start, execution_stop).

    Mean returns/signs follow registered candidate order. Invalid returns have
    a missing mean and sign zero; valid zero means also have sign zero.
    """
    fold: Fold
    execution_start: int
    execution_stop: int
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
