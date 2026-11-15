"""Pure value contracts; device arrays remain borrowed on their owning runtime.

Array axes are batch B, decision day T, execution event E and asset S. Host
arrays use NumPy; device arrays expose the CUDA array interface without loading
a device library. Field documentation defines dtype, units and ownership.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import Callable, Generic, Literal, Mapping, Protocol, TypeAlias, TypeVar

import numpy as np
from numpy.typing import NDArray

MarketValue = TypeVar('MarketValue')
PanelValue = TypeVar('PanelValue')
SpecValue = TypeVar('SpecValue')
LibraryValue = TypeVar('LibraryValue')


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
    null_shift: bool


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
    expr: str
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
    expression: str
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


class SessionBackend(Protocol):
    def context(self) -> ResearchContext: ...
    def refresh(self) -> ResearchContext: ...
    def grid(self) -> DataGrid: ...
    def describe(self) -> dict: ...
    def evaluate(self, candidates: tuple[Candidate, ...], context: str) -> EvaluationReport: ...
    def evaluate_device(self, candidates: tuple[Candidate, ...], context: str) -> DeviceBatch: ...
    def commit(self, candidates: tuple[Candidate, ...], context: str) -> CommitReport: ...
    def register(self, batch: ArrayBatch, context: str) -> RegisteredBatch: ...
    def explore(self, seeds: tuple[FamilySeed, ...], context: str) -> FamilyResult: ...
    def portfolio(self, context: str) -> PortfolioReport: ...
    def trial_weights(self, weights: Array, context: str) -> WeightReport: ...
    def commit_weights(self, weights: tuple[float, ...], context: str) -> WeightCommit: ...
    def close(self) -> None: ...
