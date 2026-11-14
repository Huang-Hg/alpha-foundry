"""Runtime-independent study, array-backend and data ownership contracts."""
from dataclasses import dataclass
from typing import Callable, Mapping


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
    funding: object
    trading_fees: object
    liquidation_fees: object
    isolated_loss_cap: object
    liquidations: object
    liquidation_events: object
    first_liquidation_ms: object
    bankrupt: object
    ending_equity: object
    max_gross_leverage: object
    model: str


@dataclass(frozen=True)
class SimulationResult:
    """Simulate returns full (B,T); blend returns (B,T-start). Risk is always full-grid."""
    returns: object
    statistics: object
    risk: ExecutionRiskBatch | None


@dataclass(frozen=True)
class ReturnBatch:
    """Books on the full grid, returns on the research window; risk remains full-grid."""
    books: object
    returns: object
    valid: object
    maxdd: object
    risk: ExecutionRiskBatch | None


@dataclass(frozen=True)
class Backend:
    name: str
    xp: object
    impl: Mapping[str, Callable]
    unsupported: Callable
    max_window: int
    max_pair_window: int
    to_host: Callable
    close: Callable


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
    dates: object
    assets: tuple[str, ...]
    values: Mapping[str, object]
    members: object
    contract: DataContract


@dataclass(frozen=True)
class Window:
    dates: object
    start: int

    @property
    def stat_dates(self):
        return self.dates[self.start:]


@dataclass(frozen=True)
class LoadedStudy:
    study: Study
    market: object
    panel: object
    host_spec: object
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
