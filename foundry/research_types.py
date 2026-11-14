"""Typed research contracts. Device buffers stay opaque and are owned by their runtime."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from foundry.contracts import StudySnapshot
from foundry.candidates import Source


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
    book: object
    returns: object
    metrics: Metrics
    valid_returns: bool
    risk: RiskSummary | None = None


@dataclass
class EvaluationRow:
    source: Source
    ok: bool
    error: str | None
    precision: str
    tree: object = None
    book: object = None
    returns: object = None
    valid_returns: bool = False
    maxdd: object = None
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


class LibraryStore(Protocol):
    def read(self, study): ...
    def commit(self, expected, replacement) -> None: ...


class ContextConflict(ValueError):
    """The supplied study or portfolio version no longer names the current state."""


@dataclass(frozen=True)
class WeightTrials:
    feasible: object
    utility: object
    delta: object


@dataclass(frozen=True)
class WeightMetrics:
    valid: object
    wsharpe: object
    sharpe: object
    cagr: object
    recent_growth: object
    maxdd: object
    delta: object
    required: object
    checks: object
    utility: object

    @property
    def feasible(self):
        return self.valid & self.checks.all(-1)


@dataclass(frozen=True)
class WeightCommit:
    before: ResearchContext
    context: ResearchContext
    changed: bool
    utility: float
