"""Immutable portfolio books and moments assembled from evaluated tensors."""
from dataclasses import dataclass
from types import ModuleType

from foundry.config import PortfolioConfig
from foundry.portfolio.library import Member
from foundry.types import Backend, FloatArray, Metrics, ReturnMeasure, Source, StatsRow


@dataclass(frozen=True)
class Reference:
    """Resident (T,S) books and (T,) returns share the supplied backend's device."""

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
    xp: ModuleType
    cfg: PortfolioConfig
    measure: ReturnMeasure
    version: str
    mean_exposure: float

    @staticmethod
    def build(U: FloatArray | None, measured: StatsRow, p: FloatArray, ppy: float,
              members: tuple[Member, ...], seed: bool, backend: Backend,
              cfg: PortfolioConfig, version: str, *, mean_exposure: float) -> 'Reference':
        """Derive reference moments without owning evaluation or mutable state."""
        xp = backend.xp
        moments = backend.measure(measured.returns, p, ppy)
        ws, growth = backend.to_host(xp.stack((moments.weighted_sharpe, moments.recent_growth))).tolist()
        return Reference(U, measured.book, measured.returns, p, measured.metrics,
                         measured.metrics.sharpe, measured.metrics.cagr, measured.metrics.maxdd,
                         ws, growth, len(members), seed, tuple(m.source for m in members),
                         xp, cfg, backend.measure, version, mean_exposure)
