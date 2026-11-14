"""The portfolio's additive book and its reference metrics."""
from dataclasses import dataclass
from foundry.contracts import ReturnMeasure
from foundry.portfolio.library import SEED
from foundry.portfolio.allocation import cap_book


@dataclass
class Reference:
    U: object
    W: object
    rr: object
    p: object
    metrics: object
    sharpe: float
    cagr: float
    maxdd: float
    wsharpe: float
    recent_growth: float
    n: int
    seed: bool
    keys: frozenset
    xp: object
    cfg: object
    measure: ReturnMeasure
    version: str = ''

    @staticmethod
    def build(engine, members, null_shift=False):
        with engine.resources.scope():
            xp, cfg = engine.xp, engine.portfolio_cfg
            seed = not members
            members = members or (SEED,)
            books = [book for book, _ in engine.members([m.source for m in members], null_shift)]
            U = xp.sum(xp.stack(books) * xp.asarray([m.weight for m in members])[:, None, None], axis=0)
            U[:engine.start] = 0
            st = engine.metrics([cap_book(U, engine.allocation, xp)])[0]
            p = engine.recency_weights
            moments = engine.backend.measure(st.returns, p, engine.ppy)
            ws, growth = engine.to_host(xp.stack((moments.weighted_sharpe, moments.recent_growth))).tolist()
            return Reference(U, st.book, st.returns, p, st.metrics, st.metrics.sharpe, st.metrics.cagr,
                             st.metrics.maxdd, ws, growth, len(books), seed,
                             frozenset(m.source for m in members), xp, cfg, engine.backend.measure)
