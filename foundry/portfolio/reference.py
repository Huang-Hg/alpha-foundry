"""The portfolio's additive book and its reference metrics."""
from dataclasses import dataclass, field
from foundry.core import metrics as M
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
    version: str = ''
    precision_errors: dict = field(default_factory=dict)
    force_f64: bool = False

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
            p = M.recency(engine.dates[engine.start:], cfg.recent_halflife_days, xp)
            ws, growth = engine.to_host(xp.stack((M.weighted_sharpe(st.returns, p, engine.ppy, xp), M.recent_growth(st.returns, p, engine.ppy, xp)))).tolist()
            return Reference(U, st.book, st.returns, p, st.metrics, st.metrics.sharpe, st.metrics.cagr,
                             st.metrics.maxdd, ws, growth, len(books), seed,
                             frozenset(m.source for m in members), xp, cfg)
