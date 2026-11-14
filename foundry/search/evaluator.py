"""Search rewards and proposals; research owns evaluation, admission and commits."""
from collections import OrderedDict
from dataclasses import dataclass, replace
from foundry.candidates import Formula

from foundry.research_types import Candidate, ConstraintCheck, PrecisionDiagnostics
from foundry.search.policy import CEM


@dataclass(frozen=True)
class BehaviorState:
    book: object
    returns: object
    expr: str
    portfolio_version: str


@dataclass(frozen=True)
class MiningResult:
    expr: str
    log_reward: float
    passed: bool
    g1: bool
    g2: bool
    rho: float | None = None
    utility: float | None = None
    error: str | None = None
    co_loss: float = 0.
    novelty: float = 0.
    duplicate: bool = False
    weight: float = 0.
    delta: tuple[ConstraintCheck, ...] = ()
    precision: PrecisionDiagnostics | None = None
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


def standardized(r, xp):
    centered = r - r.mean(axis=1, keepdims=True)
    norm = xp.sqrt(xp.sum(centered * centered, axis=1, keepdims=True))
    valid = xp.isfinite(r).all(1) & (norm[:, 0] > 1e-8)
    return xp.where(valid[:, None], centered / xp.maximum(norm, 1e-8), 0), valid


class Archive:
    def __init__(self, capacity=32):
        self.capacity, self.version = capacity, 0
        self.items = OrderedDict()
        self.populations = {}

    def update(self, staged):
        for key, value in staged.items():
            self.items[key] = value
            self.items.move_to_end(key)
        while len(self.items) > self.capacity:
            self.items.popitem(last=False)
        if staged:
            self.version += 1


class Evaluator:
    gate = None
    COUNTERS = ('requested', 'unique_evaluations', 'g1', 'g2', 'low_corr', 'behavior_duplicates', 'fp64_refined', 'weight_trials')

    def __init__(self, research, archive, rho_max=.9):
        self.research, self.portfolio = research, research.portfolio
        self.engine, self.archive, self.rho_max = research.engine, archive, rho_max
        self.ref, self.version = self.portfolio.reference, self.portfolio.version
        self.context_id = research.context.id
        self.archive_version = archive.version
        self.pending, self.cache = OrderedDict(), {}
        self.best = None
        self.best_score = -float('inf')
        self.counters = dict.fromkeys(self.COUNTERS, 0)
        self.weights, self.books, self.member_returns = research.member_states(self.context_id)
        xp = self.engine.xp
        self.reference_vectors, self.reference_valid = standardized(xp.concatenate((self.ref.rr[None], self.member_returns)), xp)
        self.archive_vectors = standardized(xp.stack([v.returns for v in archive.items.values()]), xp)[0] if archive.items else xp.empty((0, len(self.ref.rr)))

    def check_context(self):
        self.research.require_context(self.context_id)
        if self.archive.version != self.archive_version:
            raise RuntimeError('archive changed inside frozen search epoch')

    def context(self):
        xp = self.engine.xp
        r, weights = self.ref.rr, xp.asarray(self.weights)
        block = xp.minimum(xp.arange(len(r)) * 6 // len(r), 5)
        means = xp.bincount(block, weights=r, minlength=6) / xp.maximum(xp.bincount(block, minlength=6), 1)
        sig = xp.sqrt(xp.mean(r * r))
        member_vol = self.member_returns.std(1)
        values = xp.concatenate((xp.asarray([self.ref.wsharpe / 3, self.ref.sharpe / 3, self.ref.recent_growth, self.ref.cagr]),
            xp.stack((xp.sum(weights * weights), xp.sum(weights * member_vol) * self.engine.ppy ** .5,
                      xp.mean(xp.sum(self.ref.W, axis=1)), xp.asarray(len(self.weights) / 40))),
            means / xp.maximum(sig, 1e-8),
            xp.stack((xp.asarray(len(self.archive.items) / max(1, self.archive.capacity)),
                      xp.mean(self.archive_vectors @ self.reference_vectors[0]) if len(self.archive_vectors) else xp.asarray(0.)))))
        return self.engine.to_host(xp.tanh(xp.nan_to_num(values))).tolist()

    def evaluate(self, expressions):
        self.check_context()
        self.counters['requested'] += len(expressions)
        fresh = [x for x in dict.fromkeys(expressions) if x not in self.cache]
        for lo in range(0, len(fresh), self.engine.candidate_batch):
            sources = tuple(Formula(expression) for expression in fresh[lo:lo + self.engine.candidate_batch])
            batch = self.research.evaluate_device(sources, self.context_id)
            self.counters['unique_evaluations'] += len(batch.rows)
            for row in batch.rows:
                self.cache[row.source.expression] = MiningResult(row.source.expression, -12., False, False, False, error=row.error)
            valid = [(r, v) for r, v in zip(batch.rows, batch.verdicts) if v.stopped_at != 'G1']
            if valid:
                self._valid_batch(valid)
        return [self.cache[x] for x in expressions]

    def _behavior(self, returns):
        xp = self.engine.xp
        vectors, nonzero = standardized(returns, xp)
        correlations = xp.clip(vectors @ self.reference_vectors.T, -1, 1)
        rho = xp.maximum(0, xp.where(self.reference_valid[None], correlations, -xp.inf).max(1))
        low_corr = nonzero & xp.isfinite(rho) & (rho <= self.rho_max)
        novelty = xp.maximum(0, vectors @ self.archive_vectors.T).max(1) if len(self.archive_vectors) else xp.zeros(len(returns))
        return vectors, nonzero, rho, low_corr, novelty

    def _duplicate(self, key, book):
        if key not in self.archive.items:
            return False
        old = self.archive.items[key]
        return old.portfolio_version == self.version and bool(self.engine.xp.array_equal(book, old.book))

    def _stage(self, staged, books, returns, rows):
        for key, i in list(staged.items())[-self.archive.capacity:]:
            self.pending[key] = BehaviorState(books[i].copy(), returns[i].copy(), rows[i].source.expression, self.version)
            self.pending.move_to_end(key)
            while len(self.pending) > self.archive.capacity:
                self.pending.popitem(last=False)

    def _valid_batch(self, pairs):
        xp = self.engine.xp
        rows, verdicts = zip(*pairs)
        books, returns = xp.stack([r.book for r in rows]), xp.stack([r.returns for r in rows])
        _, nonzero, rho, low_corr, novelty = self._behavior(returns)
        down = self.ref.rr < 0
        co_loss = xp.sum(xp.maximum(-returns, 0) * down, axis=1) / xp.maximum(xp.sum(xp.abs(returns), axis=1), 1e-12)
        proposals = [v.proposal for v in verdicts]
        inputs = xp.asarray([[p.passed, p.utility, p.violation, len(r.source.expression) / 512] for p, r in zip(proposals, rows)])
        feasible = inputs[:, 0].astype(xp.bool_) & low_corr
        distance = inputs[:, 2] + xp.maximum(0, rho - self.rho_max) + (~nonzero)
        reward = xp.where(feasible, 1 + 3 / (1 + xp.exp(-(inputs[:, 1] - .25 * co_loss - .1 * inputs[:, 3] - .25 * novelty))), -6 - xp.minimum(4, distance))
        reports = self.engine.to_host(xp.stack((reward, feasible, rho, co_loss, novelty, nonzero), axis=1)).tolist()
        staged = OrderedDict()
        for i, (row, verdict, proposal, values, key) in enumerate(zip(rows, verdicts, proposals, reports, self.engine.fingerprints(books))):
            duplicate = self._duplicate(key, books[i])
            accepted = bool(values[1]) and not duplicate
            self.cache[row.source.expression] = MiningResult(row.source.expression, values[0] if not duplicate else -10., accepted, True,
                proposal.passed, values[2], proposal.utility, co_loss=values[3], novelty=values[4], duplicate=duplicate,
                weight=proposal.weight, delta=proposal.checks, precision=proposal.precision, backend=proposal.backend)
            self.counters['g1'] += 1
            self.counters['g2'] += int(proposal.passed)
            self.counters['low_corr'] += int(accepted)
            self.counters['behavior_duplicates'] += int(duplicate)
            self.counters['weight_trials'] += proposal.trials
            self.counters['fp64_refined'] += int(proposal.backend.endswith('-f64'))
            if accepted and proposal.utility > self.best_score:
                self.best = row.source.expression
                self.best_score = proposal.utility
            if not duplicate and values[5]:
                staged[key] = i
                staged.move_to_end(key)
        self._stage(staged, books, returns, rows)

    def accept(self):
        self.check_context()
        if self.best is None:
            return None
        result = self.research.commit((Candidate(self.best, Formula(self.best)),), self.context_id)
        row = result.rows[0]
        assert row.accepted
        return CandidateAcceptance(row.row.source.expression, row.weight, row.verdict.proposal.utility)

    def reweight(self, iterations=3, count=32, seed=1):
        self.check_context()
        if len(self.portfolio.library.members) < 2:
            return None
        xp = self.engine.xp
        books = xp.stack(self.books)
        base = xp.asarray(self.weights, dtype=xp.float64)
        inside = xp.clip(base, 1e-6, 1 - 1e-6)
        sampler = CEM(self.engine.to_host(xp.log(inside / (1 - inside))))
        best = None
        evaluations = 0
        for iteration in range(iterations):
            weights = xp.asarray(sampler.sample(count, seed + iteration))
            deletes = xp.broadcast_to(base, (len(base), len(base))).copy()
            deletes[xp.arange(len(base)), xp.arange(len(base))] = 0
            trials = xp.concatenate((weights, base[None], deletes))
            score_parts = []
            for lo in range(0, len(trials), self.engine.candidate_batch):
                part = trials[lo:lo + self.engine.candidate_batch]
                trial = self.research.trial_weights(part, books, self.context_id)
                feasible, utility = trial.feasible, trial.utility
                scores = xp.where(feasible, utility, -1e6)
                score_parts.append(scores)
                at = int(xp.argmax(scores))
                if bool(feasible[at]) and (best is None or float(scores[at]) > best[0]):
                    best = (float(scores[at]), part[at].copy())
                evaluations += len(part)
            scores = xp.concatenate(score_parts)
            nfeasible = int((scores[:count] > -1e5).sum())
            if nfeasible:
                sampler.update(self.engine.to_host(scores[:count]), min(nfeasible, max(1, count // 4)))
        if best is None:
            return ReweightResult(False, evaluations)
        values = self.engine.to_host(best[1]).tolist()
        committed = self.research.commit_weights(values, self.context_id)
        return ReweightResult(committed.changed, evaluations, committed.utility, tuple(values))


@dataclass(frozen=True)
class CandidateAcceptance:
    expr: str
    weight: float
    utility: float


@dataclass(frozen=True)
class ReweightResult:
    changed: bool
    evaluations: int
    utility: float | None = None
    weights: tuple[float, ...] = ()
