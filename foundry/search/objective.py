"""Composed discovery objectives; dense rewards never replace hard admission."""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, fields, replace
from typing import cast

from foundry.config import SearchConfig
from foundry.search.archive import (
    Archive, BehaviorIdentity, BehaviorState, ReplayBehavior, TargetBehavior, same_behavior, standardized,
)
from foundry.search.diagnostics import (
    CalibrationPopulation,
    calibrate,
    diagnose,
    mean_correlation,
    measures,
    required_sharpe,
)
from foundry.study.session import ResidentPort, Session
from foundry.types import (
    BoolArray,
    Candidate,
    CandidateAcceptance,
    DiscoveryReport,
    DiscoveryRow,
    EvaluationRow,
    FloatArray,
    PortfolioView,
    LearningFeedback,
    ProposalBatch,
    ContextConflict,
    Proposal,
    ReweightResult,
    SingleGate,
    Verdict,
)


@dataclass(frozen=True)
class Behavior:
    nonzero: BoolArray
    rho: FloatArray
    low_correlation: BoolArray
    novelty: FloatArray


@dataclass(frozen=True)
class ScoringBatch:
    port: ResidentPort
    reference: PortfolioView
    member_returns: FloatArray
    candidates: tuple[Candidate, ...]
    rows: tuple[EvaluationRow, ...]
    verdicts: tuple[Verdict, ...]
    returns: FloatArray
    behavior: Behavior
    rho_max: float


@dataclass(frozen=True)
class PortfolioValues:
    reward: float
    feasible: float
    rho: float
    co_loss: float
    novelty: float
    nonzero: float


@dataclass(frozen=True)
class SingleValues:
    reward: float
    feasible: float
    gate: float
    rho: float
    novelty: float
    nonzero: float
    score: float
    sharpe: float
    cagr: float
    maxdd: float
    rho_portfolio: float
    appraisal: float
    required_sharpe: float
    worth: float
    rho_members: float


def normalized_violation(proposal: Proposal, cfg: SearchConfig) -> float:
    """Normalize raw admission margins for discovery rewards, never for verdicts."""
    if not proposal.valid:
        return float('inf')
    scales = (cfg.violation_scale_wsharpe, cfg.violation_scale_sharpe,
              cfg.violation_scale_cagr, cfg.violation_scale_growth)
    distance = 0.
    for check, scale in zip(proposal.checks[:4], scales):
        if check.required is None:
            return float('inf')
        distance += max(check.required - check.delta, 0.) / scale
    return distance + any(check.name == 'capacity' and not check.passed for check in proposal.checks)


@dataclass(frozen=True)
class Scored:
    rows: tuple[DiscoveryRow, ...]
    rewards: FloatArray


class PortfolioObjective:
    name = 'portfolio'
    gate: SingleGate | None = None

    def score(self, batch: ScoringBatch) -> Scored:
        port, ref, behavior = batch.port, batch.reference, batch.behavior
        xp, returns = port.xp, batch.returns
        nonzero, rho, low_corr, novelty = behavior.nonzero, behavior.rho, behavior.low_correlation, behavior.novelty
        down = ref.rr < 0
        co_loss = xp.sum(xp.maximum(-returns, 0) * down, axis=1) / xp.maximum(xp.sum(xp.abs(returns), axis=1), 1e-12)
        proposals = [verdict.proposal for verdict in batch.verdicts]
        inputs = xp.asarray([[proposal.passed, proposal.utility, normalized_violation(proposal, port.settings.search),
                              port.normalized_complexity(row.source)]
                             for proposal, row in zip(proposals, batch.rows)])
        feasible = inputs[:, 0].astype(xp.bool_) & low_corr
        distance = inputs[:, 2] + xp.maximum(0, rho - batch.rho_max) + (~nonzero)
        reward = xp.where(feasible, 1 + 3 / (1 + xp.exp(-(inputs[:, 1] - .25 * co_loss - .1 * inputs[:, 3] - .25 * novelty))), -6 - xp.minimum(4, distance))
        reports = port.to_host(xp.stack((reward, feasible, rho, co_loss, novelty, nonzero), axis=1)).tolist()
        output = []
        for candidate, verdict, proposal, values in zip(batch.candidates, batch.verdicts, proposals, reports):
            value = PortfolioValues(*values)
            output.append(DiscoveryRow(candidate, value.reward, bool(value.feasible), True,
                proposal.passed, verdict, rho=value.rho, utility=proposal.utility,
                co_loss=value.co_loss, novelty=value.novelty, weight=proposal.weight,
                delta=proposal.checks, backend=proposal.backend))
        return Scored(tuple(output), reward.astype(xp.float32))


def utility(sharpe: FloatArray, maxdd: FloatArray, cagr: FloatArray, port: ResidentPort) -> FloatArray:
    cfg, xp = port.settings.single, port.xp
    u_s = xp.tanh(xp.maximum(sharpe, 0) / cfg.sharpe_scale)
    u_d = xp.maximum(0, 1 - xp.abs(maxdd) / cfg.drawdown_full)
    u_r = xp.tanh(xp.maximum(cagr, 0) / cfg.return_scale)
    return u_s * (cfg.weight_sharpe + cfg.weight_drawdown * u_d + cfg.weight_return * u_r)


@dataclass(frozen=True)
class SingleObjective:
    gate: SingleGate
    name = 'single'

    def score(self, batch: ScoringBatch) -> Scored:
        port, ref, behavior = batch.port, batch.reference, batch.behavior
        xp, cfg, portfolio_cfg = port.xp, port.settings.single, port.settings.portfolio
        returns = batch.returns
        maxdd = xp.stack([xp.asarray(row.maxdd) for row in batch.rows])
        nonzero, rho, low_corr, novelty = behavior.nonzero, behavior.rho, behavior.low_correlation, behavior.novelty
        measured = measures(port, returns, ref)
        sharpe, cagr, years = measured.sharpe, measured.cagr, measured.years
        rho_p, appraisal = measured.rho_portfolio, measured.appraisal
        rho_m = mean_correlation(returns, batch.member_returns, xp)
        required = required_sharpe(rho_p, rho_m, ref, self.gate.required_appraisal, cfg, portfolio_cfg, xp)
        room = ref.n < portfolio_cfg.max_members
        worth = nonzero & (sharpe >= required)
        gate = worth & (years > cfg.recent_year_sharpe).all(1)
        admitted = xp.asarray([verdict.passed for verdict in batch.verdicts])
        feasible = gate & low_corr & room & admitted
        violation = (xp.maximum(required - sharpe, 0) / cfg.violation_scale_sharpe
                     + xp.maximum(cfg.recent_year_sharpe - years, 0).sum(1) / cfg.violation_scale_year
                     + xp.maximum(0, rho - batch.rho_max) + (~nonzero) + (~admitted) + (0 if room else 1))
        score = utility(sharpe, maxdd, cagr, port)
        reward = xp.where(feasible, 1 + 3 * score, -6 - xp.minimum(4, violation))
        host = port.to_host(xp.concatenate((xp.stack((reward, feasible, gate, rho, novelty, nonzero,
                         score, sharpe, cagr, maxdd, rho_p, appraisal, required, worth, rho_m), axis=1), years), axis=1)).tolist()
        output = []
        width = len(fields(SingleValues))
        for candidate, verdict, values in zip(batch.candidates, batch.verdicts, host):
            value = SingleValues(*values[:width])
            proposal = verdict.proposal
            output.append(DiscoveryRow(candidate, value.reward, bool(value.feasible), True,
                bool(value.worth), verdict, rho=value.rho, utility=value.score, novelty=value.novelty,
                weight=proposal.weight, delta=proposal.checks, backend=proposal.backend,
                g3=bool(value.gate), sharpe=value.sharpe, cagr=value.cagr, maxdd=value.maxdd,
                rho_portfolio=value.rho_portfolio, rho_members=value.rho_members, appraisal=value.appraisal,
                required_sharpe=value.required_sharpe, recent_years=tuple(values[width:])))
        return Scored(tuple(output), reward.astype(xp.float32))


class Discovery:
    """One immutable epoch for an in-process Session and its ResidentPort.

    Remote sessions from foundry.connect do not expose resident tensors. Use
    their evaluate, diagnose, judge_family and commit methods instead.
    """
    COUNTERS = ('requested', 'unique_evaluations', 'g1', 'g2', 'low_corr', 'behavior_duplicates', 'weight_trials')

    def __init__(self, session: Session, archive: Archive | None = None,
                 objective: str | PortfolioObjective | SingleObjective = 'portfolio',
                 rho_max: float = .9, calibration: CalibrationPopulation | None = None):
        if not 0 < rho_max <= 1:
            raise ValueError('discovery correlation limit must be in (0,1]')
        self.session, self.port, self.rho_max = session, session.resident, rho_max
        self.archive = Archive(self.port.settings.search.archive_size) if archive is None else archive
        archive = self.archive
        self.ref, self.version = self.port.reference, self.port.context.portfolio
        self.context_id, self.archive_version = self.port.context.id, archive.version
        if objective == 'portfolio':
            self.objective = PortfolioObjective()
        elif objective == 'single':
            if calibration is None:
                raise ValueError('single discovery requires an explicit calibration population')
            self.objective = SingleObjective(calibrate(self.port, calibration))
        elif isinstance(objective, (PortfolioObjective, SingleObjective)):
            self.objective = objective
        else:
            raise ValueError('unknown discovery objective')
        self.gate = self.objective.gate
        self.pending: OrderedDict[Candidate, BehaviorState] = OrderedDict()
        self.cache: dict[Candidate, DiscoveryRow] = {}
        self.rewards: dict[Candidate, FloatArray] = {}
        self.best: Candidate | None = None
        self.best_score = -float('inf')
        self.counters = dict.fromkeys(self.COUNTERS + (('g3',) if self.objective.name == 'single' else ()), 0)
        with self.port.scope():
            _, self.member_returns = self.port.member_returns()
            xp = self.port.xp
            self.reference_vectors, self.reference_valid = standardized(xp.concatenate((self.ref.rr[None], self.member_returns)), xp)
            self.archive_vectors = archive.vectors(self.port)
            self.diagnostics = diagnose(self.port, archive, gate=self.gate)

    def check_context(self) -> None:
        self.port.context
        if self.archive.version != self.archive_version:
            raise RuntimeError('archive changed inside frozen search epoch')

    def context(self) -> list[float]:
        self.check_context()
        return [feature.value for feature in self.diagnostics.features]

    def evaluate(self, candidates: Sequence[Candidate]) -> DiscoveryReport:
        self.check_context()
        identities: dict[str, Candidate] = {}
        for candidate in candidates:
            if candidate.id in identities and identities[candidate.id] != candidate:
                raise ValueError('a discovery candidate ID cannot name conflicting sources in one batch')
            identities[candidate.id] = candidate
        self.counters['requested'] += len(candidates)
        fresh = [candidate for candidate in dict.fromkeys(candidates) if candidate not in self.cache]
        with self.port.scope():
            for lo in range(0, len(fresh), self.port.candidate_batch):
                part = tuple(fresh[lo:lo + self.port.candidate_batch])
                batch = self.port.evaluate(part, report=False, delayed=self.port.settings.search.noise_control)
                self.counters['unique_evaluations'] += len(batch.rows)
                valid = []
                for candidate, row, verdict in zip(part, batch.rows, batch.verdicts):
                    self.cache[candidate] = DiscoveryRow(candidate, -12., False, False, False, verdict, error=row.error)
                    self.rewards[candidate] = self.port.xp.asarray(-12., dtype=self.port.xp.float32)
                    if verdict.stopped_at != 'G1':
                        valid.append((candidate, row, verdict))
                if valid:
                    self._score(tuple(valid))
        return DiscoveryReport(self.port.context, self.objective.name, tuple(self.cache[candidate] for candidate in candidates),
                               self.port.settings.search.noise_control)

    def feedback(self, batch: ProposalBatch) -> LearningFeedback:
        """Seal ordered device feedback after evaluation, before the single writer commits."""
        self.check_context()
        if batch.context != self.port.context:
            raise ContextConflict('proposal batch belongs to a different research context')
        if any(candidate not in self.cache for candidate in batch.candidates):
            raise ValueError('evaluate every proposal before requesting feedback')
        with self.port.scope():
            values = self.port.xp.stack([self.rewards[candidate] for candidate in batch.candidates])
            return LearningFeedback(batch, self.port.feedback_vector(values))

    def _behavior(self, returns: FloatArray) -> Behavior:
        xp = self.port.xp
        vectors, nonzero = standardized(returns, xp)
        correlations = xp.clip(vectors @ self.reference_vectors.T, -1, 1)
        rho = xp.maximum(0, xp.where(self.reference_valid[None], correlations, -xp.inf).max(1))
        low_corr = nonzero & xp.isfinite(rho) & (rho <= self.rho_max)
        novelty = xp.maximum(0, vectors @ self.archive_vectors.T).max(1) if len(self.archive_vectors) else xp.zeros(len(returns))
        return Behavior(nonzero, rho, low_corr, novelty)

    def _score(self, valid: tuple[tuple[Candidate, EvaluationRow, Verdict], ...]) -> None:
        xp = self.port.xp
        candidates, rows, verdicts = zip(*valid)
        returns = xp.stack([row.returns for row in rows])
        targets = [(i, row.book) for i, row in enumerate(rows) if row.book is not None]
        identities: dict[int, BehaviorIdentity] = {
            i: ReplayBehavior(cast(str, row.behavior_digest)) for i, row in enumerate(rows) if row.book is None}
        if targets:
            books = xp.stack([book for _, book in targets])
            identities.update((i, TargetBehavior(fingerprint, books[j]))
                              for j, ((i, _), fingerprint) in enumerate(zip(targets, self.port.fingerprints(books))))
        behavior = self._behavior(returns)
        scored = self.objective.score(ScoringBatch(self.port, self.ref, self.member_returns,
                                                  candidates, rows, verdicts, returns, behavior, self.rho_max))
        nonzero = self.port.to_host(behavior.nonzero).tolist()
        staged: OrderedDict[Candidate, int] = OrderedDict()
        for i, (report, has_returns) in enumerate(zip(scored.rows, nonzero)):
            duplicate = self.archive.duplicate(identities[i], self.version, self.port)
            accepted = report.passed and not duplicate
            result = replace(report, log_reward=-10. if duplicate else report.log_reward,
                             passed=accepted, duplicate=duplicate)
            self.cache[report.candidate] = result
            self.rewards[report.candidate] = xp.asarray(-10., dtype=xp.float32) if duplicate else scored.rewards[i]
            self.counters['g1'] += 1
            self.counters['g2'] += int(result.g2)
            if self.objective.name == 'single':
                self.counters['g3'] += int(result.g3)
            self.counters['low_corr'] += int(accepted)
            self.counters['behavior_duplicates'] += int(duplicate)
            self.counters['weight_trials'] += result.decision.proposal.trials
            if accepted and result.utility > self.best_score:
                self.best, self.best_score = result.candidate, result.utility
            if not duplicate and has_returns:
                staged[result.candidate] = i
                staged.move_to_end(result.candidate)
        for i in staged.values():
            candidate = candidates[i]
            identity = identities[i]
            for previous in tuple(key for key, state in self.pending.items()
                                  if same_behavior(identity, state.behavior, xp)):
                del self.pending[previous]
            if self.archive.capacity:
                owned = (TargetBehavior(identity.fingerprint, identity.book.copy())
                         if isinstance(identity, TargetBehavior) else identity)
                self.pending[candidate] = BehaviorState(candidate, owned, returns[i].copy(),
                                                        self.version, self.port.context.snapshot.id)
                self.pending.move_to_end(candidate)
                while len(self.pending) > self.archive.capacity:
                    self.pending.popitem(last=False)

    def accept(self) -> CandidateAcceptance | None:
        self.check_context()
        if self.best is None or self.port.settings.search.noise_control:
            return None
        reviewed = DiscoveryReport(self.port.context, self.objective.name, (self.cache[self.best],))
        result = self.session.commit(reviewed)
        row = result.rows[0]
        if not row.accepted:
            raise RuntimeError('selected candidate no longer passes admission in the frozen context')
        return CandidateAcceptance(self.best, row.weight, row.decision.proposal.utility)

    def reweight(self, iterations: int = 3, count: int = 32, seed: int = 1) -> ReweightResult:
        self.check_context()
        return self.session.reweight(iterations=iterations, count=count, seed=seed)
