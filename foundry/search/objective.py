"""Composed discovery objectives; dense rewards never replace hard admission."""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, fields, replace

from foundry.search.archive import Archive, BehaviorState, standardized
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


class PortfolioObjective:
    name = 'portfolio'
    gate: SingleGate | None = None

    def score(self, batch: ScoringBatch) -> tuple[DiscoveryRow, ...]:
        port, ref, behavior = batch.port, batch.reference, batch.behavior
        xp, returns = port.xp, batch.returns
        nonzero, rho, low_corr, novelty = behavior.nonzero, behavior.rho, behavior.low_correlation, behavior.novelty
        down = ref.rr < 0
        co_loss = xp.sum(xp.maximum(-returns, 0) * down, axis=1) / xp.maximum(xp.sum(xp.abs(returns), axis=1), 1e-12)
        proposals = [verdict.proposal for verdict in batch.verdicts]
        inputs = xp.asarray([[proposal.passed, proposal.utility, proposal.violation,
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
        return tuple(output)


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

    def score(self, batch: ScoringBatch) -> tuple[DiscoveryRow, ...]:
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
        return tuple(output)


class Discovery:
    """One immutable portfolio/archive epoch, shared by formulas and registered arrays."""
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
        self.best: Candidate | None = None
        self.best_score = -float('inf')
        self.counters = dict.fromkeys(self.COUNTERS + (('g3',) if self.objective.name == 'single' else ()), 0)
        with self.port.scope():
            self.weights, self.books, self.member_returns = self.port.member_states()
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
                    if verdict.stopped_at != 'G1':
                        valid.append((candidate, row, verdict))
                if valid:
                    self._score(tuple(valid))
        return DiscoveryReport(self.port.context, self.objective.name, tuple(self.cache[candidate] for candidate in candidates))

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
        books, returns = xp.stack([row.book for row in rows]), xp.stack([row.returns for row in rows])
        behavior = self._behavior(returns)
        scored = self.objective.score(ScoringBatch(self.port, self.ref, self.member_returns,
                                                  candidates, rows, verdicts, returns, behavior, self.rho_max))
        nonzero = self.port.to_host(behavior.nonzero).tolist()
        staged: OrderedDict[Candidate, tuple[int, int]] = OrderedDict()
        for i, (report, fingerprint, has_returns) in enumerate(zip(scored, self.port.fingerprints(books), nonzero)):
            duplicate = self.archive.duplicate(fingerprint, books[i], self.version, self.port)
            accepted = report.passed and not duplicate
            result = replace(report, log_reward=-10. if duplicate else report.log_reward,
                             passed=accepted, duplicate=duplicate)
            self.cache[report.candidate] = result
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
                staged[result.candidate] = (fingerprint, i)
                staged.move_to_end(result.candidate)
        for fingerprint, i in staged.values():
            candidate = candidates[i]
            for previous in tuple(key for key, state in self.pending.items()
                                  if state.fingerprint == fingerprint and bool(xp.array_equal(books[i], state.book))):
                del self.pending[previous]
            if self.archive.capacity:
                self.pending[candidate] = BehaviorState(candidate, fingerprint, books[i].copy(), returns[i].copy(),
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
