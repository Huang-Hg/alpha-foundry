"""Typed in-process research, atomic portfolio commands and the public session boundary."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from foundry.types import ArraySource
from foundry.portfolio import admission
from foundry.portfolio.reweight import optimize
from foundry.types import ReweightResult
from foundry.portfolio.library import Library, Member, SEED
from foundry.portfolio.portfolio import Portfolio
from foundry.numeric.allocation import cap_book
from foundry.study.feedback import FeedbackVector
from foundry.types import (
    ComputeDevice,Candidate, CandidateResult, CommitResult, ContextConflict, DeviceBatch,
                                    EvaluationResult, PortfolioReport, ResearchContext, WeightCommit,
                                    WeightReport, WeightTrials)
from contextlib import ExitStack
from pathlib import Path
from typing import Callable, TypeVar, TYPE_CHECKING, Literal
from contextlib import AbstractContextManager
from types import ModuleType
from foundry.ops.parse import validate
from foundry.types import ReturnMoments, Array, HostArray, PortfolioView, ResourceStatus, BoolArray, IntArray, EqualWeight
from types import MappingProxyType
from foundry.config import Settings, path
from foundry.portfolio.reference import Reference
from foundry.study.store import FileArrayStore, FileLibraryStore, MemoryLibraryStore
from foundry.types import CandidateReport, CommitReport, Source, FloatArray, LibraryStore, ArrayStore
from foundry.types import DataGrid, RegisteredBatch
from collections.abc import Mapping, Sequence
from foundry.types import ArrayBatch, Formula
from foundry.types import (Candidate, ContextConflict, DataGrid, EvaluationReport,
                                   FamilySeed, RegisteredBatch, WeightReport)
from foundry.types import SessionBackend, CandidateGrid, FamilyResult, Diagnostics, DiscoveryReport, ExecutionSchedule
from foundry.types import ProgramPlan, ProgramSimulation, StrategyProgram, ProgramReference, ProgramSource, ProgramCatalog, ProgramCheckpoint, SimulationResult
from foundry.types import ResidentEvaluation
from foundry.study import programs
from foundry.study.program_registry import ProgramRegistry
from foundry.study.replay_evaluation import ReplayEvaluation
from foundry.study.cache import ReviewedBatch


if TYPE_CHECKING:
    from foundry.study.pipeline import Pipeline
    from foundry.study.resources import Runtime

Tensor = TypeVar('Tensor', bound=Array)


class Ledger:
    def __init__(self, runtime: Runtime, store: LibraryStore[Library], arrays: ArrayStore,
                 registry: ProgramRegistry | None = None):
        pipeline = runtime.pipeline
        self.runtime, self.pipeline, self.store, self.arrays = runtime, pipeline, store, arrays
        self.programs = (ProgramRegistry(pipeline.candidate_batch, pipeline.context.id)
                         if registry is None else registry)
        self.replay = ReplayEvaluation(runtime, arrays, self.programs)
        self.orders = runtime.settings.execution.model == 'orders'
        self.reviewed = ReviewedBatch(min(pipeline.candidate_batch, pipeline.cache.reports.capacity_bytes // 4096))
        library = store.read(pipeline.study)
        if not self.orders:
            for part in pipeline.batches(tuple(member.source for member in library.members)):
                self._prepare_sources(part)
                pipeline.members(part)
        self.portfolio = self._build_portfolio(library)

    def _build_portfolio(self, library: Library) -> Portfolio:
        pipeline = self.pipeline
        with pipeline.resources.scope():
            members = library.members or (SEED,)
            if self.orders:
                coefficients = pipeline.xp.asarray([[member.weight for member in members]])
                _, rows, replayed = self.replay.simulate(tuple(member.source for member in members), coefficients)
                measured = rows[0]
                reference = Reference.build(None, measured, pipeline.recency_weights, pipeline.ppy, members,
                    not library.members, pipeline.backend, pipeline.portfolio_cfg, library.version,
                    mean_exposure=float(replayed.mean_net[0]))
                return Portfolio(library, reference)
            books = [book for book, _ in pipeline.members(tuple(m.source for m in members))]
            xp = pipeline.xp
            U = xp.sum(xp.stack(books) * xp.asarray([m.weight for m in members])[:, None, None], axis=0)
            U[:pipeline.decision_start] = 0
            measured = pipeline.metrics([cap_book(U, pipeline.allocation, xp)])[0]
            reference = Reference.build(U, measured, pipeline.recency_weights, pipeline.ppy, members,
                not library.members, pipeline.backend, pipeline.portfolio_cfg, library.version,
                mean_exposure=float(measured.book.sum(axis=1).mean()))
            return Portfolio(library, reference)

    def _admit(self, rows, portfolio: Portfolio):
        pipeline = self.pipeline
        with pipeline.resources.scope():
            valid = [(i, row) for i, row in enumerate(rows) if row.ok and row.valid_returns]
            selected = {}
            for part in pipeline.batches(valid):
                ids, measured = zip(*part)
                plan = admission.plan(measured, portfolio.reference)
                if self.orders:
                    result = self._order_trials(measured, portfolio, plan.weights)
                else:
                    books = pipeline.xp.ascontiguousarray(pipeline.xp.stack([row.book for row in measured]))
                    result = pipeline.blend(books, pipeline.xp.asarray(portfolio.reference.U, dtype=pipeline.xp.float64),
                                            plan.owners, plan.weights.ravel())
                proposals = admission.proposals(measured, portfolio.reference, plan, result, pipeline.ppy,
                    f'{pipeline.backend.name}-{pipeline.settings.execution.precision}' if self.orders
                    else f'{pipeline.spec.backend}-{pipeline.precision}', pipeline.to_host)
                selected.update(zip(ids, proposals))
            return admission.admit(rows, selected)

    def _order_trials(self, rows, portfolio: Portfolio, shares: FloatArray) -> SimulationResult:
        """Batch new variants over one owner slot; bound live accounts per replay chunk."""
        xp = self.pipeline.xp
        current = portfolio.library.members or (SEED,)
        sources = tuple(member.source for member in current)
        prepared = self.replay.members(sources)
        base = xp.asarray([member.weight for member in current])
        groups: dict[ProgramSource | int, list[int]] = {}
        for i, row in enumerate(rows):
            source = row.source
            identity = replace(source, scope='') if isinstance(source, (ArraySource, ProgramSource)) else source
            key = replace(source, variant=0) if isinstance(source, ProgramSource) and identity not in sources else i
            groups.setdefault(key, []).append(i)
        B, K = shares.shape
        returns = xp.empty((B, K, len(portfolio.reference.rr)), dtype=xp.float64)
        statistics = xp.empty((B, K, 5), dtype=xp.float64)
        for indices in groups.values():
            source = rows[indices[0]].source
            identity = replace(source, scope='') if isinstance(source, (ArraySource, ProgramSource)) else source
            existing = identity in sources
            at = sources.index(identity) if existing else len(sources)
            members = prepared if existing else prepared + self.replay.members((source,))
            if len(indices) > 1:
                variants = xp.repeat(xp.asarray([rows[i].source.variant for i in indices], dtype=xp.int64), K)
                members = members[:-1] + (replace(members[-1], variant_indices=variants),)
            index = xp.asarray(indices, dtype=xp.int64)
            part = shares[index].ravel()
            weights = xp.zeros((len(part), len(members)), dtype=xp.float64)
            weights[:, :len(current)] = (1 - part[:, None]) * base
            weights[:, at] += part
            result, _, _ = self.replay.simulate_members(members, weights, report=False)
            returns[index] = result.returns.reshape(len(indices), K, -1)
            statistics[index] = result.statistics.reshape(len(indices), K, 5)
        return SimulationResult(returns.reshape(B*K, -1), statistics.reshape(B*K, 5), None)

    def _prepare_sources(self, sources):
        pipeline = self.pipeline
        with pipeline.resources.scope():
            if any(isinstance(source, ProgramSource) for source in sources):
                raise ValueError('Program candidates require Execution(model="orders") before opening the session')
            arrays = tuple(source for source in dict.fromkeys(sources) if isinstance(source, ArraySource))
            for part in pipeline.batches(arrays):
                values, missing = [], False
                for source in part:
                    if source.snapshot != self.pipeline.context.id:
                        raise ContextConflict('candidate array belongs to a different research snapshot')
                    resident = pipeline.inputs.read(source) if source.scope else None
                    if pipeline.has_array(source):
                        values.append(pipeline.cache.arrays[source])
                    else:
                        batch = resident if source.scope else self.arrays.read(source)
                        if not source.scope:
                            pipeline.validate_batch(batch)
                        values.append(pipeline.xp.asarray(batch.values[0]))
                        missing = True
                if missing:
                    # Keep the complete requested group together: evicting an old
                    # allocation cannot discard an earlier cache hit in this call.
                    pipeline.register_arrays(part, pipeline.xp.stack(values), owned=True)

    def register_batch(self, batch, context_id):
        self.require_context(context_id)
        pipeline = self.pipeline
        with pipeline.resources.scope():
            pipeline.validate_batch(batch)
            if pipeline.inputs.count + self.programs.count + len(batch.ids) > pipeline.candidate_batch:
                raise ValueError('Resident candidate capacity exceeded; release a batch or split the input')
            token, lease = pipeline.inputs.register(batch)
            if not self.orders:
                pipeline.register_arrays(lease.sources, lease.batch.values, owned=True)
            candidates = tuple(Candidate(identity, source) for identity, source in zip(batch.ids, lease.sources))
            self.reviewed.clear()
            return RegisteredBatch(self.context, candidates, token)

    def register_program(self, program: StrategyProgram | ProgramReference, context_id: str) -> RegisteredBatch:
        self.require_context(context_id)
        if not self.orders:
            raise ValueError('Program candidates require Execution(model="orders") before opening the session')
        token, candidates = self.programs.register(program, occupied=self.pipeline.inputs.count)
        with ExitStack() as cleanup:
            cleanup.callback(self.programs.release, token)
            programs.plan(self.runtime, self.programs.read(candidates[0].source))
            cleanup.pop_all()
        self.reviewed.clear()
        return RegisteredBatch(self.context, candidates, token)

    def release_batch(self, token: str) -> None:
        with self.pipeline.resources.scope():
            if self.programs.owns(token):
                self.programs.release(token)
                self.reviewed.clear()
                return
            sources = self.pipeline.inputs.release(token)
            self.pipeline.cache.arrays.drop(sources)
            self.reviewed.clear()

    def retain(self, candidates: tuple[Candidate, ...], context_id: str) -> RegisteredBatch:
        """Persist only explicitly selected live sources; query paths never write."""
        self.require_context(context_id)
        pipeline = self.pipeline
        with pipeline.resources.scope():
            for candidate in candidates:
                if isinstance(candidate.source, ArraySource) and candidate.source.snapshot != pipeline.context.id:
                    raise ContextConflict('candidate array belongs to a different research snapshot')
            live = tuple(dict.fromkeys(c.source for c in candidates if isinstance(c.source, ArraySource) and c.source.scope))
            for source in live:
                pipeline.inputs.read(source)
            pending = [source for source in live if source not in pipeline.inputs.retained]
            for part, batch in pipeline.inputs.select(pending):
                host = replace(batch, values=pipeline.resources.stage_host(batch.values),
                    observed_at=pipeline.resources.stage_host(batch.observed_at),
                    fit_at=pipeline.resources.stage_host(batch.fit_at),
                    update_mask=None if batch.update_mask is None else pipeline.resources.stage_host(batch.update_mask))
                pipeline.resources.synchronize()
                saved = tuple(replace(source, scope='') for source in part)
                self.arrays.put(host, saved)
                pipeline.inputs.retained.update(zip(part, saved))
            sources = {}
            for candidate in candidates:
                source = candidate.source
                if isinstance(source, ProgramSource):
                    sources[source] = self.programs.retain(source)
                elif isinstance(source, ArraySource) and source.scope:
                    sources[source] = pipeline.inputs.retained[source]
                else:
                    sources[source] = source
            return RegisteredBatch(self.context, tuple(Candidate(c.id, sources[c.source]) for c in candidates), '')

    @property
    def context(self):
        snapshot = self.pipeline.context
        version = self.portfolio.version
        identity = hashlib.sha256(repr((snapshot.id, version)).encode()).hexdigest()
        return ResearchContext(identity, snapshot, version)

    def require_context(self, context_id):
        if context_id != self.context.id:
            raise ContextConflict('stale research context; describe the current study again')

    def evaluate_device(self, sources, context_id, *, report=False, delayed=False):
        self.require_context(context_id)
        return self._evaluate_device(sources, report, delayed)

    def _evaluate_device(self, sources, report, delayed):
        rows, verdicts = [], []
        for lo in range(0, len(sources), self.pipeline.candidate_batch):
            part = sources[lo:lo + self.pipeline.candidate_batch]
            measured = self._evaluate_sources(part, report=report, delayed=delayed)
            admitted = self._admit(measured, self.portfolio)
            if self.orders and report and not delayed:
                self.reviewed.put(self.context.id, measured, admitted)
            rows.extend(measured)
            verdicts.extend(admitted)
        return DeviceBatch(tuple(rows), tuple(verdicts))

    def _evaluate_sources(self, sources, *, report=True, delayed=False):
        if self.orders:
            return self.replay.evaluate(sources, report=report, delayed=delayed)
        self._prepare_sources(sources)
        return self.pipeline.evaluate_sources(sources, report=report, delayed=delayed)

    def evaluate(self, candidates, context_id):
        self.require_context(context_id)
        sources = tuple(dict.fromkeys(c.source for c in candidates))
        by_source = {}
        for lo in range(0, len(sources), self.pipeline.candidate_batch):
            part = sources[lo:lo + self.pipeline.candidate_batch]
            batch = self.evaluate_device(part, context_id, report=True)
            for source, row, verdict in zip(part, batch.rows, batch.verdicts):
                by_source[source] = (replace(row, tree=None, book=None, returns=None, maxdd=None), verdict)
        return EvaluationResult(self.context, tuple(CandidateResult(c.id, *by_source[c.source]) for c in candidates))

    def commit(self, candidates, context_id):
        self.require_context(context_id)
        before, original = self.context, self.portfolio
        working = original
        results = []
        # A request fitting one device batch benefits from full parallelism.
        # Multi-batch submissions bound work invalidated by early acceptances.
        adaptive = len(candidates) > self.pipeline.commit_batch
        lookahead = 1 if adaptive else self.pipeline.commit_batch
        for lo in range(0, len(candidates), self.pipeline.candidate_batch):
            part = candidates[lo:lo + self.pipeline.candidate_batch]
            sources = tuple(c.source for c in part)
            reviewed = self.reviewed.read(before.id, sources) if self.orders and working.version == original.version else {}
            if reviewed:
                self.programs.read_many(tuple(source for source in reviewed if isinstance(source, ProgramSource)))
            missing = tuple(source for source in sources if source not in reviewed)
            measured = iter(self._evaluate_sources(missing, report=True)) if missing else iter(())
            rows = [replace(reviewed[source].row) if source in reviewed else next(measured) for source in sources]
            materialized = {i for i, source in enumerate(sources) if source not in reviewed}
            at = 0
            while at < len(part):
                # Same version: read-only trial batches can run together. After
                # the first acceptance the suffix must be judged on its successor.
                first = at
                stop = min(len(part), at + lookahead)
                # Only server-owned reports from this exact context can replace
                # trials. A successor portfolio forces fresh suffix admission.
                pending = [i for i in range(at, stop) if working.version != original.version or sources[i] not in reviewed]
                needed = [i for i in pending if i not in materialized]
                if needed:
                    for i, row in zip(needed, self._evaluate_sources(tuple(sources[i] for i in needed), report=True)):
                        rows[i] = row
                    materialized.update(needed)
                admitted = dict(zip(pending, self._admit([rows[i] for i in pending], working))) if pending else {}
                verdicts = tuple(admitted[i] if i in admitted else reviewed[sources[i]].verdict for i in range(at, stop))
                for row, verdict, candidate in zip(rows[at:stop], verdicts, part[at:stop]):
                    at += 1
                    report = replace(row, tree=None, book=None, returns=None, maxdd=None)
                    results.append(CandidateResult(candidate.id, report, verdict, verdict.passed,
                                                    verdict.proposal.weight if verdict.passed else 0.))
                    if verdict.passed:
                        durable = self.retain((candidate,), before.id).candidates[0].source
                        row = replace(row, source=durable)
                        if not self.orders:
                            self.pipeline.retain_member(row)
                        successor = (working.library.allocate(row.source, verdict.proposal.weight) if self.orders
                                     else working.accepted(row, verdict))
                        working = self._build_portfolio(successor)
                        # An early acceptance invalidates speculative suffixes.
                        # Use its observed distance for the next version.
                        if adaptive:
                            lookahead = at - first
                        break
                else:
                    lookahead = min(2 * lookahead, self.pipeline.commit_batch)
        if working.version != original.version:
            self.store.commit(original.library, working.library)
            self.portfolio = working
            self.reviewed.clear()
        return CommitResult(before, self.context, tuple(results))

    def portfolio_report(self, context_id):
        self.require_context(context_id)
        members = self.portfolio.library.members or (SEED,)
        return PortfolioReport(self.context, tuple(m.source for m in members), tuple(m.weight for m in members),
                               self.portfolio.reference.metrics, not self.portfolio.library.members)

    def evaluate_weights(self, weights, context_id):
        with self.pipeline.resources.scope():
            self.require_context(context_id)
            xp = self.pipeline.xp
            matrix = self.validate_weights(weights)
            result = self.weight_trials(matrix, context_id)
            host = self.pipeline.to_host
            return WeightReport(self.context, tuple(tuple(row) for row in host(matrix).tolist()),
                                tuple(host(result.feasible).tolist()), tuple(host(result.utility).tolist()),
                                tuple(tuple(row) for row in host(result.delta).tolist()))

    def validate_weights(self, weights: FloatArray) -> FloatArray:
        xp = self.pipeline.xp
        matrix = xp.asarray(weights, dtype=xp.float64)
        members = self.portfolio.library.members or (SEED,)
        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] != len(members):
            raise ValueError('weights must be a nonempty (trials, portfolio members) matrix')
        if not bool(xp.isfinite(matrix).all()) or bool((matrix < 0).any()):
            raise ValueError('portfolio component weights must be finite and nonnegative')
        if self.runtime.settings.account.composition == 'independent' and bool((matrix.sum(1) > 1).any()):
            raise ValueError('Independent account capital allocations cannot exceed total initial cash')
        return matrix

    def member_states(self, context_id):
        with self.pipeline.resources.scope():
            self.require_context(context_id)
            if self.orders:
                raise ValueError('Order portfolios expose member_returns; account-dependent programs have no static books')
            from_seed = self.portfolio.library.members
            members = from_seed or (SEED,)
            states = self.pipeline.members([m.source for m in members])
            return [m.weight for m in members], [b for b, _ in states], self.pipeline.xp.stack([r for _, r in states])

    def member_returns(self, context_id):
        if self.orders:
            self.require_context(context_id)
            members = self.portfolio.library.members or (SEED,)
            rows = self.replay.evaluate(tuple(member.source for member in members), report=False)
            return tuple(member.weight for member in members), self.pipeline.xp.stack([row.returns for row in rows])
        weights, _, returns = self.member_states(context_id)
        return tuple(weights), returns

    def weight_trials(self, weights, context_id):
        self.require_context(context_id)
        if self.orders:
            members = self.portfolio.library.members or (SEED,)
            result, _, _ = self.replay.simulate(tuple(member.source for member in members), weights, report=False)
            return self._weight_metrics(result.returns, result.statistics[:, 3], self.portfolio.reference)
        _, books, _ = self.member_states(context_id)
        return self.trial_weights(weights, self.pipeline.xp.stack(books), context_id)


    def trial_weights(self, weights, books, context_id):
        with self.pipeline.resources.scope():
            self.require_context(context_id)
            xp, ref = self.pipeline.xp, self.portfolio.reference
            result = self.pipeline.returns(cap_book(xp.tensordot(weights, books, axes=(1, 0)), self.pipeline.allocation, xp))
            return self._weight_metrics(result.returns, result.maxdd, ref)


    def _weight_metrics(self, rr, dd, ref):
        measured = admission.weight_metrics(rr, dd, ref, self.pipeline.ppy, dd.shape)
        return WeightTrials(measured.feasible, measured.utility, measured.delta)

    def reweight(self, context_id: str, *, iterations: int = 3, count: int = 32, seed: int = 1) -> ReweightResult:
        from foundry.backtest.weights import prepare as prepare_weights
        self.require_context(context_id)
        if len(self.portfolio.library.members) < 2:
            return ReweightResult(self.context, self.context, False, 0)
        pipeline = self.pipeline
        with pipeline.resources.scope():
            if self.orders:
                members = self.portfolio.library.members
                weights = [member.weight for member in members]
                prepared = self.replay.members(tuple(member.source for member in members))
            else:
                weights, books, _ = self.member_states(context_id)
                resident_books = pipeline.xp.stack(books)
            proposals = optimize(weights, prepare_weights(), xp=pipeline.xp, to_host=pipeline.to_host,
                iterations=iterations, count=count, seed=seed, batch_size=pipeline.candidate_batch,
                capital_budget=self.runtime.settings.account.composition == 'independent')
            try:
                trial = next(proposals)
                while True:
                    if self.orders:
                        result, _, _ = self.replay.simulate_members(prepared, trial, report=False)
                        score = self._weight_metrics(result.returns, result.statistics[:, 3], self.portfolio.reference)
                    else:
                        score = self.trial_weights(trial, resident_books, context_id)
                    trial = proposals.send(score)
            except StopIteration as completed:
                result = completed.value
            if not result.weights:
                return ReweightResult(self.context, self.context, False, result.evaluations)
            committed = self.commit_weights(result.weights, context_id)
            return ReweightResult(committed.before, committed.context, committed.changed, result.evaluations, committed.utility, result.weights)

    def commit_weights(self, weights, context_id):
        with self.pipeline.resources.scope():
            self.require_context(context_id)
            before, original = self.context, self.portfolio
            xp = self.pipeline.xp
            values = xp.asarray(weights, dtype=xp.float64)
            current = original.library.members or (SEED,)
            if values.shape != (len(current),) or not bool(xp.isfinite(values).all()) or bool((values < 0).any()):
                raise ValueError('weights must be one finite nonnegative value per portfolio member')
            self.validate_weights(values[None])
            members = tuple(Member(m.source, float(w)) for m, w in zip(current, values) if w > 0)
            if not members:
                members = (Member(SEED.source, 0.),)
            working = self._build_portfolio(Library(members, original.library.study))
            ref = working.reference
            trial = self._weight_metrics(ref.rr[None], self.pipeline.xp.asarray([ref.maxdd]), original.reference)
            if not bool(trial.feasible[0]):
                return WeightCommit(before, before, False, float(trial.utility[0]))
            self.store.commit(original.library, working.library)
            self.portfolio = working
            self.reviewed.clear()
            return WeightCommit(before, self.context, True, float(trial.utility[0]))


@dataclass(frozen=True)
class DiscoveryServices:
    diagnose: Callable[[ResidentPort], Diagnostics]
    judge_family: Callable[[ResidentPort, Sequence[FamilySeed | CandidateGrid]], FamilyResult]


class LocalBackend:
    def __init__(self, runtime: Runtime, research: Ledger, describe: Callable[[Ledger], dict], discovery: DiscoveryServices):
        self.runtime, self.research, self._describe = runtime, research, describe
        self.discovery = discovery

    def context(self):
        return self.research.context

    def refresh(self):
        with self.runtime.resources.scope():
            previous = self.research
            replacement = Ledger(self.runtime, previous.store, previous.arrays, previous.programs)
            previous.portfolio = replacement.portfolio
            previous.reviewed.clear()
        return self.research.context

    def grid(self):
        engine = self.research.pipeline
        return DataGrid(engine.context.id, engine.decision_times, tuple(engine.panel.assets))

    def describe(self):
        return self._describe(self.research)

    def plan_program(self, program: StrategyProgram | ProgramReference, context: str) -> ProgramPlan:
        self.research.require_context(context)
        with self.runtime.resources.scope():
            if isinstance(program, ProgramReference):
                program = self.research.programs.resolve(program)
            return programs.plan(self.runtime, program)

    def simulate(self, program: StrategyProgram | ProgramReference, context: str, *, until: int | None = None,
                 checkpoint: ProgramCheckpoint | None = None) -> ProgramSimulation:
        self.research.require_context(context)
        if isinstance(program, ProgramReference):
            program = self.research.programs.resolve(program)
        return programs.simulate(self.runtime, self.research.context, program, until=until, checkpoint=checkpoint)

    def register_program(self, program: StrategyProgram | ProgramReference, context: str) -> RegisteredBatch:
        with self.runtime.resources.scope():
            return self.research.register_program(program, context)

    def evaluate(self, candidates, context):
        return evaluation_report(self.research.evaluate(candidates, context))

    def evaluate_device(self, candidates, context):
        self.research.require_context(context)
        return self.resident().evaluate(candidates, report=True)

    def commit(self, candidates, context):
        return evaluation_report(self.research.commit(candidates, context))

    def register(self, batch, context):
        with self.runtime.resources.scope():
            return self.research.register_batch(batch, context)

    def retain(self, candidates, context):
        return self.research.retain(candidates, context)

    def release(self, lease):
        self.research.release_batch(lease)

    def judge_family(self, families, context):
        self.research.require_context(context)
        return self.discovery.judge_family(self.resident(), families)

    def diagnose(self, context):
        self.research.require_context(context)
        return self.discovery.diagnose(self.resident())

    def portfolio(self, context):
        return self.research.portfolio_report(context)

    def trial_weights(self, weights, context):
        return self.research.evaluate_weights(weights, context)

    def commit_weights(self, weights, context):
        return self.research.commit_weights(weights, context)

    def reweight(self, context, iterations, count, seed):
        return self.research.reweight(context, iterations=iterations, count=count, seed=seed)

    def resident(self):
        return ResidentPort(self.research)

    def close(self):
        self.research.reviewed.clear()
        self.research.programs.close()
        self.runtime.close()


FormulaInput = str | Sequence[str] | Mapping[str, str]
CandidateInput = FormulaInput | Candidate | Sequence[Candidate] | RegisteredBatch


def formula_candidates(values: FormulaInput) -> tuple[Candidate, ...]:
    if isinstance(values, str):
        values = (values,)
    if isinstance(values, Mapping):
        pairs = tuple(values.items())
    else:
        pairs = tuple((hashlib.sha256(text.encode()).hexdigest(), text) for text in values)
    if not pairs or any(not identity or not isinstance(text, str) or not text.strip() for identity, text in pairs):
        raise ValueError('provide at least one nonempty formula and candidate ID')
    if len({identity for identity, _ in pairs}) != len(pairs):
        raise ValueError('candidate IDs must be unique; use named formulas for repeated expressions')
    return tuple(Candidate(identity, Formula(text)) for identity, text in pairs)


class Session:
    """One owned research session; closing a remote client never stops its server.

    Queries use the cached context. Commands consume the context of the reviewed
    report and do not refresh or retry after a conflict. Call refresh explicitly
    when another writer has changed the portfolio.
    """
    def __init__(self, backend: SessionBackend[ResidentPort]):
        self._backend = backend
        self._context = backend.context()
        self.grid = backend.grid()
        self._leases: set[str] = set()

    @property
    def context(self):
        return self._context

    def describe(self):
        """Return the machine-readable fields, operators and execution contract."""
        return self._backend.describe()

    def refresh(self):
        """Explicitly observe a portfolio changed by another writer."""
        self._context = self._backend.refresh()
        self.grid = self._backend.grid()
        return self._context

    def _candidates(self, inputs: CandidateInput) -> tuple[Candidate, ...]:
        if isinstance(inputs, RegisteredBatch):
            if inputs.context.snapshot.id != self._context.snapshot.id:
                raise ContextConflict('registered inputs belong to a different data snapshot')
            candidates = inputs.candidates
        elif isinstance(inputs, Candidate):
            candidates = (inputs,)
        elif isinstance(inputs, (str, Mapping)):
            candidates = formula_candidates(inputs)
        else:
            values = tuple(inputs)
            candidates = values if values and isinstance(values[0], Candidate) else formula_candidates(values)
        if not candidates or any(not isinstance(value, Candidate) for value in candidates):
            raise TypeError('provide a nonempty batch of typed candidates or formulas')
        if any(not isinstance(value.id, str) or not value.id for value in candidates) or len({value.id for value in candidates}) != len(candidates):
            raise ValueError('candidate IDs must be nonempty and unique')
        if any(not isinstance(value.source, (Formula, ArraySource, ProgramSource, EqualWeight)) for value in candidates):
            raise TypeError('candidate sources must use a published source type')
        return tuple(candidates)

    def evaluate(self, inputs: CandidateInput | StrategyProgram | ProgramReference) -> EvaluationReport:
        if isinstance(inputs, (StrategyProgram, ProgramReference)):
            registered = self.register(inputs)
            with ExitStack() as cleanup:
                cleanup.callback(self.release, registered)
                report = self._backend.evaluate(registered.candidates, self._context.id)
                cleanup.pop_all()
            return replace(report, lease=registered.lease)
        return self._backend.evaluate(self._candidates(inputs), self._context.id)

    def plan_program(self, program: StrategyProgram | ProgramReference) -> ProgramPlan:
        """Inspect devices, bounded state and order latency before simulation."""
        return self._backend.plan_program(program, self._context.id)

    def simulate(self, program: StrategyProgram | ProgramReference, *, until: int | None = None,
                 checkpoint: ProgramCheckpoint | None = None) -> ProgramSimulation:
        """Audit a fresh actual account; portfolio admission requires replay."""
        return self._backend.simulate(program, self._context.id, until=until, checkpoint=checkpoint)

    def evaluate_device(self, inputs: CandidateInput | StrategyProgram | ProgramReference) -> ResidentEvaluation:
        """Return resident arrays and their committable report from one query."""
        if isinstance(inputs, (StrategyProgram, ProgramReference)):
            registered = self.register(inputs)
            with ExitStack() as cleanup:
                cleanup.callback(self.release, registered)
                result = self._evaluate_resident(registered.candidates)
                cleanup.pop_all()
            return replace(result, report=replace(result.report, lease=registered.lease))
        return self._evaluate_resident(self._candidates(inputs))

    def _evaluate_resident(self, candidates: tuple[Candidate, ...]) -> ResidentEvaluation:
        context = self._context
        batch = self._backend.evaluate_device(candidates, context.id)
        result = EvaluationResult(context, tuple(CandidateResult(candidate.id, row, verdict)
            for candidate, row, verdict in zip(candidates, batch.rows, batch.verdicts)))
        report = EvaluationReport(context, tuple(candidate_report(row) for row in result.rows))
        return ResidentEvaluation(report, batch)

    def commit(self, report: EvaluationReport | DiscoveryReport | ResidentEvaluation) -> CommitReport:
        """Submit the reviewed candidates against exactly the reviewed portfolio."""
        if isinstance(report, ResidentEvaluation):
            report = report.report
        if not isinstance(report, (EvaluationReport, DiscoveryReport)):
            raise TypeError('commit requires an EvaluationReport, ResidentEvaluation or DiscoveryReport with portfolio admission')
        if isinstance(report, DiscoveryReport) and report.delayed:
            raise ValueError('delayed noise-control reports cannot be submitted; evaluate the actual candidates first')
        candidates = (tuple(row.candidate for row in report.rows) if isinstance(report, DiscoveryReport)
                      else tuple(Candidate(row.id, row.source) for row in report.rows))
        result = self._backend.commit(candidates, report.context.id)
        self._context = result.context
        if isinstance(report, EvaluationReport) and report.lease:
            self.release(report)
        return result

    def register(self, values: FloatArray | StrategyProgram | ProgramReference, *, kind: Literal['scores', 'targets'] | None = None,
                 ids: Sequence[str] | None = None, grid: DataGrid | None = None, producer: str | None = None,
                 observed_at: IntArray | None = None, fit_at: IntArray | None = None, complexity: Sequence[float] | None = None,
                 unit: Literal['score', 'equity_fraction', 'quantity', 'notional'] = 'score',
                 update_mask: BoolArray | None = None) -> RegisteredBatch:
        """Own a bounded resident batch until release; this command does not persist it."""
        if isinstance(values, (StrategyProgram, ProgramReference)):
            if any(value is not None for value in (kind, ids, grid, producer, observed_at, fit_at, complexity, update_mask)) or unit != 'score':
                raise TypeError('Program registration takes only its program or deployed reference')
            registered = self._backend.register_program(values, self._context.id)
            self._leases.add(registered.lease)
            return registered
        if any(value is None for value in (kind, ids, grid, producer, observed_at, fit_at)):
            raise TypeError('Array registration requires kind, ids, grid, producer, observed_at and fit_at')
        batch = ArrayBatch(kind, tuple(ids), values, grid.snapshot, producer, grid.decision_times, grid.assets,
                           observed_at, fit_at, None if complexity is None else tuple(complexity), unit, update_mask)
        registered = self._backend.register(batch, self._context.id)
        self._leases.add(registered.lease)
        return registered

    def retain(self, inputs: CandidateInput) -> RegisteredBatch:
        """Return durable candidate references; the original lease still needs release."""
        return self._backend.retain(self._candidates(inputs), self._context.id)

    def release(self, batch: RegisteredBatch | EvaluationReport | ResidentEvaluation) -> None:
        """End a resident lease; retained references remain usable after release."""
        if isinstance(batch, ResidentEvaluation):
            batch = batch.report
        if not isinstance(batch, (RegisteredBatch, EvaluationReport)) or not batch.lease:
            raise ValueError('release requires a registered resident batch lease')
        self._backend.release(batch.lease)
        self._leases.discard(batch.lease)

    def diagnose(self) -> Diagnostics:
        """Inspect portfolio gaps without changing its context or membership."""
        return self._backend.diagnose(self._context.id)

    def judge_family(self, inputs: FormulaInput | CandidateGrid | Sequence[CandidateGrid], *, hypothesis: str = '') -> FamilyResult:
        if isinstance(inputs, CandidateGrid):
            families = (inputs,)
        elif not isinstance(inputs, (str, Mapping)):
            values = tuple(inputs)
            if values and isinstance(values[0], CandidateGrid):
                if any(not isinstance(value, CandidateGrid) for value in values):
                    raise TypeError('families must contain only CandidateGrid values')
                families = values
            else:
                families = tuple(FamilySeed(c.id, c.source.expression, hypothesis) for c in formula_candidates(values))
        else:
            families = tuple(FamilySeed(c.id, c.source.expression, hypothesis) for c in formula_candidates(inputs))
        return self._backend.judge_family(families, self._context.id)

    def portfolio(self):
        return self._backend.portfolio(self._context.id)

    def trial_weights(self, weights: FloatArray | Sequence[Sequence[float]]) -> WeightReport:
        """Evaluate a (trials, members) matrix in portfolio() member order."""
        return self._backend.trial_weights(weights, self._context.id)

    def commit_weights(self, report: WeightReport, index: int) -> WeightCommit:
        """Submit one reviewed weight vector; admission remains authoritative."""
        result = self._backend.commit_weights(report.weights[index], report.context.id)
        self._context = result.context
        return result

    def reweight(self, *, iterations: int = 3, count: int = 32, seed: int = 1) -> ReweightResult:
        """Optimize and publish an admissible successor against the current context."""
        if not 0 <= iterations <= 100 or not 2 <= count <= 8192 or seed < 0:
            raise ValueError('reweight requires iterations in [0,100], count in [2,8192] and a nonnegative seed')
        result = self._backend.reweight(self._context.id, iterations, count, seed)
        self._context = result.context
        return result

    @property
    def resident(self) -> ResidentPort:
        """Borrow a read-only in-process view bound to the current context."""
        return self._backend.resident()

    def close(self):
        try:
            for lease in tuple(self._leases):
                self._backend.release(lease)
                self._leases.remove(lease)
        finally:
            self._backend.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


def candidate_report(result: CandidateResult) -> CandidateReport:
    row = result.row
    return CandidateReport(result.id, row.source, row.ok, row.error, row.precision,
        row.metrics, row.risk, result.verdict, result.accepted, result.weight)


def evaluation_report(result: EvaluationResult | CommitResult) -> EvaluationReport | CommitReport:
    rows = tuple(candidate_report(row) for row in result.rows)
    return CommitReport(result.before, result.context, rows) if isinstance(result, CommitResult) else EvaluationReport(result.context, rows)


class ResidentPort:
    """Context-bound, zero-copy reads; no portfolio writes or resource ownership.

    Host views are non-writable. CUDA tensors are borrowed on the owning stream;
    callers must not modify them or retain them after the session closes.
    """
    __slots__ = ('__ledger', '__context')

    def __init__(self, ledger: Ledger):
        self.__ledger, self.__context = ledger, ledger.context

    @property
    def context(self) -> ResearchContext:
        self.__ledger.require_context(self.__context.id)
        return self.__context

    @property
    def xp(self) -> ModuleType:
        return self.__ledger.pipeline.xp

    @property
    def to_host(self) -> Callable[[Array], HostArray]:
        return self.__ledger.pipeline.to_host

    @property
    def device(self) -> ComputeDevice:
        self.context
        return self.__ledger.pipeline.resources.placement

    def feedback_vector(self, values: FloatArray) -> FeedbackVector:
        """Transfer ownership of a small vector; its lease survives portfolio CAS."""
        self.context
        return FeedbackVector(values, self.xp, self.device)

    @property
    def ppy(self) -> float:
        return self.__ledger.pipeline.ppy

    @property
    def settings(self) -> Settings:
        return self.__ledger.pipeline.settings

    @property
    def candidate_batch(self) -> int:
        return self.__ledger.pipeline.candidate_batch

    @property
    def vocabulary(self) -> tuple[str, ...]:
        return tuple(self.__ledger.pipeline.vocabulary)

    @property
    def window_caps(self) -> tuple[int, int]:
        pipeline = self.__ledger.pipeline
        return (pipeline.backend.max_window, pipeline.backend.max_pair_window)

    @property
    def grid(self) -> DataGrid:
        pipeline = self.__ledger.pipeline
        return DataGrid(self.context.snapshot.id, pipeline.decision_times, pipeline.panel.assets)

    @property
    def measurement_times(self) -> tuple[str, ...]:
        pipeline = self.__ledger.pipeline
        return tuple(str(day) for day in pipeline.measurement_times[pipeline.measurement_start:])

    @property
    def decision_start(self) -> int:
        return self.__ledger.pipeline.decision_start

    @property
    def measurement_start(self) -> int:
        return self.__ledger.pipeline.measurement_start

    @property
    def execution_schedule(self) -> ExecutionSchedule:
        return self.__ledger.pipeline.schedule

    @property
    def recency_weights(self) -> FloatArray:
        self.context
        return self._readonly(self.__ledger.pipeline.recency_weights)

    def resource_status(self) -> ResourceStatus:
        self.context
        return self.__ledger.pipeline.resources.status()

    def scope(self) -> AbstractContextManager[None]:
        self.context
        return self.__ledger.pipeline.resources.scope()

    def measure_returns(self, returns: FloatArray) -> ReturnMoments:
        with self.scope():
            pipeline = self.__ledger.pipeline
            return pipeline.backend.measure(returns, pipeline.recency_weights, pipeline.ppy)

    def yearly_sharpe(self, returns: FloatArray, *, last: int) -> FloatArray:
        with self.scope():
            pipeline = self.__ledger.pipeline
            return pipeline.years.sharpe(returns, pipeline.ppy, last=last)

    def normalized_complexity(self, source: Source) -> float:
        if isinstance(source, Formula):
            parsed = validate(source.expression, self.vocabulary, self.settings.expr.max_cost)
            if not parsed['ok']:
                raise ValueError(parsed['error'])
            return source.normalized_complexity(parsed['tree'].cost(), self.settings.expr.max_cost)
        return source.normalized_complexity(0, self.settings.expr.max_cost)

    def formula_error(self, expression: str) -> str | None:
        checked = validate(expression, self.vocabulary, self.settings.expr.max_cost)
        return None if checked['ok'] else checked['error']

    @property
    def persistent(self) -> bool:
        return self.__ledger.store.persistent

    @property
    def eligibility(self) -> BoolArray:
        self.context
        return self._readonly(self.__ledger.pipeline.universe)

    @property
    def features(self) -> Mapping[str, Array]:
        """Borrow causal decision-aligned fields, including completed OHLCV.

        Formula eligibility is a separate grammar restriction. The market
        panel contains observed inputs only, never future execution labels.
        """
        with self.scope():
            panel = self.__ledger.pipeline.panel
            return MappingProxyType({name: self._readonly(value) for name, value in panel.values().items()})

    @property
    def categories(self):
        return MappingProxyType({key: tuple(names) for key, names in self.__ledger.pipeline.panel.categories().items()})

    def _readonly(self, value: Tensor) -> Tensor:
        view = value.view()
        if self.__ledger.pipeline.backend.name != 'cuda':
            view.flags.writeable = False
        return view

    @property
    def reference(self) -> PortfolioView:
        self.context
        ref = self.__ledger.portfolio.reference
        return PortfolioView(None if ref.U is None else self._readonly(ref.U),
            None if ref.W is None else self._readonly(ref.W), self._readonly(ref.rr), self._readonly(ref.p),
            ref.metrics, ref.sharpe, ref.cagr, ref.maxdd, ref.wsharpe, ref.recent_growth, ref.n, ref.seed, ref.keys,
            ref.version, ref.mean_exposure)

    def member_states(self):
        weights, books, returns = self.__ledger.member_states(self.context.id)
        return tuple(weights), tuple(self._readonly(book) for book in books), self._readonly(returns)

    def member_returns(self) -> tuple[tuple[float, ...], FloatArray]:
        weights, returns = self.__ledger.member_returns(self.context.id)
        return weights, self._readonly(returns)

    def evaluate(self, candidates: tuple[Candidate, ...], *, report=True, delayed=False) -> DeviceBatch:
        batch = self.__ledger.evaluate_device(tuple(c.source for c in candidates), self.context.id, report=report, delayed=delayed)
        rows = tuple(replace(row, book=None if row.book is None else self._readonly(row.book),
                             returns=None if row.returns is None else self._readonly(row.returns))
                     for row in batch.rows)
        return DeviceBatch(rows, batch.verdicts)

    def fingerprints(self, books: FloatArray) -> tuple:
        self.context
        return tuple(self.__ledger.pipeline.fingerprints(books))

    def trial_weights(self, weights: FloatArray) -> WeightTrials:
        with self.__ledger.pipeline.resources.scope():
            matrix = self.__ledger.validate_weights(weights)
            return self.__ledger.weight_trials(matrix, self.context.id)


def open_session(settings: Settings, portfolio, describe: Callable[[Ledger], dict], discovery: DiscoveryServices,
                 *, initial: PortfolioReport | None = None, persist: bool = True,
                 programs: ProgramCatalog | None = None) -> Session:
    from foundry.study.resources import configured_study, open_runtime
    study = configured_study(settings)
    with ExitStack() as cleanup:
        runtime = open_runtime(settings, study)
        cleanup.callback(runtime.close)
        disk = FileLibraryStore(Path(portfolio)) if portfolio is not None else None
        if initial is not None:
            if initial.context.snapshot.id != runtime.pipeline.context.id:
                raise ContextConflict('checkpoint belongs to a different computation snapshot')
            members = () if initial.baseline else tuple(Member(source, weight) for source, weight in zip(initial.sources, initial.weights))
            library = Library(members, study)
            if disk is not None and persist and disk.read(study).version != library.version:
                raise ContextConflict('persistent portfolio changed since the checkpoint')
        else:
            library = disk.read(study) if disk is not None else Library((), study)
        store = disk if disk is not None and persist else MemoryLibraryStore(library)
        arrays = FileArrayStore(path(settings.portfolio.artifact_dir))
        registry = ProgramRegistry(runtime.pipeline.candidate_batch, runtime.pipeline.context.id, programs)
        cleanup.callback(registry.close)
        with runtime.resources.scope():
            session = Session(LocalBackend(runtime, Ledger(runtime, store, arrays, registry), describe, discovery))
        cleanup.pop_all()
        return session
