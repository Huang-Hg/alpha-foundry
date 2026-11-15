"""Typed in-process research, atomic portfolio commands and the public session boundary."""
from __future__ import annotations

import hashlib
from dataclasses import replace
from foundry.types import ArraySource
from foundry.portfolio import admission, family
from foundry.portfolio.reweight import optimize
from foundry.backtest.weights import prepare as prepare_weights
from foundry.types import ReweightResult
from foundry.portfolio.library import Library, Member, SEED
from foundry.portfolio.portfolio import Portfolio
from foundry.numeric.allocation import cap_book
from foundry.types import (Candidate, CandidateResult, CommitResult, ContextConflict, DeviceBatch,
                                    EvaluationResult, PortfolioReport, ResearchContext, WeightCommit,
                                    WeightReport, WeightTrials)
from contextlib import ExitStack
from pathlib import Path
from typing import Callable
from types import MappingProxyType
from foundry.config import Settings, path
from foundry.portfolio.reference import Reference
from foundry.study.pipeline import Pipeline
from foundry.study.resources import Runtime, configured_study, open_runtime
from foundry.study.store import FileArrayStore, MemoryArrayStore, FileLibraryStore, MemoryLibraryStore
from foundry.types import CandidateReport, CommitReport, Source, FloatArray, LibraryStore, ArrayStore
from foundry.types import DataGrid, RegisteredBatch
from collections.abc import Mapping, Sequence
from foundry.types import ArrayBatch, Formula
from foundry.types import (Candidate, ContextConflict, DataGrid, EvaluationReport,
                                   FamilySeed, RegisteredBatch, WeightReport)
from foundry.types import SessionBackend


class Ledger:
    def __init__(self, pipeline: Pipeline, store: LibraryStore[Library], arrays: ArrayStore):
        self.pipeline, self.store, self.arrays = pipeline, store, arrays
        library = store.read(pipeline.study)
        for part in pipeline.batches(tuple(member.source for member in library.members)):
            self._prepare_sources(part)
            pipeline.members(part)
        self.portfolio = self._build_portfolio(library)

    def _build_portfolio(self, library: Library) -> Portfolio:
        pipeline = self.pipeline
        with pipeline.resources.scope():
            members = library.members or (SEED,)
            books = [book for book, _ in pipeline.members(tuple(m.source for m in members))]
            xp = pipeline.xp
            U = xp.sum(xp.stack(books) * xp.asarray([m.weight for m in members])[:, None, None], axis=0)
            U[:pipeline.start] = 0
            measured = pipeline.metrics([cap_book(U, pipeline.allocation, xp)])[0]
            reference = Reference.build(U, measured, pipeline.recency_weights, pipeline.ppy, members,
                not library.members, pipeline.backend, pipeline.portfolio_cfg, pipeline.settings.search, library.version)
            return Portfolio(library, reference)

    def _admit(self, rows, portfolio: Portfolio):
        pipeline = self.pipeline
        with pipeline.resources.scope():
            valid = [(i, row) for i, row in enumerate(rows) if row.ok and row.valid_returns]
            selected = {}
            for part in pipeline.batches(valid):
                ids, measured = zip(*part)
                plan = admission.plan(measured, portfolio.reference)
                result = pipeline.blend(plan.books, plan.reference_book, plan.owners, plan.weights.ravel())
                proposals = admission.proposals(measured, portfolio.reference, plan, result, pipeline.ppy,
                    f'{pipeline.spec.backend}-{pipeline.precision}', pipeline.to_host)
                selected.update(zip(ids, proposals))
            return admission.admit(rows, selected)

    def _prepare_sources(self, sources):
        for source in dict.fromkeys(sources):
            if isinstance(source, ArraySource):
                if source.snapshot != self.pipeline.context.id:
                    raise ContextConflict('candidate array belongs to a different research snapshot')
                if not self.pipeline.has_array(source):
                    batch = self.arrays.read(source)
                    self.pipeline.validate_batch(batch)
                    self.pipeline.register_arrays((source,), batch.values)

    def register_batch(self, batch, context_id):
        self.require_context(context_id)
        self.pipeline.validate_batch(batch)
        host = replace(batch, values=self.pipeline.to_host(batch.values),
                       observed_through=self.pipeline.to_host(batch.observed_through),
                       fit_through=self.pipeline.to_host(batch.fit_through))
        sources = self.arrays.put(host)
        self.pipeline.register_arrays(sources, batch.values)
        return tuple(Candidate(identity, source) for identity, source in zip(batch.ids, sources))

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

    def evaluate_noise_device(self, sources, context_id):
        self.require_context(context_id)
        return self._evaluate_device(sources, True, True)

    def _evaluate_device(self, sources, report, delayed):
        rows, verdicts = [], []
        for lo in range(0, len(sources), self.pipeline.candidate_batch):
            part = sources[lo:lo + self.pipeline.candidate_batch]
            self._prepare_sources(part)
            measured = self.pipeline.evaluate_sources(part, report=report, delayed=delayed)
            rows.extend(measured)
            verdicts.extend(self._admit(measured, self.portfolio))
        return DeviceBatch(tuple(rows), tuple(verdicts))

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
            self._prepare_sources(sources)
            rows = self.pipeline.evaluate_sources(sources, report=True)
            at = 0
            while at < len(part):
                # Same version: read-only trial batches can run together. After
                # the first acceptance the suffix must be judged on its successor.
                first = at
                stop = min(len(part), at + lookahead)
                verdicts = self._admit(rows[at:stop], working)
                for row, verdict, candidate in zip(rows[at:stop], verdicts, part[at:stop]):
                    at += 1
                    report = replace(row, tree=None, book=None, returns=None, maxdd=None)
                    results.append(CandidateResult(candidate.id, report, verdict, verdict.passed,
                                                    verdict.proposal.weight if verdict.passed else 0.))
                    if verdict.passed:
                        self.pipeline.retain_member(row)
                        working = self._build_portfolio(working.accepted(row, verdict))
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
        return CommitResult(before, self.context, tuple(results))

    def family(self, seeds, context_id):
        self.require_context(context_id)
        pipeline = self.pipeline
        categories = {field: names for names in pipeline.panel.categories().values() for field in names}
        planned = family.plan(seeds, pipeline.vocabulary, categories, pipeline.expr_cfg.max_cost)
        real, twins = {}, {}
        for expressions, delayed, output in ((planned.expressions, False, real), (planned.twin_expressions, True, twins)):
            for part in pipeline.batches(expressions):
                batch = self.evaluate_device(tuple(Formula(expr) for expr in part), context_id, report=True, delayed=delayed)
                output.update(family.points(batch, self.portfolio.reference))
        return family.judge(planned, real, twins, self.context)

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
            _, books, _ = self.member_states(context_id)
            result = self.trial_weights(matrix, xp.stack(books), context_id)
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
        return matrix

    def member_states(self, context_id):
        with self.pipeline.resources.scope():
            self.require_context(context_id)
            from_seed = self.portfolio.library.members
            members = from_seed or (SEED,)
            states = self.pipeline.members([m.source for m in members])
            return [m.weight for m in members], [b for b, _ in states], self.pipeline.xp.stack([r for _, r in states])


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
        self.require_context(context_id)
        if len(self.portfolio.library.members) < 2:
            return ReweightResult(False, 0)
        pipeline = self.pipeline
        with pipeline.resources.scope():
            weights, books, _ = self.member_states(context_id)
            resident_books = pipeline.xp.stack(books)
            proposals = optimize(weights, prepare_weights(), xp=pipeline.xp, to_host=pipeline.to_host,
                iterations=iterations, count=count, seed=seed, batch_size=pipeline.candidate_batch)
            try:
                trial = next(proposals)
                while True:
                    trial = proposals.send(self.trial_weights(trial, resident_books, context_id))
            except StopIteration as completed:
                result = completed.value
            if not result.weights:
                return ReweightResult(False, result.evaluations)
            committed = self.commit_weights(result.weights, context_id)
            return ReweightResult(committed.changed, result.evaluations, committed.utility, result.weights)

    def commit_weights(self, weights, context_id):
        with self.pipeline.resources.scope():
            self.require_context(context_id)
            before, original = self.context, self.portfolio
            xp = self.pipeline.xp
            values = xp.asarray(weights, dtype=xp.float64)
            current = original.library.members or (SEED,)
            if values.shape != (len(current),) or not bool(xp.isfinite(values).all()) or bool((values < 0).any()):
                raise ValueError('weights must be one finite nonnegative value per portfolio member')
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
            return WeightCommit(before, self.context, True, float(trial.utility[0]))


class LocalBackend:
    def __init__(self, runtime: Runtime, research: Ledger, describe: Callable[[Ledger], dict]):
        self.runtime, self.research, self._describe = runtime, research, describe

    def context(self):
        return self.research.context

    def refresh(self):
        with self.runtime.resources.scope():
            previous = self.research
            replacement = Ledger(self.runtime.pipeline, previous.store, previous.arrays)
            previous.portfolio = replacement.portfolio
        return self.research.context

    def grid(self):
        engine = self.research.pipeline
        return DataGrid(engine.context.id, tuple(str(day) for day in engine.dates), tuple(engine.panel.assets))

    def describe(self):
        return self._describe(self.research)

    def evaluate(self, candidates, context):
        return evaluation_report(self.research.evaluate(candidates, context))

    def evaluate_device(self, candidates, context):
        self.research.require_context(context)
        return self.resident().evaluate(candidates, report=True)

    def commit(self, candidates, context):
        return evaluation_report(self.research.commit(candidates, context))

    def register(self, batch, context):
        with self.runtime.resources.scope():
            candidates = self.research.register_batch(batch, context)
        return RegisteredBatch(self.research.context, candidates)

    def explore(self, seeds, context):
        return self.research.family(seeds, context)

    def portfolio(self, context):
        return self.research.portfolio_report(context)

    def trial_weights(self, weights, context):
        return self.research.evaluate_weights(weights, context)

    def commit_weights(self, weights, context):
        return self.research.commit_weights(weights, context)

    def resident(self):
        return ResidentPort(self.research)

    def close(self):
        self.runtime.close()


FormulaInput = str | Sequence[str] | Mapping[str, str]


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

    def _candidates(self, inputs):
        if isinstance(inputs, RegisteredBatch):
            if inputs.context.snapshot.id != self._context.snapshot.id:
                raise ContextConflict('registered inputs belong to a different data snapshot')
            return inputs.candidates
        return formula_candidates(inputs)

    def evaluate(self, inputs: FormulaInput | RegisteredBatch) -> EvaluationReport:
        return self._backend.evaluate(self._candidates(inputs), self._context.id)

    def evaluate_device(self, inputs: FormulaInput | RegisteredBatch):
        """Return resident arrays in an in-process session, without report serialization."""
        return self._backend.evaluate_device(self._candidates(inputs), self._context.id)

    def commit(self, report: EvaluationReport):
        """Submit the reviewed candidates against exactly the reviewed portfolio."""
        result = self._backend.commit(tuple(Candidate(row.id, row.source) for row in report.rows), report.context.id)
        self._context = result.context
        return result

    def register(self, values, *, kind, ids, grid: DataGrid, producer,
                 observed_through, fit_through, complexity=None) -> RegisteredBatch:
        """Register model outputs; coordinates and information cutoffs remain explicit."""
        batch = ArrayBatch(kind, tuple(ids), values, grid.snapshot, producer, grid.dates, grid.assets,
                           observed_through, fit_through, None if complexity is None else tuple(complexity))
        return self._backend.register(batch, self._context.id)

    def explore(self, formulas: FormulaInput, *, hypothesis: str):
        seeds = tuple(FamilySeed(c.id, c.source.expression, hypothesis) for c in formula_candidates(formulas))
        return self._backend.explore(seeds, self._context.id)

    def portfolio(self):
        return self._backend.portfolio(self._context.id)

    def trial_weights(self, weights) -> WeightReport:
        """Evaluate a (trials, members) matrix in portfolio() member order."""
        return self._backend.trial_weights(weights, self._context.id)

    def commit_weights(self, report: WeightReport, index: int):
        """Submit one reviewed weight vector; admission remains authoritative."""
        result = self._backend.commit_weights(report.weights[index], report.context.id)
        self._context = result.context
        return result

    @property
    def resident(self) -> ResidentPort:
        """Borrow a read-only in-process view bound to the current context."""
        return self._backend.resident()

    def close(self):
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
    def xp(self):
        return self.__ledger.pipeline.xp

    @property
    def to_host(self):
        return self.__ledger.pipeline.to_host

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
        return (min(len(pipeline.dates), pipeline.backend.max_window),
                min(len(pipeline.dates), pipeline.backend.max_pair_window))

    @property
    def grid(self) -> DataGrid:
        pipeline = self.__ledger.pipeline
        return DataGrid(self.context.snapshot.id, tuple(str(day) for day in pipeline.dates), pipeline.panel.assets)

    @property
    def features(self):
        self.context
        return MappingProxyType({name: self._readonly(value) for name, value in self.__ledger.pipeline.fields.items()})

    @property
    def categories(self):
        return MappingProxyType({key: tuple(names) for key, names in self.__ledger.pipeline.panel.categories().items()})

    def _readonly(self, value: FloatArray) -> FloatArray:
        view = value.view()
        if self.__ledger.pipeline.backend.name != 'cuda':
            view.flags.writeable = False
        return view

    @property
    def reference(self) -> Reference:
        self.context
        ref = self.__ledger.portfolio.reference
        return replace(ref, U=self._readonly(ref.U), W=self._readonly(ref.W), rr=self._readonly(ref.rr), p=self._readonly(ref.p))

    def member_states(self):
        weights, books, returns = self.__ledger.member_states(self.context.id)
        return tuple(weights), tuple(self._readonly(book) for book in books), self._readonly(returns)

    def evaluate(self, candidates: tuple[Candidate, ...], *, report=True, delayed=False) -> DeviceBatch:
        batch = self.__ledger.evaluate_device(tuple(c.source for c in candidates), self.context.id, report=report, delayed=delayed)
        rows = tuple(replace(row, book=self._readonly(row.book), returns=self._readonly(row.returns))
                     if row.book is not None else replace(row) for row in batch.rows)
        return DeviceBatch(rows, batch.verdicts)

    def fingerprints(self, books: FloatArray) -> tuple:
        self.context
        return tuple(self.__ledger.pipeline.fingerprints(books))

    def trial_weights(self, weights: FloatArray) -> WeightTrials:
        _, books, _ = self.member_states()
        with self.__ledger.pipeline.resources.scope():
            matrix = self.__ledger.validate_weights(weights)
            return self.__ledger.trial_weights(matrix, self.xp.stack(books), self.context.id)


def open_session(settings: Settings, portfolio, describe: Callable[[Ledger], dict]) -> Session:
    study = configured_study(settings)
    with ExitStack() as cleanup:
        runtime = open_runtime(settings, study)
        cleanup.callback(runtime.close)
        if portfolio is None:
            store, arrays = MemoryLibraryStore(Library((), study)), MemoryArrayStore()
        else:
            store = FileLibraryStore(Path(portfolio))
            arrays = FileArrayStore(path(settings.portfolio.artifact_dir))
        with runtime.resources.scope():
            session = Session(LocalBackend(runtime, Ledger(runtime.pipeline, store, arrays), describe))
        cleanup.pop_all()
        return session
