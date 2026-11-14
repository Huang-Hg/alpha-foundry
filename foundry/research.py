"""Shared read-only evaluation and atomic portfolio commands."""
import hashlib
from dataclasses import replace
from foundry.candidates import ArraySource

from foundry.portfolio import admission, family
from foundry.portfolio.library import Library, Member, SEED
from foundry.portfolio.portfolio import Portfolio
from foundry.portfolio.allocation import cap_book
from foundry.research_types import (Candidate, CandidateResult, CommitResult, ContextConflict, DeviceBatch,
                                    EvaluationResult, ResearchContext, WeightCommit, WeightTrials)


class Research:
    def __init__(self, engine, store, arrays, null_shift=False):
        self.engine, self.store, self.arrays = engine, store, arrays
        library = store.read(engine.study)
        sources = tuple(member.source for member in library.members)
        for lo in range(0, len(sources), engine.candidate_batch):
            part = sources[lo:lo + engine.candidate_batch]
            self._prepare_sources(part)
            engine.members(part, null_shift)
        self.portfolio = Portfolio(engine, library, null_shift)

    def _prepare_sources(self, sources):
        for source in dict.fromkeys(sources):
            if isinstance(source, ArraySource):
                if source.snapshot != self.engine.context.id:
                    raise ContextConflict('candidate array belongs to a different research snapshot')
                if not self.engine.has_array(source):
                    batch = self.arrays.read(source)
                    self.engine.validate_batch(batch)
                    self.engine.register_arrays((source,), batch.values)

    def register_batch(self, batch, context_id):
        self.require_context(context_id)
        self.engine.validate_batch(batch)
        host = replace(batch, values=self.engine.to_host(batch.values),
                       observed_through=self.engine.to_host(batch.observed_through),
                       fit_through=self.engine.to_host(batch.fit_through))
        sources = self.arrays.put(host)
        self.engine.register_arrays(sources, batch.values)
        return tuple(Candidate(identity, source) for identity, source in zip(batch.ids, sources))

    @property
    def context(self):
        snapshot = self.engine.context
        version = self.portfolio.version
        identity = hashlib.sha256(repr((snapshot.id, version, self.portfolio.null_shift)).encode()).hexdigest()
        return ResearchContext(identity, snapshot, version, self.portfolio.null_shift)

    def require_context(self, context_id):
        if context_id != self.context.id:
            raise ContextConflict('stale research context; describe the current study again')

    def evaluate_device(self, sources, context_id, *, report=False):
        self.require_context(context_id)
        return self._evaluate_device(sources, report, self.portfolio.null_shift)

    def evaluate_noise_device(self, sources, context_id):
        self.require_context(context_id)
        return self._evaluate_device(sources, True, True)

    def _evaluate_device(self, sources, report, null_shift):
        rows, verdicts = [], []
        for lo in range(0, len(sources), self.engine.candidate_batch):
            part = sources[lo:lo + self.engine.candidate_batch]
            self._prepare_sources(part)
            measured = self.engine.evaluate_sources(part, report=report, null_shift=null_shift)
            rows.extend(measured)
            verdicts.extend(self.portfolio.admit(measured))
        return DeviceBatch(tuple(rows), tuple(verdicts))

    def evaluate(self, candidates, context_id):
        self.require_context(context_id)
        sources = tuple(dict.fromkeys(c.source for c in candidates))
        by_source = {}
        for lo in range(0, len(sources), self.engine.candidate_batch):
            part = sources[lo:lo + self.engine.candidate_batch]
            batch = self.evaluate_device(part, context_id, report=True)
            for source, row, verdict in zip(part, batch.rows, batch.verdicts):
                by_source[source] = (replace(row, tree=None, book=None, returns=None, maxdd=None), verdict)
        return EvaluationResult(self.context, tuple(CandidateResult(c.id, *by_source[c.source]) for c in candidates))

    def commit(self, candidates, context_id):
        self.require_context(context_id)
        before, original = self.context, self.portfolio
        working = original
        results = []
        for lo in range(0, len(candidates), self.engine.candidate_batch):
            part = candidates[lo:lo + self.engine.candidate_batch]
            sources = tuple(c.source for c in part)
            self._prepare_sources(sources)
            rows = self.engine.evaluate_sources(sources, report=True, null_shift=working.null_shift)
            at = 0
            while at < len(part):
                # Same version: read-only trial batches can run together. After
                # the first acceptance the suffix must be judged on its successor.
                stop = min(len(part), at + self.engine.commit_batch)
                verdicts = working.admit(rows[at:stop])
                for row, verdict, candidate in zip(rows[at:stop], verdicts, part[at:stop]):
                    at += 1
                    report = replace(row, tree=None, book=None, returns=None, maxdd=None)
                    results.append(CandidateResult(candidate.id, report, verdict, verdict.passed,
                                                    verdict.proposal.weight if verdict.passed else 0.))
                    if verdict.passed:
                        self.engine.retain_member(row, working.null_shift)
                        working = working.accepted(row, verdict)
                        break
        if working.version != original.version:
            self.store.commit(original.library, working.library)
            self.portfolio = working
        return CommitResult(before, self.context, tuple(results))

    def family(self, seeds, context_id):
        self.require_context(context_id)
        return family.expand(self, seeds, context_id)

    def member_states(self, context_id):
        with self.engine.resources.scope():
            self.require_context(context_id)
            from_seed = self.portfolio.library.members
            members = from_seed or (SEED,)
            states = self.engine.members([m.source for m in members], self.portfolio.null_shift)
            return [m.weight for m in members], [b for b, _ in states], self.engine.xp.stack([r for _, r in states])


    def trial_weights(self, weights, books, context_id):
        with self.engine.resources.scope():
            self.require_context(context_id)
            xp, ref = self.engine.xp, self.portfolio.reference
            result = self.engine.returns(cap_book(xp.tensordot(weights, books, axes=(1, 0)), self.engine.allocation, xp), 'f64')
            return self._weight_metrics(result.returns, result.maxdd, ref)


    def _weight_metrics(self, rr, dd, ref):
        measured = admission.weight_metrics(rr, dd, ref, self.engine.ppy, dd.shape)
        return WeightTrials(measured.feasible, measured.utility, measured.delta)

    def commit_weights(self, weights, context_id):
        with self.engine.resources.scope():
            self.require_context(context_id)
            before, original = self.context, self.portfolio
            members = tuple(Member(m.source, w) for m, w in zip(original.library.members, weights) if w > 0)
            working = Portfolio(self.engine, Library(members, original.library.study), original.null_shift)
            ref = working.reference
            trial = self._weight_metrics(ref.rr[None], self.engine.xp.asarray([ref.maxdd]), original.reference)
            if not bool(trial.feasible[0]):
                return WeightCommit(before, before, False, float(trial.utility[0]))
            self.store.commit(original.library, working.library)
            self.portfolio = working
            return WeightCommit(before, self.context, True, float(trial.utility[0]))
