"""Injected discovery lifecycle with policy training overlapping portfolio commands."""
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import replace
import time

from foundry.search.archive import Archive
from foundry.search.diagnostics import CalibrationPopulation, sample_calibration
from foundry.search.objective import Discovery
from foundry.study.session import Session
from foundry.types import (
    CheckpointStore, DiscoveryReport, MiningCheckpoint, MiningEpoch, MiningStatus,
    MiningTimings, ProposalBatch, ProposalProducer, ProposalRequest, Proposer, ProposerState,
    LearningFeedback, ContextConflict, TelemetrySink, TrainingResult,
)


def learn(proposer: ProposalProducer, feedback: LearningFeedback) -> tuple[TrainingResult, float]:
    """The worker borrows only immutable feedback, never mutable session state."""
    started = time.perf_counter()
    learned = proposer.learn(feedback)
    return learned, time.perf_counter() - started


class MiningDriver:
    """One serial session writer with named, resumable heterogeneous proposers."""

    def __init__(self, session: Session, proposers: Sequence[Proposer], checkpoints: CheckpointStore,
                 telemetry: TelemetrySink, *, signature: str) -> None:
        if not proposers or len({proposer.name for proposer in proposers}) != len(proposers):
            raise ValueError('A mining run requires nonempty, uniquely named proposers')
        self.session, self.proposers = session, tuple(proposers)
        self.checkpoints, self.telemetry, self.signature = checkpoints, telemetry, signature
        self.cfg = session.resident.settings.search
        if session.resident.persistent and not self.cfg.accept:
            raise ValueError('a persistent discovery session requires search.accept=True; use an in-memory session otherwise')
        if self.cfg.noise_control and self.cfg.seed_portfolio != 'equal':
            raise ValueError('The noise control delays every expression book: set search.seed_portfolio=equal')
        if self.cfg.seed_portfolio == 'equal' and self.cfg.accept:
            raise ValueError('An equal-weight experiment cannot overwrite the production portfolio')
        if self.cfg.objective == 'single' and (self.cfg.accept or self.cfg.noise_control):
            raise ValueError('The single-alpha experiment requires an in-memory portfolio and undelayed candidates')
        self.archive = Archive(self.cfg.archive_size)

    def _calibration(self) -> CalibrationPopulation | None:
        if self.cfg.objective != 'single':
            return None
        return sample_calibration(self.session.resident)

    def _discovery(self, calibration: CalibrationPopulation | None) -> Discovery:
        return Discovery(self.session, self.archive, objective=self.cfg.objective,
                         rho_max=self.cfg.rho_max, calibration=calibration)

    def _publish(self, status: MiningStatus) -> None:
        self._status = status
        self.telemetry.publish(status)

    def run(self) -> MiningStatus:
        """Publish lifecycle failures at this boundary and propagate the original error."""
        self._started = time.perf_counter()
        resources = self.session.resident.resource_status()
        self._status = MiningStatus('ready', 0, self.cfg.epochs, 0, {}, 0., self.session.portfolio(),
            objective=self.cfg.objective, noise_control=self.cfg.noise_control,
            device=resources.device, used_mib=resources.used_mib)
        try:
            with ExitStack() as lifetime:
                producers = tuple(lifetime.enter_context(p.open(self.session.resident.device)) for p in self.proposers)
                return self._run(producers)
        except KeyboardInterrupt:
            self._publish(replace(self._status, phase='interrupted',
                                  elapsed_seconds=time.perf_counter() - self._started))
            raise
        except Exception:
            self._publish(replace(self._status, phase='failed',
                                  elapsed_seconds=time.perf_counter() - self._started))
            raise

    def _run(self, producers: tuple[ProposalProducer, ...]) -> MiningStatus:
        cfg = self.cfg
        checkpoint = self.checkpoints.read()
        begin, accepted, elapsed = 0, 0, 0.
        totals: Counter[str] = Counter()
        if checkpoint is not None:
            if checkpoint.signature != self.signature:
                raise ValueError('Resume data/config/protocol mismatch')
            if checkpoint.portfolio.context.id != self.session.context.id:
                raise ValueError('Resume portfolio context mismatch')
            if tuple(state.name for state in checkpoint.proposers) != tuple(p.name for p in producers):
                raise ValueError('Resume proposer configuration mismatch')
            for proposer, state in zip(producers, checkpoint.proposers):
                proposer.load(state.payload)
            begin, accepted, elapsed = checkpoint.epoch, checkpoint.accepted, checkpoint.elapsed_seconds
            totals.update(checkpoint.counters)
        started = time.perf_counter() - elapsed
        self._started = started
        status = replace(self._status, epoch=begin, accepted=accepted, counters=dict(totals), elapsed_seconds=elapsed)
        self._publish(status)
        with self.session.resident.scope(), ThreadPoolExecutor(max_workers=1) as trainer:
            calibration = self._calibration()
            if checkpoint is not None and checkpoint.archive:
                rebuild = self._discovery(calibration)
                rebuild.evaluate(checkpoint.archive)
                self.archive.update(rebuild.pending)
            for epoch in range(begin, cfg.epochs):
                if self.telemetry.stopping():
                    status = replace(status, phase='stopped')
                    break
                discovery = self._discovery(calibration)
                archive_version = self.archive.version
                status = replace(status, phase='sampling', epoch=epoch + 1, recent=(), timings=MiningTimings(),
                                 gate=discovery.diagnostics.gate, proposer=None, training=None)
                self._publish(status)
                sample_start = time.perf_counter()
                batch = None
                for offset in range(len(producers)):
                    proposer = producers[(epoch + offset) % len(producers)]
                    request = ProposalRequest(self.session.context, cfg.batch_size, tuple(discovery.context()))
                    batch = proposer.propose(request)
                    if batch is not None:
                        if not isinstance(batch, ProposalBatch):
                            raise TypeError('Proposers must return ProposalBatch or None')
                        if batch.context != request.context:
                            raise ContextConflict('proposer returned a batch for a different research context')
                        if not batch.candidates:
                            raise ValueError('Proposers must return nonempty candidate batches')
                        break
                if batch is None:
                    status = replace(status, phase='complete', epoch=epoch)
                    break
                with ExitStack() as batch_lifetime:
                    batch_lifetime.callback(proposer.release, batch)
                    timings = MiningTimings(sampling_seconds=time.perf_counter() - sample_start)
                    status = replace(status, phase='evaluating', timings=timings, proposer=proposer.name)
                    self._publish(status)
                    eval_start = time.perf_counter()
                    rows = []
                    batch_size = self.session.resident.candidate_batch
                    for lo in range(0, len(batch.candidates), batch_size):
                        report = discovery.evaluate(batch.candidates[lo:lo + batch_size])
                        rows.extend(report.rows)
                        progress = totals.copy()
                        progress.update(discovery.counters)
                        resources = self.session.resident.resource_status()
                        status = replace(status, counters=dict(progress), recent=tuple(rows[-8:]),
                                         elapsed_seconds=time.perf_counter() - started,
                                         device=resources.device, used_mib=resources.used_mib)
                        self._publish(status)
                    report = DiscoveryReport(report.context, report.objective, tuple(rows), report.delayed)
                    feedback = discovery.feedback(batch)
                    batch_lifetime.callback(feedback.rewards.close)
                    timings = replace(timings, evaluation_seconds=time.perf_counter() - eval_start)
                    status = replace(status, phase='training', timings=timings)
                    self._publish(status)
                    training = trainer.submit(learn, proposer, feedback)
                    batch_lifetime.callback(training.result)
                    commit_start = time.perf_counter()
                    candidate = discovery.accept()
                    timings = replace(timings, commit_seconds=time.perf_counter() - commit_start)
                    accepted += int(candidate is not None)
                    status = replace(status, phase='reweighting', accepted=accepted, timings=timings)
                    self._publish(status)
                    reweight = None
                    reweight_start = time.perf_counter()
                    if not cfg.noise_control and cfg.reweight_iterations and (candidate is not None or epoch % cfg.reweight_every == 0):
                        reweight = self.session.reweight(iterations=cfg.reweight_iterations,
                                                         count=cfg.reweight_batch, seed=cfg.seed + epoch)
                    timings = replace(timings, reweight_seconds=time.perf_counter() - reweight_start)
                    retention_start = time.perf_counter()
                    self.archive.update(discovery.pending)
                    if self.archive.items:
                        retained = self.session.retain(tuple(self.archive.items))
                        self.archive.retained(retained.candidates)
                    retention_seconds = time.perf_counter() - retention_start
                    learned, training_seconds = training.result()
                    timings = replace(timings, training_seconds=training_seconds, retention_seconds=retention_seconds)
                totals.update(discovery.counters)
                totals['auxiliary_evaluations'] += batch.auxiliary
                totals['reweight_evaluations'] += reweight.evaluations if reweight is not None else 0
                portfolio = self.session.portfolio()
                resources = self.session.resident.resource_status()
                status = replace(status, phase='checkpoint', counters=dict(totals), portfolio=portfolio,
                                 elapsed_seconds=time.perf_counter() - started, timings=timings, training=learned,
                                 device=resources.device, used_mib=resources.used_mib)
                record = MiningEpoch(epoch + 1, self.signature, proposer.name, archive_version, report,
                                     candidate.candidate if candidate is not None else None,
                                     reweight, learned, timings, dict(discovery.counters), tuple(self.archive.items))
                checkpoint_start = time.perf_counter()
                self.telemetry.record(record)
                self.checkpoints.publish(MiningCheckpoint(self.signature, epoch + 1, portfolio,
                    tuple(self.archive.items), dict(totals), accepted, time.perf_counter() - started,
                    tuple(ProposerState(p.name, p.dump()) for p in producers)))
                timings = replace(timings, checkpoint_seconds=time.perf_counter() - checkpoint_start)
                status = replace(status, timings=timings)
                self._publish(status)
            else:
                status = replace(status, phase='complete')
        status = replace(status, elapsed_seconds=time.perf_counter() - started)
        self._publish(status)
        return status
