"""Registered model and agent candidate batches using the shared discovery loop."""
from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
import hashlib

from foundry.search.proposer.batch import BatchOwner
from foundry.types import Candidate, ComputeDevice, LearningFeedback, ProposalBatch, ProposalRequest, RegisteredBatch, TrainingResult


class ExternalProposer:
    """Feedback borrows only its batch vector; portfolio writes remain with the driver."""

    def __init__(self, batches: Iterable[Candidate | RegisteredBatch | Sequence[Candidate]], *,
                 name: str = 'external',
                 feedback: Callable[[LearningFeedback], TrainingResult] | None = None) -> None:
        self.name, self.feedback = name, feedback
        self.batches = tuple((batch,) if isinstance(batch, Candidate) else batch.candidates
                             if isinstance(batch, RegisteredBatch) else tuple(batch) for batch in batches)
        if any(not batch or any(not isinstance(candidate, Candidate) for candidate in batch) for batch in self.batches):
            raise ValueError('External proposals require nonempty Candidate batches')
        self.signature = hashlib.sha256(repr(self.batches).encode()).hexdigest()

    @contextmanager
    def open(self, device: ComputeDevice) -> Iterator[ExternalRun]:
        run = ExternalRun(self)
        try:
            yield run
        finally:
            run.batches.close()


class ExternalRun:
    def __init__(self, specification: ExternalProposer) -> None:
        self.name, self.specification = specification.name, specification
        self.position = 0
        self.batches = BatchOwner()

    def propose(self, request: ProposalRequest) -> ProposalBatch | None:
        self.batches.ready()
        if self.position == len(self.specification.batches):
            return None
        candidates = self.specification.batches[self.position]
        self.position += 1
        return self.batches.issue(request, candidates, len(candidates))

    def learn(self, feedback: LearningFeedback) -> TrainingResult:
        self.batches.check(feedback.batch, learning=True)
        callback = self.specification.feedback
        result = TrainingResult() if callback is None else callback(feedback)
        self.batches.learned = True
        return result

    def release(self, batch: ProposalBatch) -> None:
        self.batches.release(batch)

    def dump(self) -> bytes:
        self.batches.ready()
        return f'foundry-external-v2\n{self.specification.signature}\n{self.position}\n'.encode('ascii')

    def load(self, payload: bytes) -> None:
        self.batches.ready()
        version, signature, position = payload.decode('ascii').splitlines()
        cursor = int(position)
        if version != 'foundry-external-v2' or signature != self.specification.signature or not 0 <= cursor <= len(self.specification.batches):
            raise ValueError('External proposer checkpoint does not match its candidate batches')
        self.position = cursor
