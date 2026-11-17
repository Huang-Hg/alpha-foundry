"""Single-batch ownership shared by heterogeneous proposal producers."""
from uuid import uuid4

from foundry.types import Candidate, ContextConflict, ProposalBatch, ProposalRequest


class BatchOwner:
    def __init__(self) -> None:
        self.identity = uuid4().hex
        self.sequence = 0
        self.active: ProposalBatch | None = None
        self.learned = False
        self.closed = False

    def ready(self) -> None:
        if self.closed:
            raise RuntimeError('proposal producer is closed')
        if self.active is not None:
            raise ContextConflict('release the active proposal batch before producing another')

    def issue(self, request: ProposalRequest, candidates: tuple[Candidate, ...], ordinary: int) -> ProposalBatch:
        self.ready()
        if not candidates or not 0 < ordinary <= len(candidates):
            raise ValueError('proposal batches require candidates and a valid ordinary count')
        self.sequence += 1
        self.active = ProposalBatch(self.identity, self.sequence, request.context, ordinary, candidates)
        self.learned = False
        return self.active

    def check(self, batch: ProposalBatch, *, learning: bool = False) -> None:
        if self.closed or self.active is not batch or learning and self.learned:
            raise ContextConflict('foreign, stale or already learned proposal batch')

    def release(self, batch: ProposalBatch) -> None:
        self.check(batch)
        self.active = None

    def close(self) -> None:
        self.active = None
        self.closed = True
