"""Registered model and agent candidate batches using the shared discovery loop."""
from collections.abc import Callable, Iterable, Sequence
import hashlib

from foundry.types import Candidate, ProposalBatch, RegisteredBatch, TrainingResult


class ExternalProposer:
    """Feedback runs in the training worker and must not access the session's device."""

    def __init__(self, batches: Iterable[Candidate | RegisteredBatch | Sequence[Candidate]], *,
                 name: str = 'external',
                 feedback: Callable[[ProposalBatch, Sequence[float]], TrainingResult] | None = None) -> None:
        self.name, self.feedback = name, feedback
        self.batches = tuple((batch,) if isinstance(batch, Candidate) else batch.candidates
                             if isinstance(batch, RegisteredBatch) else tuple(batch) for batch in batches)
        if any(not batch or any(not isinstance(candidate, Candidate) for candidate in batch) for batch in self.batches):
            raise ValueError('External proposals require nonempty Candidate batches')
        self.signature = hashlib.sha256(repr(self.batches).encode()).hexdigest()
        self.position = 0

    def sample(self, count: int, probes: int, context: Sequence[float]) -> ProposalBatch | None:
        if self.position == len(self.batches):
            return None
        candidates = self.batches[self.position]
        self.position += 1
        return ProposalBatch(self.position, len(candidates), 0, candidates)

    def train(self, batch: ProposalBatch, rewards: Sequence[float], *, steps: int,
              learning_rate: float, opd_strength: float) -> TrainingResult:
        return TrainingResult() if self.feedback is None else self.feedback(batch, rewards)

    def dump(self) -> bytes:
        return f'foundry-external-v1\n{self.signature}\n{self.position}\n'.encode('ascii')

    def load(self, payload: bytes) -> None:
        version, signature, position = payload.decode('ascii').splitlines()
        cursor = int(position)
        if version != 'foundry-external-v1' or signature != self.signature or not 0 <= cursor <= len(self.batches):
            raise ValueError('External proposer checkpoint does not match its candidate batches')
        self.position = cursor
