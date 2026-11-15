"""Uniform legal formula proposals with the same candidate and checkpoint contract."""
from collections.abc import Sequence

from foundry.search.proposer.gfn import GfnProposer
from foundry.types import ProposalBatch, TrainingResult


class RandomProposer:
    def __init__(self, fields: Sequence[str], seed: int, max_tokens: int, max_cost: int,
                 max_ts_depth: int, window_caps: tuple[int, int], context_size: int,
                 *, name: str = 'random') -> None:
        self.name = name
        self.policy = GfnProposer(fields, seed, max_tokens, max_cost, max_ts_depth,
                                  window_caps, context_size, name=name)

    def sample(self, count: int, probes: int, context: Sequence[float]) -> ProposalBatch:
        return self.policy.draw(count + 12 * probes, 0, context, uniform=True)

    def train(self, batch: ProposalBatch, rewards: Sequence[float], *, steps: int,
              learning_rate: float, opd_strength: float) -> TrainingResult:
        return TrainingResult()

    def dump(self) -> bytes:
        return self.policy.dump()

    def load(self, payload: bytes) -> None:
        self.policy.load(payload)
