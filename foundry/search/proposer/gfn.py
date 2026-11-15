"""Candidate-native GFlowNet proposals over the registered formula grammar."""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from foundry.search.proposer.grammar import grammar
from foundry.types import Candidate, Formula, ProposalBatch, TrainingResult

if TYPE_CHECKING:
    from foundry.search.proposer._gfn import Policy as NativePolicy


class GfnProposer:
    """Sample and train one native policy; native calls release the Python GIL."""

    native: NativePolicy

    def __init__(self, fields: Sequence[str], seed: int, max_tokens: int, max_cost: int,
                 max_ts_depth: int, window_caps: tuple[int, int], context_size: int,
                 *, name: str = 'gfn') -> None:
        from foundry.search.proposer import _gfn

        self.name = name
        self.spec = grammar(list(fields), window_caps)
        self.native = _gfn.Policy(self.spec, max_tokens, max_cost, max_ts_depth, seed, context_size)

    def sample(self, count: int, probes: int, context: Sequence[float]) -> ProposalBatch:
        return self.draw(count, probes, context, uniform=False)

    def draw(self, count: int, probes: int, context: Sequence[float], *, uniform: bool) -> ProposalBatch:
        """Keep ordinary candidates before the native paired-teacher trajectories."""
        batch = self.native.sample(count, probes, context, uniform)
        candidates = tuple(Candidate(expression, Formula(expression)) for expression in batch['expressions'])
        return ProposalBatch(batch['epoch'], batch['ordinary'], batch['probe_count'], candidates)

    def train(self, batch: ProposalBatch, rewards: Sequence[float], *, steps: int,
              learning_rate: float, opd_strength: float) -> TrainingResult:
        learned = self.native.train(rewards, batch.epoch, steps, learning_rate, opd_strength)
        return TrainingResult(learned['loss_before'], learned['loss_after'], learned['teachers'], learned['logz'])

    def gradient_check(self, rewards: Sequence[float]) -> float:
        return self.native.gradient_check(rewards)['max_gradient_error']

    def dump(self) -> bytes:
        return self.native.dump()

    def load(self, payload: bytes) -> None:
        self.native.load(payload)
