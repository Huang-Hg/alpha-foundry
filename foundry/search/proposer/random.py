"""Uniform legal formula proposals with the same candidate and checkpoint contract."""
from collections.abc import Sequence
import json

import numpy as np

from foundry.search.proposer.sampling import Sampler
from foundry.types import Candidate, Formula, ProposalBatch, TrainingResult


class RandomProposer:
    def __init__(self, fields: Sequence[str], seed: int, max_tokens: int, max_cost: int,
                 max_ts_depth: int, window_caps: tuple[int, int], context_size: int,
                 *, name: str = 'random') -> None:
        self.name = name
        self.sampler = Sampler(fields, seed, max_tokens, max_cost, max_ts_depth, window_caps, context_size)

    def sample(self, count: int, probes: int, context: Sequence[float]) -> ProposalBatch:
        sampler = self.sampler
        sampler.sample(count + 12 * probes, 0, context,
                       lambda states: np.stack([sampler.legal(s) for s in states]), uniform=True)
        candidates = tuple(Candidate(expression, Formula(expression)) for expression in sampler.expressions())
        return ProposalBatch(sampler.epoch, sampler.ordinary, 0, candidates)

    def train(self, batch: ProposalBatch, rewards: Sequence[float], *, steps: int,
              learning_rate: float, opd_strength: float) -> TrainingResult:
        return TrainingResult()

    def dump(self) -> bytes:
        return json.dumps(dict(protocol='foundry-random-v1', grammar=self.sampler.spec,
            budgets=[self.sampler.max_tokens, self.sampler.max_cost, self.sampler.max_ts_depth],
            context_size=self.sampler.context_size, rng=self.sampler.rng.bit_generator.state,
            epoch=self.sampler.epoch)).encode()

    def load(self, payload: bytes) -> None:
        metadata = json.loads(payload)
        if metadata['protocol'] != 'foundry-random-v1':
            raise ValueError('random checkpoint protocol mismatch')
        if (metadata['grammar'] != self.sampler.spec or metadata['budgets'] != [self.sampler.max_tokens,
                self.sampler.max_cost, self.sampler.max_ts_depth] or metadata['context_size'] != self.sampler.context_size):
            raise ValueError('random checkpoint grammar mismatch')
        self.sampler.rng.bit_generator.state = metadata['rng']
        self.sampler.epoch = int(metadata['epoch'])
