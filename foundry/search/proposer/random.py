"""Uniform legal formula proposals with the same candidate and checkpoint contract."""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
import json

from foundry.search.proposer.batch import BatchOwner
from foundry.search.proposer.sampling import Sampler
from foundry.types import Candidate, ComputeDevice, Formula, LearningFeedback, ProposalBatch, ProposalRequest, TrainingResult


class RandomProposer:
    def __init__(self, fields: Sequence[str], seed: int, max_tokens: int, max_cost: int,
                 max_ts_depth: int, window_caps: tuple[int, int], context_size: int,
                 *, name: str = 'random') -> None:
        self.name = name
        self.grammar = (tuple(fields), seed, max_tokens, max_cost, max_ts_depth, window_caps, context_size)

    @contextmanager
    def open(self, device: ComputeDevice) -> Iterator[RandomRun]:
        run = RandomRun(self)
        try:
            yield run
        finally:
            run.batches.close()


class RandomRun:
    def __init__(self, specification: RandomProposer) -> None:
        self.name = specification.name
        self.sampler = Sampler(*specification.grammar)
        self.batches = BatchOwner()

    def propose(self, request: ProposalRequest) -> ProposalBatch:
        self.batches.ready()
        self.sampler.sample(request.count, request.features)
        candidates = tuple(Candidate(expression, Formula(expression)) for expression in self.sampler.expressions())
        return self.batches.issue(request, candidates, len(candidates))

    def learn(self, feedback: LearningFeedback) -> TrainingResult:
        self.batches.check(feedback.batch, learning=True)
        self.batches.learned = True
        return TrainingResult()

    def release(self, batch: ProposalBatch) -> None:
        self.batches.release(batch)

    def dump(self) -> bytes:
        self.batches.ready()
        return json.dumps(dict(protocol='foundry-random-v2', grammar=self.sampler.spec,
            budgets=[self.sampler.max_tokens, self.sampler.max_cost, self.sampler.max_ts_depth],
            context_size=self.sampler.context_size, rng=self.sampler.rng.bit_generator.state,
            epoch=self.sampler.epoch)).encode()

    def load(self, payload: bytes) -> None:
        self.batches.ready()
        metadata = json.loads(payload)
        if metadata['protocol'] != 'foundry-random-v2':
            raise ValueError('random checkpoint protocol mismatch')
        if (metadata['grammar'] != self.sampler.spec or metadata['budgets'] != [self.sampler.max_tokens,
                self.sampler.max_cost, self.sampler.max_ts_depth] or metadata['context_size'] != self.sampler.context_size):
            raise ValueError('random checkpoint grammar mismatch')
        self.sampler.rng.bit_generator.state = metadata['rng']
        self.sampler.epoch = int(metadata['epoch'])
