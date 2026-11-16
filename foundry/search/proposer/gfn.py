"""Candidate-native GFlowNet with Python-owned learning on an explicit device."""
from __future__ import annotations

from collections.abc import Sequence
from io import BytesIO
import json
from threading import Lock
from typing import TYPE_CHECKING

import numpy as np

from foundry.search.proposer.sampling import Sampler
from foundry.types import Candidate, ContextConflict, Formula, ProposalBatch, TrainingResult

if TYPE_CHECKING:
    from foundry.search.proposer.model import Learner

PROTOCOL = b'foundry-gfn-torch-v1\n'


class GfnProposer:
    """Batch inference and FP32 autograd; CUDA is explicit and never silently downgraded."""

    learner: Learner

    def __init__(self, fields: Sequence[str], seed: int, max_tokens: int, max_cost: int,
                 max_ts_depth: int, window_caps: tuple[int, int], context_size: int,
                 *, name: str = 'gfn', device: str = 'cuda') -> None:
        from foundry.search.proposer.model import Learner

        self.name = name
        self.sampler = Sampler(fields, seed, max_tokens, max_cost, max_ts_depth, window_caps, context_size)
        self.learner = Learner(self.sampler, device)
        self.trained = -1
        self.lock = Lock()

    def sample(self, count: int, probes: int, context: Sequence[float]) -> ProposalBatch:
        with self.lock:
            sampler = self.sampler
            sampler.sample(count, probes, context, lambda states: self.learner.probabilities(sampler, states))
            self.learner.prepare(sampler)
            candidates = tuple(Candidate(expression, Formula(expression)) for expression in sampler.expressions())
            return ProposalBatch(sampler.epoch, sampler.ordinary, len(sampler.probes), candidates)

    def train(self, batch: ProposalBatch, rewards: Sequence[float], *, steps: int,
              learning_rate: float, opd_strength: float) -> TrainingResult:
        with self.lock:
            if batch.epoch != self.sampler.epoch or self.trained == batch.epoch or self.learner.batch is None:
                raise ContextConflict('stale or already trained batch')
            values = np.asarray(rewards, dtype=np.float32)
            if (values.shape != (len(self.sampler.samples),) or not np.isfinite(values).all() or
                    not 0 < steps <= 100 or not 0 < learning_rate <= .1 or not 0 <= opd_strength <= 10):
                raise ValueError('invalid training input')
            result = self.learner.train(self.learner.batch, values, steps, learning_rate, opd_strength)
            self.trained = batch.epoch
            return result

    def dump(self) -> bytes:
        """Serialize only JSON metadata and numeric arrays; never pickle executable objects."""
        with self.lock:
            sampler = self.sampler
            metadata = dict(grammar=sampler.spec, context_size=sampler.context_size,
                            budgets=[sampler.max_tokens, sampler.max_cost, sampler.max_ts_depth],
                            rng=sampler.rng.bit_generator.state, epoch=sampler.epoch)
            stream = BytesIO()
            np.savez(stream, metadata=np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8),
                     **self.learner.arrays())
            return PROTOCOL + stream.getvalue()

    def load(self, payload: bytes) -> None:
        with self.lock:
            if not payload.startswith(PROTOCOL):
                raise ValueError('policy checkpoint protocol mismatch')
            sampler = self.sampler
            with np.load(BytesIO(payload[len(PROTOCOL):]), allow_pickle=False) as bundle:
                if set(bundle.files) != {'metadata', 'w1', 'w2', 'z'}:
                    raise ValueError('policy checkpoint arrays mismatch')
                metadata = json.loads(bundle['metadata'].tobytes())
                if metadata['grammar'] != sampler.spec or metadata['budgets'] != [
                        sampler.max_tokens, sampler.max_cost, sampler.max_ts_depth]:
                    raise ValueError('policy checkpoint grammar mismatch')
                if metadata['context_size'] != sampler.context_size:
                    raise ValueError('policy checkpoint context dimension mismatch')
                arrays = {name: bundle[name] for name in ('w1', 'w2', 'z')}
                shapes = dict(w1=(32, sampler.context_size + 64), w2=(len(sampler.actions), 33),
                              z=(sampler.context_size + 1,))
                if any(value.shape != shapes[name] or value.dtype != np.float32 or not np.isfinite(value).all()
                       for name, value in arrays.items()):
                    raise ValueError('policy checkpoint parameter contract mismatch')
                rng = np.random.default_rng()
                rng.bit_generator.state = metadata['rng']
                self.learner.restore(arrays)
            sampler.rng, sampler.epoch = rng, int(metadata['epoch'])
            sampler.samples, sampler.probes = [], []
            self.trained = sampler.epoch
