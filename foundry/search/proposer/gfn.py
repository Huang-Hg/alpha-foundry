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

PROTOCOL = b'foundry-gfn-torch-v2\n'


class GfnProposer:
    """Batch inference and FP32 autograd; CUDA is explicit and never silently downgraded."""

    learner: Learner

    def __init__(self, fields: Sequence[str], seed: int, max_tokens: int, max_cost: int,
                 max_ts_depth: int, window_caps: tuple[int, int], context_size: int,
                 *, name: str = 'gfn', device: str = 'cuda') -> None:
        from foundry.search.proposer.model import Learner

        self.name = name
        self.sampler = Sampler(fields, seed, max_tokens, max_cost, max_ts_depth, window_caps, context_size)
        self.learner = Learner(self.sampler, device, seed)
        self.trained = -1
        self.lock = Lock()

    def sample(self, count: int, probes: int, context: Sequence[float]) -> ProposalBatch:
        with self.lock:
            sampler = self.sampler
            if not 0 < count <= 8192 or not 0 <= probes <= count:
                raise ValueError('invalid batch')
            values = np.asarray(context, dtype=np.float32)
            if values.shape != (sampler.context_size,) or not np.isfinite(values).all():
                raise ValueError(f'context needs {sampler.context_size} finite values')
            sequences = self.learner.sample(count, probes, values)
            sampler.epoch += 1
            sampler.ordinary = count
            candidates = tuple(Candidate(expression, Formula(expression))
                               for expression in map(sampler.expression, sequences))
            return ProposalBatch(sampler.epoch, count, (len(candidates) - count) // 12, candidates)

    def train(self, batch: ProposalBatch, rewards: Sequence[float], *, steps: int,
              learning_rate: float, opd_strength: float) -> TrainingResult:
        with self.lock:
            if batch.epoch != self.sampler.epoch or self.trained == batch.epoch or self.learner.batch is None:
                raise ContextConflict('stale or already trained batch')
            values = np.asarray(rewards, dtype=np.float32)
            if (values.shape != (self.sampler.ordinary + 12 * len(self.learner.batch.probe_ids),) or not np.isfinite(values).all() or
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
                            device=self.learner.device.type, epoch=sampler.epoch)
            stream = BytesIO()
            np.savez(stream, metadata=np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8),
                     rng=self.learner.random_state(), **self.learner.arrays())
            return PROTOCOL + stream.getvalue()

    def load(self, payload: bytes) -> None:
        with self.lock:
            if not payload.startswith(PROTOCOL):
                raise ValueError('policy checkpoint protocol mismatch')
            sampler = self.sampler
            with np.load(BytesIO(payload[len(PROTOCOL):]), allow_pickle=False) as bundle:
                if set(bundle.files) != {'metadata', 'rng', 'w1', 'w2', 'z'}:
                    raise ValueError('policy checkpoint arrays mismatch')
                metadata = json.loads(bundle['metadata'].tobytes())
                if metadata['grammar'] != sampler.spec or metadata['budgets'] != [
                        sampler.max_tokens, sampler.max_cost, sampler.max_ts_depth]:
                    raise ValueError('policy checkpoint grammar mismatch')
                if metadata['context_size'] != sampler.context_size:
                    raise ValueError('policy checkpoint context dimension mismatch')
                if metadata['device'] != self.learner.device.type:
                    raise ValueError('policy checkpoint random device mismatch')
                arrays = {name: bundle[name] for name in ('w1', 'w2', 'z')}
                shapes = dict(w1=(32, sampler.context_size + 64), w2=(len(sampler.actions), 33),
                              z=(sampler.context_size + 1,))
                if any(value.shape != shapes[name] or value.dtype != np.float32 or not np.isfinite(value).all()
                       for name, value in arrays.items()):
                    raise ValueError('policy checkpoint parameter contract mismatch')
                self.learner.restore(arrays, bundle['rng'])
            sampler.epoch = int(metadata['epoch'])
            self.trained = sampler.epoch
