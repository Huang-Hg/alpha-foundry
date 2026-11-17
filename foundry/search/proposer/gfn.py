"""Candidate-native GFlowNet with Python-owned learning on an explicit device."""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from io import BytesIO
import json
from threading import Lock
from typing import TYPE_CHECKING

import numpy as np

from foundry.config import SearchConfig
from foundry.search.proposer.batch import BatchOwner
from foundry.search.proposer.sampling import Sampler
from foundry.types import Candidate, ComputeDevice, Formula, LearningFeedback, ProposalBatch, ProposalRequest, TrainingResult

if TYPE_CHECKING:
    from foundry.search.proposer.model import Learner

PROTOCOL = b'foundry-gfn-torch-v3\n'


class GfnProposer:
    """Resource-free policy specification; learning settings belong to this producer."""

    def __init__(self, fields: Sequence[str], seed: int, max_tokens: int, max_cost: int,
                 max_ts_depth: int, window_caps: tuple[int, int], context_size: int,
                 *, name: str = 'gfn', probes: int = SearchConfig.probes,
                 steps: int = SearchConfig.train_steps, learning_rate: float = SearchConfig.learning_rate,
                 opd_strength: float = SearchConfig.opd_strength) -> None:
        if not 0 <= probes <= 8192 or not 0 < steps <= 100 or not 0 < learning_rate <= .1 or not 0 <= opd_strength <= 10:
            raise ValueError('invalid GFlowNet learning settings')
        self.name = name
        self.grammar = (tuple(fields), seed, max_tokens, max_cost, max_ts_depth, window_caps, context_size)
        self.probes, self.steps, self.learning_rate, self.opd_strength = probes, steps, learning_rate, opd_strength

    @contextmanager
    def open(self, device: ComputeDevice) -> Iterator[GfnRun]:
        run = GfnRun(self, device)
        try:
            yield run
        finally:
            run.close()


class GfnRun:
    """Own one optimizer, CUDA stream and current trace until the run closes."""
    learner: Learner

    def __init__(self, specification: GfnProposer, device: ComputeDevice) -> None:
        from foundry.search.proposer.model import Learner

        self.name, self.specification = specification.name, specification
        self.device = device
        self.sampler = Sampler(*specification.grammar)
        self.learner = Learner(self.sampler, device, specification.grammar[1])
        self.batches = BatchOwner()
        self.lock = Lock()

    def propose(self, request: ProposalRequest) -> ProposalBatch:
        with self.lock:
            self.batches.ready()
            sampler, count, probes = self.sampler, request.count, self.specification.probes
            if not 0 < count <= 8192 or probes > count:
                raise ValueError('invalid batch')
            values = np.asarray(request.features, dtype=np.float32)
            if values.shape != (sampler.context_size,) or not np.isfinite(values).all():
                raise ValueError(f'context needs {sampler.context_size} finite values')
            sequences = self.learner.sample(count, probes, values)
            sampler.epoch += 1
            candidates = tuple(Candidate(expression, Formula(expression))
                               for expression in map(sampler.expression, sequences))
            return self.batches.issue(request, candidates, count)

    def learn(self, feedback: LearningFeedback) -> TrainingResult:
        with self.lock:
            self.batches.check(feedback.batch, learning=True)
            if feedback.rewards.device != self.device:
                raise ValueError('feedback and policy must reside on the same device')
            cfg = self.specification
            result = self.learner.train(self.learner.batch, feedback.rewards,
                                        cfg.steps, cfg.learning_rate, cfg.opd_strength)
            self.batches.learned = True
            return result

    def release(self, batch: ProposalBatch) -> None:
        with self.lock:
            self.batches.release(batch)
            self.learner.release()

    def close(self) -> None:
        with self.lock:
            self.batches.close()
            self.learner.close()

    def dump(self) -> bytes:
        """Serialize only JSON metadata and numeric arrays; never pickle executable objects."""
        with self.lock:
            self.batches.ready()
            sampler = self.sampler
            metadata = dict(grammar=sampler.spec, context_size=sampler.context_size,
                            budgets=[sampler.max_tokens, sampler.max_cost, sampler.max_ts_depth],
                            device=self.learner.device.type, epoch=sampler.epoch,
                            learning=[self.specification.probes, self.specification.steps,
                                      self.specification.learning_rate, self.specification.opd_strength])
            stream = BytesIO()
            np.savez(stream, metadata=np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8),
                     rng=self.learner.random_state(), **self.learner.arrays())
            return PROTOCOL + stream.getvalue()

    def load(self, payload: bytes) -> None:
        with self.lock:
            self.batches.ready()
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
                cfg = self.specification
                if metadata['learning'] != [cfg.probes, cfg.steps, cfg.learning_rate, cfg.opd_strength]:
                    raise ValueError('policy checkpoint learning settings mismatch')
                arrays = {name: bundle[name] for name in ('w1', 'w2', 'z')}
                shapes = dict(w1=(32, sampler.context_size + 64), w2=(len(sampler.actions), 33),
                              z=(sampler.context_size + 1,))
                if any(value.shape != shapes[name] or value.dtype != np.float32 or not np.isfinite(value).all()
                       for name, value in arrays.items()):
                    raise ValueError('policy checkpoint parameter contract mismatch')
                self.learner.restore(arrays, bundle['rng'])
            sampler.epoch = int(metadata['epoch'])
