"""Pure CEM and leave-one-out proposals driven by externally evaluated scores."""
from __future__ import annotations

from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass
from types import ModuleType

import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from foundry.backtest.weights import WeightKernel
from foundry.types import Array, FloatArray, HostArray, WeightTrials


class CEM:
    """Cross-entropy search over independent additive member-weight logits."""

    weights: HostArray

    def __init__(self, initial_logits: HostArray | Sequence[float], kernel: WeightKernel) -> None:
        self.kernel = kernel
        self.mean = np.array(initial_logits, dtype=np.float64)
        self.sd = np.full(len(self.mean), .5)

    def sample(self, count: int, seed: int) -> HostArray:
        """Return (count,N) host float64 weights scored by the next update."""
        self.weights = self.kernel.sample(self.mean, self.sd, count, seed)
        return self.weights

    def update(self, scores: HostArray, elite: int) -> None:
        self.kernel.update(self.weights, scores, elite, self.mean, self.sd)


@dataclass(frozen=True)
class ReweightProposal:
    evaluations: int
    utility: float | None
    weights: tuple[float, ...]


def optimize(base: FloatArray, kernel: WeightKernel, *, xp: ModuleType,
             to_host: Callable[[Array], HostArray], iterations: int = 3,
             count: int = 32, seed: int = 1, batch_size: int = 256
             ) -> Generator[FloatArray, WeightTrials, ReweightProposal]:
    """Yield resident (B,N) trial weights and receive their hard admission scores.

    The caller owns books, execution, resource scopes and atomic publication.
    Empty result weights mean no feasible trial; no portfolio state is mutated.
    """
    base = xp.asarray(base, dtype=xp.float64)
    inside = xp.clip(base, 1e-6, 1 - 1e-6)
    sampler = CEM(to_host(xp.log(inside / (1 - inside))), kernel)
    best: tuple[float, FloatArray] | None = None
    evaluations = 0
    for iteration in range(iterations):
        weights = xp.asarray(sampler.sample(count, seed + iteration))
        deletes = xp.broadcast_to(base, (len(base), len(base))).copy()
        deletes[xp.arange(len(base)), xp.arange(len(base))] = 0
        trials = xp.concatenate((weights, base[None], deletes))
        score_parts = []
        for lo in range(0, len(trials), batch_size):
            part = trials[lo:lo + batch_size]
            trial = yield part
            feasible, utility = trial.feasible, trial.utility
            scores = xp.where(feasible, utility, -1e6)
            score_parts.append(scores)
            at = int(xp.argmax(scores))
            if bool(feasible[at]) and (best is None or float(scores[at]) > best[0]):
                best = (float(scores[at]), part[at].copy())
            evaluations += len(part)
        scores = xp.concatenate(score_parts)
        nfeasible = int((scores[:count] > -1e5).sum())
        if nfeasible:
            sampler.update(to_host(scores[:count]), min(nfeasible, max(1, count // 4)))
    if best is None:
        return ReweightProposal(evaluations, None, ())
    return ReweightProposal(evaluations, best[0], tuple(to_host(best[1]).tolist()))
