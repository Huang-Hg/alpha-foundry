"""Candidate-native behavioral archive shared by every proposal modality."""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import ModuleType

from foundry.study.session import ResidentPort
from foundry.types import BoolArray, Candidate, ContextConflict, FloatArray


def standardized(returns: FloatArray, xp: ModuleType) -> tuple[FloatArray, BoolArray]:
    centered = returns - returns.mean(axis=1, keepdims=True)
    norm = xp.sqrt(xp.sum(centered * centered, axis=1, keepdims=True))
    valid = xp.isfinite(returns).all(1) & (norm[:, 0] > 1e-8)
    return xp.where(valid[:, None], centered / xp.maximum(norm, 1e-8), 0), valid


@dataclass(frozen=True)
class BehaviorState:
    candidate: Candidate
    fingerprint: int
    book: FloatArray
    returns: FloatArray
    portfolio_version: str
    snapshot: str


class Archive:
    """Bounded candidate keys with cross-modality book equality verification."""

    def __init__(self, capacity: int = 32):
        if capacity < 0:
            raise ValueError('archive capacity must be nonnegative')
        self.capacity, self.version = capacity, 0
        self.items: OrderedDict[Candidate, BehaviorState] = OrderedDict()

    def update(self, staged: Mapping[Candidate, BehaviorState]) -> None:
        for candidate, state in staged.items():
            self.items[candidate] = state
            self.items.move_to_end(candidate)
        while len(self.items) > self.capacity:
            self.items.popitem(last=False)
        if staged:
            self.version += 1

    def retained(self, candidates: Sequence[Candidate]) -> None:
        """Replace live source references after persistence, preserving behavior/version."""
        self.items = OrderedDict((candidate, BehaviorState(candidate, state.fingerprint, state.book,
            state.returns, state.portfolio_version, state.snapshot))
            for candidate, state in zip(candidates, self.items.values()))

    def duplicate(self, fingerprint: int, book: FloatArray, portfolio_version: str,
                  port: ResidentPort) -> bool:
        return any(state.fingerprint == fingerprint and state.portfolio_version == portfolio_version
                   and bool(port.xp.array_equal(book, state.book)) for state in self.items.values())

    def vectors(self, port: ResidentPort) -> FloatArray:
        if any(state.snapshot != port.context.snapshot.id for state in self.items.values()):
            raise ContextConflict('archive behaviors belong to a different study snapshot')
        xp = port.xp
        return (standardized(xp.stack([state.returns for state in self.items.values()]), xp)[0]
                if self.items else xp.empty((0, len(port.reference.rr))))
