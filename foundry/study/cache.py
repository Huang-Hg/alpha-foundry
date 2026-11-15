"""Bounded snapshot-owned caches for immutable candidate computation results."""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Hashable
from dataclasses import dataclass
from typing import Generic, TypeVar

from foundry.types import ArraySource, EvaluationRow, FloatArray, Source

Key = TypeVar('Key', bound=Hashable)
Value = TypeVar('Value')


class BoundedCache(Generic[Key, Value]):
    """Least-recently-used storage; callers retain ownership of borrowed values."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._values: OrderedDict[Key, Value] = OrderedDict()

    def __contains__(self, key: Key) -> bool:
        return key in self._values

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, key: Key) -> Value:
        value = self._values[key]
        self._values.move_to_end(key)
        return value

    def __setitem__(self, key: Key, value: Value) -> None:
        self._values[key] = value
        self._values.move_to_end(key)
        if len(self._values) > self.capacity:
            self._values.popitem(last=False)

    def clear(self) -> None:
        self._values.clear()


@dataclass(frozen=True)
class MeasurementKey:
    snapshot: str
    source: Source
    precision: str
    delayed: bool


@dataclass(frozen=True)
class MemberKey:
    source: Source
    delayed: bool


class CandidateCache:
    """A runtime owns copied inputs and borrowed (T,S) books/(T,) returns."""

    def __init__(self, candidate_batch: int, max_members: int):
        self.arrays: BoundedCache[ArraySource, FloatArray] = BoundedCache(candidate_batch)
        self.reports: BoundedCache[MeasurementKey, EvaluationRow] = BoundedCache(candidate_batch)
        self.members: BoundedCache[MemberKey, tuple[FloatArray, FloatArray]] = BoundedCache(
            max(candidate_batch, 2 * (max_members + 1)))

    def close(self) -> None:
        self.members.clear()
        self.arrays.clear()
        self.reports.clear()
