"""Bounded snapshot-owned caches for immutable candidate computation results."""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Hashable, Sequence
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


class MeasurementCache:
    """Evict complete allocations, since row views retain their whole device batch.

    The budget reserves a full (T,S) book and (T,) return per entry, plus 4 KiB
    for report metadata. Invalid entries consume the same reservation. Allocation
    groups stay intact even when only one row is recently used.
    """

    def __init__(self, budget: int, shape: tuple[int, int]):
        days, assets = shape
        self.entry_bytes = days * (assets + 1) * 8 + 4096
        self.capacity = budget // self.entry_bytes
        self.capacity_bytes = budget
        self._groups: OrderedDict[MeasurementKey, tuple[MeasurementKey, ...]] = OrderedDict()
        self._rows: dict[MeasurementKey, tuple[MeasurementKey, EvaluationRow]] = {}

    def __contains__(self, key: MeasurementKey) -> bool:
        return key in self._rows

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def reserved_bytes(self) -> int:
        return len(self) * self.entry_bytes

    def __getitem__(self, key: MeasurementKey) -> EvaluationRow:
        group, row = self._rows[key]
        self._groups.move_to_end(group)
        return row

    def put(self, keys: Sequence[MeasurementKey], rows: Sequence[EvaluationRow]) -> None:
        if len(keys) > self.capacity:
            return
        while len(self) + len(keys) > self.capacity:
            _, expired = self._groups.popitem(last=False)
            for key in expired:
                del self._rows[key]
        group = keys[0]
        self._groups[group] = tuple(keys)
        self._rows.update((key, (group, row)) for key, row in zip(keys, rows))

    def clear(self) -> None:
        self._groups.clear()
        self._rows.clear()


class CandidateCache:
    """A runtime owns copied inputs and borrowed (T,S) books/(T,) returns."""

    def __init__(self, candidate_batch: int, max_members: int, budget: int, shape: tuple[int, int]):
        self.arrays: BoundedCache[ArraySource, FloatArray] = BoundedCache(candidate_batch)
        self.reports = MeasurementCache(budget, shape)
        self.members: BoundedCache[MemberKey, tuple[FloatArray, FloatArray]] = BoundedCache(
            max(candidate_batch, 2 * (max_members + 1)))

    def close(self) -> None:
        self.members.clear()
        self.arrays.clear()
        self.reports.clear()
