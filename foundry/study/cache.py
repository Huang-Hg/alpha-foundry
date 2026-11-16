"""Bounded snapshot-owned caches for immutable candidate computation results."""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, replace
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
    """Retain only (T,) returns and compact reports, never (T,S) candidate books.

    Reserve full-grid returns and 4 KiB of report metadata per entry. Evict
    complete return allocations because row views share their batch buffer.
    Invalid entries consume the same reservation. Working books are regenerated
    in bounded batches; the cache footprint is independent of asset count.
    """

    def __init__(self, budget: int, days: int):
        self.entry_bytes = days * 8 + 4096
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
        self._rows.update((key, (group, replace(row, book=None, tree=None))) for key, row in zip(keys, rows))

    def clear(self) -> None:
        self._groups.clear()
        self._rows.clear()


class CandidateCache:
    """Own bounded inputs/member books and compact snapshot measurements."""

    def __init__(self, candidate_batch: int, max_members: int, budget: int, days: int):
        self.arrays: BoundedCache[ArraySource, FloatArray] = BoundedCache(candidate_batch)
        self.reports = MeasurementCache(budget, days)
        self.members: BoundedCache[MemberKey, tuple[FloatArray, FloatArray]] = BoundedCache(
            2 * (max_members + 1))

    def close(self) -> None:
        self.members.clear()
        self.arrays.clear()
        self.reports.clear()
