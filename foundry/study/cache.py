"""Bounded snapshot-owned caches for immutable candidate computation results."""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, replace
from typing import Generic, TypeVar

from foundry.types import ArraySource, EvaluationRow, FloatArray, Source, Verdict

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


class ArrayCache:
    """Evict whole owned batch allocations; row views never pin expired groups."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._groups: OrderedDict[int, tuple[ArraySource, ...]] = OrderedDict()
        self._rows: dict[ArraySource, tuple[int, FloatArray]] = {}
        self._sequence = 0

    def __contains__(self, source: ArraySource) -> bool:
        return source in self._rows

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, source: ArraySource) -> FloatArray:
        group, values = self._rows[source]
        self._groups.move_to_end(group)
        return values

    def _remove(self, group: int) -> None:
        for source in dict.fromkeys(self._groups.pop(group)):
            del self._rows[source]

    def put(self, sources: Sequence[ArraySource], values: FloatArray) -> None:
        if len(sources) > self.capacity:
            raise ValueError('array allocation exceeds the resident candidate bound')
        for group in {self._rows[s][0] for s in sources if s in self._rows}:
            self._remove(group)
        while sum(map(len, self._groups.values())) + len(sources) > self.capacity:
            self._remove(next(iter(self._groups)))
        self._sequence += 1
        self._groups[self._sequence] = tuple(sources)
        self._rows.update((source, (self._sequence, values[i])) for i, source in enumerate(sources))

    def clear(self) -> None:
        self._rows.clear()
        self._groups.clear()

    def drop(self, sources: Sequence[ArraySource]) -> None:
        for group in {self._rows[s][0] for s in sources if s in self._rows}:
            self._remove(group)


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
    """Retain only (V,) returns and compact reports, never (T,S) candidate books.

    Reserve full-grid returns and 4 KiB of report metadata per entry. Evict
    complete return allocations because row views share their batch buffer.
    Invalid entries consume the same reservation. Working books are regenerated
    in bounded batches; the cache footprint is independent of asset count.
    """

    def __init__(self, budget: int, periods: int):
        self.entry_bytes = periods * 8 + 4096
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


@dataclass(frozen=True)
class ReviewedCandidate:
    """Detached host reports only; no returns, books, trees or live accounts."""

    row: EvaluationRow
    verdict: Verdict

    def copy(self) -> ReviewedCandidate:
        proposal = self.verdict.proposal
        return ReviewedCandidate(replace(self.row), replace(self.verdict,
            proposal=None if proposal is None else replace(proposal)))


class ReviewedBatch:
    """Keep at most the last full reported batch under one research context.

    Capacity is bounded by the active candidate batch and the existing report
    budget. Entries own host metadata only, never candidate or account arrays.
    Copies isolate mutable public rows/proposals from trusted admission results.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.context = ''
        self._rows: dict[Source, ReviewedCandidate] = {}

    def __len__(self) -> int:
        return len(self._rows)

    def put(self, context: str, rows: Sequence[EvaluationRow], verdicts: Sequence[Verdict]) -> None:
        self.clear()
        if len(rows) > self.capacity:
            return
        self.context = context
        self._rows = {row.source: ReviewedCandidate(
            replace(row, tree=None, book=None, returns=None, maxdd=None), verdict).copy()
            for row, verdict in zip(rows, verdicts)}

    def read(self, context: str, sources: Sequence[Source]) -> dict[Source, ReviewedCandidate]:
        if context != self.context:
            return {}
        return {source: self._rows[source].copy() for source in sources if source in self._rows}

    def clear(self) -> None:
        self.context = ''
        self._rows.clear()


class CandidateCache:
    """Own bounded inputs/member books and compact snapshot measurements."""

    def __init__(self, candidate_batch: int, max_members: int, budget: int, periods: int):
        self.arrays = ArrayCache(candidate_batch)
        self.reports = MeasurementCache(budget, periods)
        self.members: BoundedCache[MemberKey, tuple[FloatArray, FloatArray]] = BoundedCache(
            2 * (max_members + 1))

    def close(self) -> None:
        self.members.clear()
        self.arrays.clear()
        self.reports.clear()
