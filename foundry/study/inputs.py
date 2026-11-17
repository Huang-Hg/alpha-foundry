"""Bounded, explicitly released candidate leases on the study's device."""
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from types import ModuleType
from uuid import uuid4

import numpy as np

from foundry.study.identity import digest_for, metadata_for, source_for
from foundry.types import Array, ArrayBatch, ArraySource, ContextConflict, HostArray


@dataclass(frozen=True)
class InputLease:
    batch: ArrayBatch
    sources: tuple[ArraySource, ...]


class ResidentInputs:
    def __init__(self, capacity: int, xp: ModuleType, leaves: Callable[[Array], HostArray]):
        self.capacity, self.xp, self.leaves = capacity, xp, leaves
        self.scope = uuid4().hex
        self.leases: dict[str, InputLease] = {}
        self.rows: dict[ArraySource, tuple[str, int]] = {}
        self.retained: dict[ArraySource, ArraySource] = {}

    @property
    def count(self) -> int:
        return sum(len(lease.sources) for lease in self.leases.values())

    def register(self, batch: ArrayBatch) -> tuple[str, InputLease]:
        if self.count + len(batch.ids) > self.capacity:
            raise ValueError('resident candidate capacity exceeded; release a batch or split the input')
        def own(values: Array) -> Array:
            module = np if isinstance(values, np.ndarray) else self.xp
            return module.array(values, copy=True, order='C')
        owned = replace(batch, values=self.xp.array(batch.values, copy=True, order='C'),
                        observed_at=own(batch.observed_at), fit_at=own(batch.fit_at),
                        update_mask=None if batch.update_mask is None or bool(self.xp.asarray(batch.update_mask).all())
                        else self.xp.array(batch.update_mask, copy=True, order='C'))
        arrays = (owned.values, owned.observed_at, owned.fit_at)
        if owned.update_mask is not None:
            arrays += (owned.update_mask,)
        leaves = tuple(self.leaves(values) for values in arrays)
        sources = tuple(source_for(metadata_for(owned, i),
            digest_for(metadata_for(owned, i), tuple(values[i] for values in arrays),
                       tuple(hashes[i] for hashes in leaves)), self.scope) for i in range(len(batch.ids)))
        key = uuid4().hex
        lease = InputLease(owned, sources)
        self.leases[key] = lease
        self.rows.update((source, (key, i)) for i, source in enumerate(sources))
        return key, lease

    def read(self, source: ArraySource) -> ArrayBatch:
        if source not in self.rows:
            raise ContextConflict('resident candidate lease ended or belongs to another session')
        token, index = self.rows[source]
        batch = self.leases[token].batch
        return replace(batch, ids=(batch.ids[index],), values=batch.values[index:index+1],
                       observed_at=batch.observed_at[index:index+1], fit_at=batch.fit_at[index:index+1],
                       update_mask=None if batch.update_mask is None else batch.update_mask[index:index+1],
                       complexity=None if batch.complexity is None else batch.complexity[index:index+1])

    def select(self, sources: Sequence[ArraySource]) -> Iterator[tuple[tuple[ArraySource, ...], ArrayBatch]]:
        """Group validated live rows by allocation; contiguous selections borrow it."""
        groups: dict[str, list[tuple[ArraySource, int]]] = {}
        for source in sources:
            token, index = self.rows[source]
            groups.setdefault(token, []).append((source, index))
        for token, rows in groups.items():
            batch = self.leases[token].batch
            indices = [index for _, index in rows]
            contiguous = indices == list(range(indices[0], indices[0] + len(indices)))
            selection = slice(indices[0], indices[0] + len(indices)) if contiguous else indices
            yield tuple(source for source, _ in rows), replace(batch,
                ids=tuple(batch.ids[i] for i in indices), values=batch.values[selection],
                observed_at=batch.observed_at[selection], fit_at=batch.fit_at[selection],
                update_mask=None if batch.update_mask is None else batch.update_mask[selection],
                complexity=None if batch.complexity is None else tuple(batch.complexity[i] for i in indices))

    def release(self, token: str) -> tuple[ArraySource, ...]:
        if token not in self.leases:
            raise ContextConflict('candidate batch lease is no longer active in this session')
        lease = self.leases.pop(token)
        self.rows = {source: (key, i) for key, live in self.leases.items() for i, source in enumerate(live.sources)}
        self.retained = {source: value for source, value in self.retained.items() if source in self.rows}
        return lease.sources

    def close(self) -> None:
        self.rows.clear()
        self.leases.clear()
        self.retained.clear()
