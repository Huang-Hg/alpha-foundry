"""Panel: a market's data as a set of named (T,S) fields on one date x asset grid.

A field is a component: it declares its name, what kind of series it is (category, scope), whether formulas may
read it, and - for a derived field - which fields it depends on and how it is computed from them with the panel's
operator backend. A panel is composed from a market's source fields plus any derived fields; values are computed on
first access and kept, on the host (numpy) or on the device (cupy) depending on the backend. New data is a new
field definition; nothing else changes.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Dict, Mapping, Protocol, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from foundry.types import Array, Backend, HostArray


class FieldCompute(Protocol):
    def __call__(self, backend: Backend, *dependencies: Array) -> Array: ...


@dataclass(frozen=True)
class Field:
    name: str
    category: str                          # family shown to formula authors ('trend', 'onchain', ...)
    scope: str = 'asset'                   # 'asset' varies across assets; 'market' is one series for every asset
    formula: bool = True                   # readable by formulas (pricing inputs such as open/close are not)
    deps: Tuple[str, ...] = ()
    compute: FieldCompute | None = None   # compute(backend, *deps) -> (T,S); None = a source field
    doc: str = ''


class Panel:
    def __init__(self, dates: NDArray[np.datetime64], assets: Sequence[str], fields: Sequence[Field],
                 sources: Mapping[str, HostArray], backend: Backend):
        dates.setflags(write=False)
        self.dates, self.assets, self.backend = dates, tuple(assets), backend
        digest=hashlib.sha256(dates.tobytes()+('\n'.join(assets)).encode())
        for name,value in sorted(sources.items()):
            digest.update(name.encode());digest.update(np.ascontiguousarray(value).tobytes())
            value.setflags(write=False)
        self.snapshot=digest.hexdigest()
        self.fields: Dict[str, Field] = {f.name: f for f in fields}
        xp = backend.xp
        self._values: Dict[str, Array] = {k: xp.ascontiguousarray(xp.asarray(v)) for k, v in sources.items()}
        if set(self._values) != {f.name for f in fields if f.compute is None}:
            raise ValueError('source fields and data differ')

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.dates), len(self.assets)

    def __getitem__(self, name: str) -> Array:
        if name not in self._values:
            f = self.fields[name]
            self._values[name] = self.backend.xp.ascontiguousarray(f.compute(self.backend, *(self[d] for d in f.deps)))
        return self._values[name]

    def values(self, names: Sequence[str] | None = None) -> Dict[str, Array]:
        return {k: self[k] for k in (names or self.fields)}

    @property
    def vocabulary(self) -> list[str]:
        """Names formulas may reference."""
        return sorted(k for k, f in self.fields.items() if f.formula)

    def categories(self) -> Dict[str, list[str]]:
        out: Dict[str, list[str]] = {}
        for k, f in self.fields.items():
            if f.formula:
                out.setdefault(f.category, []).append(k)
        return out

    def release(self) -> None:
        """Release arrays owned by this study after its runtime is closed."""
        self._values.clear()
