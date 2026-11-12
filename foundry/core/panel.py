"""Panel: a market's data as a set of named (T,S) fields on one date x asset grid.

A field is a component: it declares its name, what kind of series it is (category, scope), whether formulas may
read it, and - for a derived field - which fields it depends on and how it is computed from them with the panel's
operator backend. A panel is composed from a market's source fields plus any derived fields; values are computed on
first access and kept, on the host (numpy) or on the device (cupy) depending on the backend. New data is a new
field definition; nothing else changes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from foundry.expr.evaluate import Backend


@dataclass(frozen=True)
class Field:
    name: str
    category: str                          # family shown to formula authors ('trend', 'onchain', ...)
    scope: str = 'asset'                   # 'asset' varies across assets; 'market' is one series for every asset
    formula: bool = True                   # readable by formulas (pricing inputs such as open/close are not)
    deps: Tuple[str, ...] = ()
    compute: Optional[Callable] = None     # compute(backend, *deps) -> (T,S); None = a source field
    doc: str = ''


class Panel:
    def __init__(self, dates: np.ndarray, assets: Sequence[str], fields: Sequence[Field],
                 sources: Mapping[str, np.ndarray], backend: Backend):
        self.dates, self.assets, self.backend = dates, list(assets), backend
        self.fields: Dict[str, Field] = {f.name: f for f in fields}
        xp = backend.xp
        self._values = {k: xp.ascontiguousarray(xp.asarray(v)) for k, v in sources.items()}
        assert set(self._values) == {f.name for f in fields if f.compute is None}, 'source fields and data differ'

    @property
    def shape(self):
        return len(self.dates), len(self.assets)

    def __getitem__(self, name: str):
        if name not in self._values:
            f = self.fields[name]
            self._values[name] = self.backend.xp.ascontiguousarray(f.compute(self.backend, *(self[d] for d in f.deps)))
        return self._values[name]

    def values(self, names=None) -> Dict[str, object]:
        return {k: self[k] for k in (names or self.fields)}

    @property
    def vocabulary(self):
        """Names formulas may reference."""
        return sorted(k for k, f in self.fields.items() if f.formula)

    def categories(self) -> Dict[str, list]:
        out: Dict[str, list] = {}
        for k, f in self.fields.items():
            if f.formula:
                out.setdefault(f.category, []).append(k)
        return out
