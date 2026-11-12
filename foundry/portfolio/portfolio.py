"""A library evaluated by an engine: its cached reference, admission against it, and acceptance into it."""
from __future__ import annotations

import hashlib
import json

from foundry.portfolio import admission
from foundry.portfolio.library import Library
from foundry.portfolio.reference import Reference


class Portfolio:
    def __init__(self, engine, library: Library, null_shift=False):
        self.engine, self.library, self.null_shift = engine, library, null_shift
        self._reference = None

    @property
    def reference(self) -> Reference:
        if self._reference is None:
            self._reference = Reference.build(self.engine, self.library.members, self.null_shift)
        return self._reference

    @property
    def version(self) -> str:
        return hashlib.sha256(json.dumps(self.library.members, sort_keys=True).encode()).hexdigest()

    def admit(self, rows):
        return admission.admit(self.engine, rows, self.reference)

    def accept(self, row, verdict) -> float:
        """Add an admitted candidate at its FP64-verified weight. The verdict must be against the current reference
        (a stale verdict is refused)."""
        proposal = verdict['layers']['G2_performance']
        assert proposal['backend'] == 'cuda-f64', 'an unrefined precision cannot update the portfolio'
        assert verdict['passed'] and proposal['reference'] is self.reference and verdict['expr'] == row['expr']
        self.engine.remember(row['expr'], row['_W'], row['_rr'], self.null_shift)
        self.library.add(row['expr'], proposal['weight'])
        self.changed()
        return proposal['weight']

    def changed(self):
        """The members or weights were edited in place."""
        self._reference = None
