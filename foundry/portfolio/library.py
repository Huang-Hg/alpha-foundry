"""The portfolio library: the members of the production portfolio and their weights, on disk.

A member is a formula; its weight in (0,1] is the multiple its book is ADDED to the portfolio with (protocol v6),
not a share of a total. Writes are compare-and-swap against the file as it was loaded, so two writers cannot
silently overwrite each other; a long-running service reloads the file when its mtime moves.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List

from foundry.config import ROOT
from foundry.core.files import publish

STORE = ROOT / 'production' / 'library.json'
# The admission rules a library is built and judged under (foundry.portfolio.admission). A change of protocol means
# re-mining into a new output directory; services and clients refuse to talk across protocols.
PROTOCOL = 'combo-cuda-v6'
# The full-exposure equal-weight book over the universe: the reference of an empty library, and a member once the
# first candidate is admitted against it (the candidate was judged as an addition to it).
SEED = dict(expr='__equal_weight__', weight=1.0)


class Library:
    def __init__(self, members: List[Dict], path: Path = STORE):
        self.members = [dict(expr=m['expr'], weight=float(m['weight'])) for m in members]
        self.path = path
        self._snapshot = None

    @classmethod
    def load(cls, path: Path = STORE) -> 'Library':
        content = path.read_bytes() if path.exists() else b''
        lib = cls(json.loads(content)['members'] if content else [], path)
        lib._snapshot = content
        return lib

    def save(self) -> None:
        lock = self.path.with_suffix('.lock')
        os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        try:
            current = self.path.read_bytes() if self.path.exists() else b''
            if self._snapshot is not None and current != self._snapshot:
                raise RuntimeError('Portfolio changed on disk; refusing a stale update')
            content = json.dumps(dict(updated=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), members=self.members),
                                 ensure_ascii=False, indent=1).encode('utf-8')
            publish(self.path, content)
            self._snapshot = content
        finally:
            lock.unlink()

    def add(self, expr: str, weight: float) -> None:
        """Existing members keep their weights; a formula already in the portfolio gets the weight added."""
        assert 0 < weight <= 1
        self.members = self.members or [dict(SEED)]
        for m in self.members:
            if m['expr'] == expr:
                m['weight'] += weight
                return
        self.members.append(dict(expr=expr, weight=weight))
