"""Immutable portfolio membership and version identity."""
from dataclasses import dataclass
import hashlib
from foundry.contracts import Study
from foundry.candidates import EqualWeight, Source

PROTOCOL = 'research-v10'


@dataclass(frozen=True)
class Member:
    source: Source
    weight: float


SEED = Member(EqualWeight(), 1.)


@dataclass(frozen=True)
class Library:
    members: tuple[Member, ...]
    study: Study

    @property
    def version(self):
        return hashlib.sha256(repr((PROTOCOL, self.study, self.members)).encode()).hexdigest()

    def add(self, source, weight):
        assert 0 < weight <= 1
        members = self.members or (SEED,)
        if source in {m.source for m in members}:
            members = tuple(Member(m.source, m.weight + weight) if m.source == source else m for m in members)
        else:
            members = members + (Member(source, weight),)
        return Library(members, self.study)
