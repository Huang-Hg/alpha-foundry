"""Immutable portfolio membership and version identity."""
from dataclasses import dataclass
import hashlib
from foundry.types import Study
from foundry.types import EqualWeight, Source

PROTOCOL = 'research-v20'


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

    def add(self, source: Source, weight: float) -> 'Library':
        if not 0 < weight <= 1:
            raise ValueError('member weight must be in (0, 1]')
        members = self.members or (SEED,)
        if source in {m.source for m in members}:
            members = tuple(Member(m.source, m.weight + weight) if m.source == source else m for m in members)
        else:
            members = members + (Member(source, weight),)
        return Library(members, self.study)

    def allocate(self, source: Source, share: float) -> 'Library':
        """Allocate a convex member share for replayed account portfolios."""
        if not 0 < share <= 1:
            raise ValueError('member allocation share must be in (0, 1]')
        current = self.members or (SEED,)
        values = {member.source: member.weight * (1 - share) for member in current}
        values[source] = values.get(source, 0.) + share
        return Library(tuple(Member(key, value) for key, value in values.items() if value > 0), self.study)
