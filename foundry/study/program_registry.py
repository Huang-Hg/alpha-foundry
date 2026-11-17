"""Bounded program leases and explicit integrity-checked deployed artifacts."""
from collections.abc import Sequence
from dataclasses import dataclass, replace
from uuid import uuid4

from foundry.types import (
    Candidate, ContextConflict, ProgramArtifact, ProgramCatalog, ProgramDescriptor,
    ProgramReference, ProgramSource, StrategyProgram,
)


class DeployedPrograms:
    """Immutable deployment catalog; loaders receive verified bytes and identity.

    Implementation semantics are operator-versioned just like native kernels.
    A catalog is supplied explicitly to open/serve and must be supplied again
    when reopening durable references. This class never imports supplied code.
    """
    def __init__(self, artifacts: tuple[ProgramArtifact, ...]):
        if not isinstance(artifacts, tuple) or any(not isinstance(item, ProgramArtifact) for item in artifacts):
            raise TypeError('Deployments require a tuple of ProgramArtifact values')
        if len({item.digest for item in artifacts}) != len(artifacts):
            raise ValueError('Deployed artifact identities must be unique')
        self._artifacts = {item.digest: item for item in artifacts}

    def resolve(self, reference: ProgramReference) -> StrategyProgram:
        if not isinstance(reference, ProgramReference):
            raise TypeError('Deployed programs require an explicit ProgramReference')
        if reference.artifact_digest not in self._artifacts:
            raise ValueError('Program artifact is not deployed in the supplied catalog')
        artifact = self._artifacts[reference.artifact_digest]
        if artifact.digest != reference.artifact_digest:
            raise RuntimeError('Deployed program artifact integrity differs')
        program = artifact.build(artifact.payload, reference.parameters, artifact.digest)
        descriptor = descriptor_for(program)
        if descriptor.artifact_digest != artifact.digest or descriptor.parameters != reference.parameters:
            raise ValueError('Deployed program descriptor differs from its verified artifact or parameters')
        return program


def descriptor_for(program: StrategyProgram) -> ProgramDescriptor:
    if not isinstance(program, StrategyProgram) or not isinstance(program.descriptor, ProgramDescriptor):
        raise TypeError('Programs require a ProgramDescriptor and open(context)')
    return program.descriptor


@dataclass(frozen=True)
class ProgramLease:
    program: StrategyProgram
    sources: tuple[ProgramSource, ...]


class ProgramRegistry:
    """Only active leases own program instances; durable reads use the catalog."""
    def __init__(self, capacity: int, snapshot: str, catalog: ProgramCatalog | None = None):
        self.capacity, self.snapshot, self.catalog = capacity, snapshot, catalog
        self.scope = uuid4().hex
        self.leases: dict[str, ProgramLease] = {}
        self.rows: dict[ProgramSource, StrategyProgram] = {}

    @property
    def count(self) -> int:
        return sum(len(lease.sources) for lease in self.leases.values())

    def register(self, program: StrategyProgram | ProgramReference, *, occupied: int = 0
                 ) -> tuple[str, tuple[Candidate, ...]]:
        if isinstance(program, ProgramReference):
            program = self.resolve(program)
        descriptor = descriptor_for(program)
        if occupied + self.count + descriptor.batch_size > self.capacity:
            raise ValueError('Resident candidate capacity exceeded; release a batch or split the input')
        sources = tuple(ProgramSource(descriptor, index, self.snapshot, self.scope)
                        for index in range(descriptor.batch_size))
        token = 'program-' + uuid4().hex
        self.leases[token] = ProgramLease(program, sources)
        self.rows.update((source, program) for source in sources)
        return token, tuple(Candidate(descriptor.id if descriptor.batch_size == 1 else f'{descriptor.id}:{source.variant}',
                                      source) for source in sources)

    def resolve(self, reference: ProgramReference) -> StrategyProgram:
        """Resolve a deployed reference without acquiring a candidate lease."""
        if not isinstance(reference, ProgramReference):
            raise TypeError('Program resolution requires a typed deployed ProgramReference')
        if self.catalog is None:
            raise ValueError('Program replay requires an explicitly supplied deployed artifact catalog')
        program = self.catalog.resolve(reference)
        descriptor = descriptor_for(program)
        if descriptor.artifact_digest != reference.artifact_digest or descriptor.parameters != reference.parameters:
            raise ValueError('Program catalog returned a different artifact or parameters')
        return program

    def _validate_source(self, source: ProgramSource) -> None:
        if source.snapshot != self.snapshot:
            raise ContextConflict('Program source belongs to a different research snapshot')
        if source.scope and source not in self.rows:
            raise ContextConflict('Program lease ended or belongs to another session')

    def read(self, source: ProgramSource) -> StrategyProgram:
        self._validate_source(source)
        if source.scope:
            program = self.rows[source]
        else:
            program = self.resolve(ProgramReference(source.descriptor.artifact_digest, source.descriptor.parameters))
        if descriptor_for(program) != source.descriptor:
            raise ValueError('Program descriptor differs from the registered replay identity')
        return program

    def read_many(self, sources: Sequence[ProgramSource]) -> dict[ProgramSource, StrategyProgram]:
        """Resolve each durable descriptor once in this operation, validating every reference."""
        for source in sources:
            self._validate_source(source)
        resolved: dict[ProgramDescriptor, StrategyProgram] = {}
        found: dict[ProgramSource, StrategyProgram] = {}
        for source in sources:
            descriptor = source.descriptor
            if source.scope:
                program = self.rows[source]
            else:
                if descriptor not in resolved:
                    resolved[descriptor] = self.resolve(ProgramReference(descriptor.artifact_digest, descriptor.parameters))
                program = resolved[descriptor]
            if descriptor_for(program) != descriptor:
                raise ValueError('Program descriptor differs from the registered replay identity')
            found[source] = program
        return found

    def retain(self, source: ProgramSource) -> ProgramSource:
        self.read(source)
        durable = replace(source, scope='')
        self.read(durable)
        return durable

    def owns(self, token: str) -> bool:
        return token in self.leases

    def release(self, token: str) -> tuple[ProgramSource, ...]:
        if token not in self.leases:
            raise ContextConflict('Program lease is no longer active in this session')
        lease = self.leases.pop(token)
        self.rows = {source: live.program for live in self.leases.values() for source in live.sources}
        return lease.sources

    def close(self) -> None:
        self.rows.clear()
        self.leases.clear()
