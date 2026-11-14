"""Portfolio state and pure successor construction after admission."""
from foundry.portfolio import admission
from foundry.portfolio.library import Library
from foundry.portfolio.reference import Reference


class Portfolio:
    def __init__(self, engine, library: Library, null_shift=False):
        assert engine.study == library.study
        self.engine, self.library, self.null_shift = engine, library, null_shift
        self.reference = Reference.build(engine, library.members, null_shift)
        self.reference.version = library.version

    @property
    def version(self):
        return self.library.version

    def admit(self, rows):
        return admission.admit(self.engine, rows, self.reference)

    def accepted(self, row, verdict):
        proposal = verdict.proposal
        assert verdict.passed and verdict.source == row.source
        assert proposal.reference_version == self.version
        return Portfolio(self.engine, self.library.add(row.source, proposal.weight), self.null_shift)
