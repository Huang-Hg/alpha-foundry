"""Immutable portfolio state and pure membership successor construction."""
from dataclasses import dataclass

from foundry.portfolio.library import Library
from foundry.portfolio.reference import Reference
from foundry.types import EvaluationRow, Verdict


@dataclass(frozen=True)
class Portfolio:
    library: Library
    reference: Reference

    @property
    def version(self) -> str:
        return self.library.version

    def accepted(self, row: EvaluationRow, verdict: Verdict) -> Library:
        """Construct membership after admission; evaluation belongs to the caller."""
        return self.library.add(row.source, verdict.proposal.weight)
