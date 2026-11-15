"""Public research workflow: input normalization, review tokens and resource ownership."""
from collections.abc import Mapping, Sequence
import hashlib

from foundry.candidates import ArrayBatch, Formula
from foundry.research_types import (Candidate, ContextConflict, DataGrid, EvaluationReport,
                                   FamilySeed, RegisteredBatch, WeightReport)
from foundry.session_types import SessionBackend


FormulaInput = str | Sequence[str] | Mapping[str, str]


def formula_candidates(values: FormulaInput) -> tuple[Candidate, ...]:
    if isinstance(values, str):
        values = (values,)
    if isinstance(values, Mapping):
        pairs = tuple(values.items())
    else:
        pairs = tuple((hashlib.sha256(text.encode()).hexdigest(), text) for text in values)
    if not pairs or any(not identity or not isinstance(text, str) or not text.strip() for identity, text in pairs):
        raise ValueError('provide at least one nonempty formula and candidate ID')
    if len({identity for identity, _ in pairs}) != len(pairs):
        raise ValueError('candidate IDs must be unique; use named formulas for repeated expressions')
    return tuple(Candidate(identity, Formula(text)) for identity, text in pairs)


class Session:
    """One owned research session; closing a remote client never stops its server.

    Queries use the cached context. Commands consume the context of the reviewed
    report and do not refresh or retry after a conflict. Call refresh explicitly
    when another writer has changed the portfolio.
    """
    def __init__(self, backend: SessionBackend):
        self._backend = backend
        self._context = backend.context()
        self.grid = backend.grid()

    @property
    def context(self):
        return self._context

    def describe(self):
        """Return the machine-readable fields, operators and execution contract."""
        return self._backend.describe()

    def refresh(self):
        """Explicitly observe a portfolio changed by another writer."""
        self._context = self._backend.refresh()
        self.grid = self._backend.grid()
        return self._context

    def _candidates(self, inputs):
        if isinstance(inputs, RegisteredBatch):
            if inputs.context.snapshot.id != self._context.snapshot.id:
                raise ContextConflict('registered inputs belong to a different data snapshot')
            return inputs.candidates
        return formula_candidates(inputs)

    def evaluate(self, inputs: FormulaInput | RegisteredBatch) -> EvaluationReport:
        return self._backend.evaluate(self._candidates(inputs), self._context.id)

    def evaluate_device(self, inputs: FormulaInput | RegisteredBatch):
        """Return resident arrays in an in-process session, without report serialization."""
        return self._backend.evaluate_device(self._candidates(inputs), self._context.id)

    def commit(self, report: EvaluationReport):
        """Submit the reviewed candidates against exactly the reviewed portfolio."""
        result = self._backend.commit(tuple(Candidate(row.id, row.source) for row in report.rows), report.context.id)
        self._context = result.context
        return result

    def register(self, values, *, kind, ids, grid: DataGrid, producer,
                 observed_through, fit_through) -> RegisteredBatch:
        """Register model outputs; coordinates and information cutoffs remain explicit."""
        batch = ArrayBatch(kind, tuple(ids), values, grid.snapshot, producer, grid.dates, grid.assets,
                           observed_through, fit_through)
        return self._backend.register(batch, self._context.id)

    def explore(self, formulas: FormulaInput, *, hypothesis: str):
        seeds = tuple(FamilySeed(c.id, c.source.expression, hypothesis) for c in formula_candidates(formulas))
        return self._backend.explore(seeds, self._context.id)

    def portfolio(self):
        return self._backend.portfolio(self._context.id)

    def trial_weights(self, weights) -> WeightReport:
        """Evaluate a (trials, members) matrix in portfolio() member order."""
        return self._backend.trial_weights(weights, self._context.id)

    def commit_weights(self, report: WeightReport, index: int):
        """Submit one reviewed weight vector; admission remains authoritative."""
        result = self._backend.commit_weights(report.weights[index], report.context.id)
        self._context = result.context
        return result

    def close(self):
        self._backend.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
