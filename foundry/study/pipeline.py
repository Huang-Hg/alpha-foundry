"""Batched candidate execution over an already assembled, immutable study.

The runtime supplies arrays, operators, execution kernels and resource ownership.
The pipeline borrows resources and bounded caches, and never loads a market.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import replace
from functools import wraps
from types import ModuleType
from typing import (
    TYPE_CHECKING,
    Concatenate,
    ParamSpec,
    Protocol,
    TypeVar,
)

from foundry.numeric.allocation import cap_book

from foundry.config import CudaConfig, Settings, frequency_nanoseconds
from foundry.numeric import metrics as M
from foundry.ops.parse import validate
from foundry.portfolio.allocation import score_to_book
from foundry.study.cache import CandidateCache, MeasurementKey, MemberKey
from foundry.study.inputs import ResidentInputs
from foundry.study.transforms import EXPOSURE, INVALID, delay_shift
from foundry.types import (
    ComputeDevice,
    Array,
    HostArray,
    ArrayBatch,
    ArraySource,
    Backend,
    BoolArray,
    ContextConflict,
    EqualWeight,
    EvaluationRow,
    ExecutionRiskBatch,
    FloatArray,
    Formula,
    IntArray,
    LoadedStudy,
    Metrics,
    NO_INFORMATION,
    ReturnBatch,
    ResourceStatus,
    RiskSummary,
    SimulationResult,
    Source,
    StatsRow,
    StudySnapshot,
    YearMetrics,
)

if TYPE_CHECKING:
    from foundry.ops.ast import Node
    from cupy.cuda import Stream

    from foundry.backtest.cpu import Executor as CashHostExecutor
    from foundry.backtest.cuda import Executor as CashDeviceExecutor
    from foundry.backtest.perpetual_cpu import Executor as PerpetualHostExecutor
    from foundry.backtest.perpetual_cuda import Executor as PerpetualDeviceExecutor
    from foundry.market.a_share import AShare
    from foundry.market.crypto_spot import CryptoSpot
    from foundry.market.panel import Panel
    from foundry.market.perpetual import CryptoPerpetual
    from foundry.market.us_equity import UsEquity
    from foundry.types import ExecutionSpec, PerpetualSpec

    ExecutionBackend = CashHostExecutor | CashDeviceExecutor | PerpetualHostExecutor | PerpetualDeviceExecutor
    StudyData = LoadedStudy[AShare | UsEquity | CryptoSpot | CryptoPerpetual, Panel, ExecutionSpec | PerpetualSpec]

Item = TypeVar('Item')
Result = TypeVar('Result')
Arguments = ParamSpec('Arguments')


class DeviceResources(Protocol):
    """Explicit host/device ownership needed by candidate computation."""
    streams: tuple[nullcontext[None] | Stream, ...]
    placement: ComputeDevice

    def formulas(self, roots: Sequence[Node], fields: Mapping[str, FloatArray],
                 backend: Backend, universe: BoolArray) -> list[FloatArray]: ...

    def scope(self) -> AbstractContextManager[None]: ...
    def start_batch(self) -> None: ...
    def finish_batch(self) -> None: ...
    def synchronize(self) -> None: ...
    def stage_host(self, values: Array) -> HostArray: ...
    def content_leaves(self, values: Array) -> HostArray: ...
    def batch_sizes(self, cfg: CudaConfig, shape: tuple[int, int], events: int,
                    measurements: int) -> tuple[int, int]: ...
    def measurement_budget(self, cfg: CudaConfig) -> int: ...
    def status(self) -> ResourceStatus: ...
    def close(self) -> None: ...


def resource_scope(method: Callable[Concatenate[Pipeline, Arguments], Result]
                   ) -> Callable[Concatenate[Pipeline, Arguments], Result]:
    @wraps(method)
    def scoped(self: Pipeline, *args: Arguments.args, **kwargs: Arguments.kwargs) -> Result:
        with self.resources.scope():
            return method(self, *args, **kwargs)
    return scoped


class Pipeline:
    def __init__(self, loaded: StudyData, backend: Backend, executor: ExecutionBackend,
                 resources: DeviceResources, settings: Settings, context: StudySnapshot):
        self.study, self.market, self.panel = loaded.study, loaded.market, loaded.panel
        self.allocation = self.market.allocation
        self.backend, self.executor, self.resources = backend, executor, resources
        self.xp, self.to_host = backend.xp, backend.to_host
        self.settings = settings
        self.cuda_cfg, self.portfolio_cfg, self.expr_cfg = settings.cuda, settings.portfolio, settings.expr
        self.context = context
        self.fields = {k: self.panel[k] for k in self.panel.vocabulary}
        self.vocabulary = self.panel.vocabulary
        self.measurement_times = loaded.window.measurement_times
        self.measurement_start = loaded.window.measurement_start
        self.decision_start = loaded.window.decision_start
        self.ppy = loaded.window.periods_per_year
        self.schedule = loaded.host_spec.schedule
        self.decision_ppy = self.ppy * (frequency_nanoseconds(settings.clock.statistics)
                                        / frequency_nanoseconds(settings.clock.decision))
        self.clock = loaded.window.clock
        self.decision_times = self.clock.decision_times
        self.decision_ns = self.xp.asarray(self.clock.decision_ns)
        self.universe = self.market.universe(self.panel)
        self.slots = self.xp.maximum(self.universe.sum(1, keepdims=True), 1).astype(self.xp.float64)
        self.spec = executor.device_spec(loaded.host_spec)
        measurement_times = self.measurement_times[self.measurement_start:]
        self.years = M.Years(measurement_times, self.xp)
        ends = self.measurement_times.astype('datetime64[ns]').astype('int64') + frequency_nanoseconds(settings.clock.statistics)
        sample = self.clock.decision_ns.searchsorted(ends, side='right') - 1
        self.measurement_targets = self.xp.asarray(sample[self.measurement_start:])
        self.recency_weights = M.recency(measurement_times, settings.portfolio.recent_halflife_days, self.xp)
        self.top_periods = max(1, int(len(measurement_times) * 0.01))
        self.candidate_batch, self.trial_batch = resources.batch_sizes(settings.cuda, self.panel.shape,
            len(self.schedule.event_ns), self.schedule.measurement_count)
        self.commit_batch = self.candidate_batch if executor.parallel_trials else 1
        self.precision = context.precision.effective
        self.cache = CandidateCache(self.candidate_batch, settings.portfolio.max_members,
                                    resources.measurement_budget(settings.cuda), self.schedule.measurement_count)
        self.inputs = ResidentInputs(self.candidate_batch, self.xp, resources.content_leaves)

    def batches(self, items: Sequence[Item]) -> Iterator[Sequence[Item]]:
        """Apply the runtime's resident candidate bound at every batch boundary."""
        for lo in range(0, len(items), self.candidate_batch):
            yield items[lo:lo + self.candidate_batch]

    @resource_scope
    def validate_batch(self, batch: ArrayBatch) -> None:
        """Check external arrays once, before publishing any immutable artifact."""
        xp = self.xp
        if batch.kind not in ('scores', 'targets'):
            raise ValueError('batch kind must be scores or targets')
        if batch.kind == 'scores' and batch.unit != 'score':
            raise ValueError('scores require unit=score')
        if batch.kind == 'targets' and batch.unit not in ('equity_fraction', 'quantity', 'notional'):
            raise ValueError('targets require equity_fraction, quantity or notional units')
        if batch.kind == 'targets' and batch.unit != 'equity_fraction' and self.settings.execution.model != 'orders':
            raise ValueError('this execution plan supports only equity_fraction targets; quantity/notional require account conversion')
        if batch.snapshot != self.context.id:
            raise ContextConflict('array batch belongs to a different snapshot')
        if batch.decision_times != self.decision_times or batch.assets != self.panel.assets:
            raise ValueError('batch decision_times/assets must exactly match the ordered study axes')
        if not batch.ids or len(set(batch.ids)) != len(batch.ids) or not all(isinstance(i, str) and i for i in batch.ids):
            raise ValueError('batch candidate IDs must be nonempty and unique')
        if not isinstance(batch.producer, str) or not batch.producer:
            raise ValueError('batch producer must name an immutable model/program version')
        if batch.complexity is not None and (len(batch.complexity) != len(batch.ids)
                or any(not 0 <= value <= 1 for value in batch.complexity)):
            raise ValueError('complexity must contain one normalized value in [0,1] per candidate')
        values = batch.values
        B, D, S = len(batch.ids), *self.panel.shape
        if values.shape != (B, D, S) or values.dtype not in (xp.dtype('float32'), xp.dtype('float64')):
            raise ValueError('batch values must be float32/float64 with shape (B,D,S)')
        for name, cutoff in (('observed_at', batch.observed_at), ('fit_at', batch.fit_at)):
            if cutoff.shape != (B, D) or cutoff.dtype != xp.dtype('int64'):
                raise ValueError(f'{name} must be int64 with shape (B,D)')
        if batch.update_mask is not None:
            if batch.kind == 'scores':
                raise ValueError('update_mask is defined only for targets')
            if batch.update_mask.shape != (B, D, S) or batch.update_mask.dtype != xp.dtype('bool'):
                raise ValueError('update_mask must be bool with shape (B,D,S)')
            if self.settings.execution.model != 'orders' and not bool(xp.asarray(batch.update_mask).all()):
                raise ValueError('this execution plan requires a new target at every decision; update_mask=False requires account-state execution')
        for lo in range(0, B, self.candidate_batch):
            hi = lo + self.candidate_batch
            for name, cutoffs in (('observed_at', batch.observed_at), ('fit_at', batch.fit_at)):
                cutoff = xp.asarray(cutoffs[lo:hi])
                if bool(((cutoff != NO_INFORMATION) & (cutoff > self.decision_ns[None, :])).any()):
                    raise ValueError(f'{name} contains information later than its decision time')
            part = xp.asarray(values[lo:hi])
            if batch.kind == 'scores':
                if bool(xp.isinf(part).any()):
                    raise ValueError('scores may contain NaN, but not infinity')
            else:
                if not bool(xp.isfinite(part).all()):
                    raise ValueError('target allocation must be finite')
                if not self.allocation.allow_short and bool((part < 0).any()):
                    raise ValueError('cash targets must be nonnegative')
                if batch.unit == 'equity_fraction':
                    gross = (xp.abs(part) if self.allocation.allow_short else part).sum(2, dtype=xp.float64)
                    if bool((gross > self.allocation.max_gross + self.portfolio_cfg.weight_sum_atol).any()):
                        raise ValueError('target allocation exceeds the market gross exposure limit')
                if bool(((part != 0) & ~self.universe).any()):
                    raise ValueError('target allocation must be zero for ineligible assets')

    @resource_scope
    def register_arrays(self, sources: Sequence[ArraySource], values: FloatArray, *, owned: bool = False) -> None:
        """Own the device copy; no IO or content hashing inside computation."""
        resident = values if owned else self.xp.array(values, copy=True, order='C')
        self.cache.arrays.put(sources, resident)
        self.resources.synchronize()

    def has_array(self, source: ArraySource) -> bool:
        if source in self.cache.arrays:
            self.cache.arrays[source]
            return True
        return False

    @resource_scope
    def evaluate(self, exprs: Sequence[str], report: bool = True, *, delayed: bool = False) -> list[EvaluationRow]:
        """Evaluate formula batches through the same bounded source pipeline."""
        return self.evaluate_sources([Formula(text) for text in exprs], report, delayed=delayed)

    def _evaluate_formulas(self, exprs: Sequence[str], report: bool, delayed: bool,
                           saved: Mapping[Source, EvaluationRow]) -> list[EvaluationRow]:
        """Evaluate candidates without publishing state; device books/returns belong to this runtime."""
        rows, ids, books, finite = self.formula_intents(exprs, delayed=delayed)
        if not ids:
            return rows
        return self._measure(rows, ids, books, finite, report, saved,
                             score_budget=self.settings.sizing.rule == 'independent')

    @resource_scope
    def formula_intents(self, exprs: Sequence[str], *, delayed: bool = False
                        ) -> tuple[list[EvaluationRow], list[int], FloatArray, BoolArray]:
        """Compile formula scores to owned (B,D,S) targets without simulating accounts."""
        xp, precision = self.xp, self.precision
        rows, ids = [], []
        for expr in exprs:
            text = expr.strip()
            parsed = validate(text, self.vocabulary, self.expr_cfg.max_cost)
            row = EvaluationRow(Formula(text), parsed['ok'], parsed['error'], precision)
            rows.append(row)
            if not row.ok:
                continue
            row.tree = parsed['tree']
            why = self.backend.unsupported(row.tree)
            if why:
                raise RuntimeError(f'{self.backend.name} operator implementation required: {why}: {text}')
            ids.append(len(rows) - 1)
        if not ids:
            return rows, ids, xp.empty((0, *self.panel.shape)), xp.empty(0, dtype=xp.bool_)
        values = xp.stack(self.resources.formulas([rows[i].tree.root for i in ids],
                          self.fields, self.backend, self.universe))
        if delayed:
            values = delay_shift(values, [rows[i].source.expression for i in ids], xp, xp.nan)
        finite = (~xp.isinf(values).any(axis=(1, 2))) & xp.isfinite(values).any(axis=(1, 2))
        books = self.score_to_book(values)
        return rows, ids, books, finite

    def _measure(self, rows: list[EvaluationRow], ids: Sequence[int], books: FloatArray,
                 finite: BoolArray, report: bool, saved: Mapping[Source, EvaluationRow], *,
                 score_budget: bool = False) -> list[EvaluationRow]:
        xp = self.xp
        errors: dict[int, str] = {}
        if score_budget:
            within_budget = (xp.abs(books).sum(axis=2, dtype=xp.float64)
                             <= self.allocation.max_gross + self.portfolio_cfg.weight_sum_atol).all(axis=1)
            invalid_budget = self.to_host(~within_budget).tolist()
            errors = {i: EXPOSURE for i, invalid in zip(ids, invalid_budget) if invalid}
            if errors:
                finite = finite & within_budget
                books = xp.where(within_budget[:, None, None], books, 0.)
        if report:
            # These books are owned by this batch; match the execution prefix
            # even when cached returns avoid another account simulation.
            books[:, :self.decision_start] = 0
            pending = []
            for j, i in enumerate(ids):
                row = rows[i]
                if row.source in saved:
                    rows[i] = replace(saved[row.source], tree=row.tree, book=books[j])
                else:
                    pending.append((j, i))
            if not pending:
                return rows
            positions, indices = zip(*pending)
            selected = books if len(pending) == len(ids) else books[list(positions)]
            good_rows = self.to_host(finite[list(positions)]).tolist()
            for i, stats, good in zip(indices, self.metrics(selected), good_rows):
                row = rows[i]
                row.ok, row.error = good, None if good else errors.get(i, INVALID)
                row.book, row.returns = stats.book, stats.returns
                row.metrics, row.valid_returns = stats.metrics, stats.valid_returns
                row.risk = stats.risk
                row.maxdd = stats.metrics.maxdd
            return rows
        result = self.returns(books)
        status = self.to_host(xp.stack((finite, result.valid), axis=1)).tolist()
        risks = self._risk_summaries(result.risk, len(result.books))
        for j, (i, (good, fine)) in enumerate(zip(ids, status)):
            row = rows[i]
            row.ok, row.error = good, None if good else errors.get(i, INVALID)
            row.book, row.returns, row.maxdd, row.valid_returns = result.books[j], result.returns[j], result.maxdd[j], fine
            row.risk = risks[j]
        return rows

    @resource_scope
    def evaluate_sources(self, sources: Sequence[Source], report: bool = True, *,
                         delayed: bool = False) -> list[EvaluationRow]:
        """Snapshot-scoped measurements; returned buffers are borrowed read-only.

        Each caller gets its own mutable row. Portfolio verdicts are recomputed.
        """
        return [row for part in self.batches(sources) for row in self._evaluate_batch(part, report, delayed)]

    def _evaluate_batch(self, sources: Sequence[Source], report: bool, delayed: bool) -> list[EvaluationRow]:
        precision = self.precision
        if not report:
            return self._compute_sources(sources, report, precision, delayed, {})
        keys = [MeasurementKey(self.context.id, source, precision, delayed) for source in sources]
        unique = list(dict.fromkeys(keys))
        saved = {key.source: self.cache.reports[key] for key in unique if key in self.cache.reports}
        computed = self._compute_sources([key.source for key in unique], True, precision, delayed, saved)
        found = dict(zip(unique, computed))
        missing = [key for key in unique if key.source not in saved]
        if missing:
            self.cache.reports.put(missing, [found[key] for key in missing])
        return [replace(found[key]) for key in keys]

    def _compute_sources(self, sources: Sequence[Source], report: bool, precision: str,
                         delayed: bool, saved: Mapping[Source, EvaluationRow]) -> list[EvaluationRow]:
        """Every source becomes a book before the same execution/metrics path."""
        rows = [EvaluationRow(source, True, None, precision) for source in sources]
        formulas = [i for i, source in enumerate(sources) if isinstance(source, Formula)]
        if formulas:
            computed = self._evaluate_formulas([sources[i].expression for i in formulas], report, delayed, saved)
            for i, row in zip(formulas, computed):
                rows[i] = row
        for kind in ('scores', 'targets'):
            ids = [i for i, source in enumerate(sources) if isinstance(source, ArraySource) and source.kind == kind]
            if not ids:
                continue
            values = self.xp.stack([self.cache.arrays[sources[i]] for i in ids]).astype(self.xp.float64)
            if delayed:
                # Full-array hashes include later values: using them as a noise
                # clock would let a future edit change historical delays.
                values = delay_shift(values, [sources[i].producer + ':' + kind for i in ids],
                                     self.xp, self.xp.nan if kind == 'scores' else 0.)
            if kind == 'scores':
                finite = self.xp.isfinite(values).any(axis=(1, 2))
                books = self.score_to_book(values)
            else:
                finite = self.xp.ones(len(ids), dtype=bool)
                books = cap_book(self.xp.where(self.universe, values, 0.), self.allocation, self.xp)
            self._measure(rows, ids, books, finite, report, saved,
                          score_budget=kind == 'scores' and self.settings.sizing.rule == 'independent')
        ids = [i for i, source in enumerate(sources) if isinstance(source, EqualWeight)]
        if ids:
            self._measure(rows, ids, self.xp.stack([self.equal_weight()] * len(ids)),
                          self.xp.ones(len(ids), dtype=bool), report, saved)
        return rows

    @resource_scope
    def members(self, sources: Sequence[Source], *, delayed: bool = False) -> list[tuple[FloatArray, FloatArray]]:
        """Borrow books and returns within this pipeline's immutable study snapshot."""
        keys = [MemberKey(source, delayed) for source in sources]
        found = {key: self.cache.members[key] for key in keys if key in self.cache.members}
        missing = list(dict.fromkeys(key for key in keys if key not in found))
        for part in self.batches(missing):
            for key, row in zip(part, self.evaluate_sources([key.source for key in part], delayed=delayed)):
                source = key.source
                if not row.ok:
                    raise ValueError(f'portfolio member cannot be evaluated: {source!r}: {row.error}')
                found[key] = (row.book.copy(), row.returns.copy())
        for key in keys:
            self.cache.members[key] = found[key]
        return [found[key] for key in keys]

    @resource_scope
    def retain_member(self, row: EvaluationRow, *, delayed: bool = False) -> None:
        """Keep an admitted book while its raw input can leave the bounded cache."""
        self.cache.members[MemberKey(row.source, delayed)] = (row.book.copy(), row.returns.copy())

    @resource_scope
    def score_to_book(self, values: FloatArray) -> FloatArray:
        return score_to_book(values, self.universe, self.slots, self.allocation, self.settings.sizing, self.xp)

    @resource_scope
    def equal_weight(self) -> FloatArray:
        """Use the equal-slot baseline budget independently of score sizing."""
        if self.allocation.max_gross >= 1.:
            return self.universe / self.slots
        return self.universe * self.allocation.max_gross / self.slots

    @resource_scope
    def simulate(self, W: FloatArray) -> SimulationResult:
        return self.executor.simulate(W, self.spec, self.measurement_start, self.ppy, self.precision)

    @resource_scope
    def blend(self, W: FloatArray, U: FloatArray, owners: IntArray, weights: FloatArray) -> SimulationResult:
        return self.executor.blend(W, U, owners, weights, self.spec, self.measurement_start, self.ppy, self.precision, self.trial_batch)

    @resource_scope
    def fingerprints(self, W: FloatArray) -> list[int]:
        return self.executor.fingerprints(W)

    def _stack(self, books: FloatArray | Sequence[FloatArray]) -> FloatArray:
        xp = self.xp
        W = xp.ascontiguousarray(books if isinstance(books, xp.ndarray)
                                else xp.stack([xp.asarray(b, dtype=xp.float64) for b in books]), dtype=xp.float64).copy()
        W[:, :self.decision_start] = 0
        return W

    @resource_scope
    def returns(self, books: FloatArray | Sequence[FloatArray]) -> ReturnBatch:
        W = self._stack(books)
        result = self.simulate(W)
        rt = result.returns[:, self.measurement_start:]
        return ReturnBatch(W, rt, M.sound(rt, self.xp), result.statistics[:, 3], result.risk)

    def _risk_summaries(self, risk: ExecutionRiskBatch | None, size: int) -> list[RiskSummary | None]:
        if risk is None:
            return [None] * size
        xp = self.xp
        scalars = self.to_host(xp.stack((risk.funding[:, self.measurement_start:].sum(1),
            risk.trading_fees[:, self.measurement_start:].sum(1), risk.liquidation_fees[:, self.measurement_start:].sum(1),
            risk.isolated_loss_cap[:, self.measurement_start:].sum(1), risk.ending_equity, risk.max_gross_leverage), axis=1)).tolist()
        counts = self.to_host(risk.liquidations[:, self.measurement_start:].sum(1)).tolist()
        timestamps = self.to_host(risk.first_liquidation_ns).tolist()
        bankrupt = self.to_host(risk.bankrupt).tolist()
        return [RiskSummary(risk.model, v[0], v[1], v[2], v[3], int(count), int(timestamp), bool(failed), v[4], v[5],
                            gross_sampling=risk.gross_sampling)
                for v, count, timestamp, failed in zip(scalars, counts, timestamps, bankrupt)]

    @resource_scope
    def metrics(self, books: FloatArray | Sequence[FloatArray], result: SimulationResult | None = None) -> list[StatsRow]:
        """Compute one semantic report; transfer only report scalars to the host."""
        xp = self.xp
        W = self._stack(books)
        result = self.simulate(W) if result is None else result
        rr, st = result.returns, result.statistics
        rt, wt = rr[:, self.measurement_start:], W[:, self.decision_start:]
        lg = xp.log1p(rt)
        target_gross = xp.abs(W).sum(2)
        gross = xp.where(self.measurement_targets[None, :] >= 0,
                         target_gross[:, xp.maximum(self.measurement_targets, 0)], 0.)
        turnover = (xp.abs(xp.diff(wt, axis=1)).sum(2).mean(1)
                    if wt.shape[1] > 1 else xp.zeros(len(W)))
        ann = st[:, 1]
        sh = xp.where(st[:, 2] > 0, st[:, 0], 0)
        total = rt.sum(1)
        concentration = xp.where(total > 0, xp.sort(rt, axis=1)[:, -self.top_periods:].sum(1)
                                 / xp.where(total > 0, total, 1), xp.inf)
        vals = xp.stack((sh, ann, st[:, 3], self.backend.measure(rt, self.recency_weights, self.ppy).cagr, st[:, 4], turnover,
                         gross.mean(1), ann / xp.maximum(turnover * self.decision_ppy, 1e-12) * 1e4,
                         xp.abs(wt).max(axis=(1, 2)), concentration), axis=1)
        yearly = xp.stack((self.years.sharpe(rt, self.ppy), self.years.share(gross > 1e-9), self.years.total(lg)), axis=2)
        host = self.to_host(xp.concatenate((vals, yearly.reshape(len(W), -1), M.sound(rt, xp)[:, None]), axis=1)).tolist()
        dates = self.measurement_times[self.measurement_start:]
        risks = self._risk_summaries(result.risk, len(W))
        return [StatsRow(W[i], rt[i], Metrics(*v[:10],
                    yearly=tuple(YearMetrics(int(year), int(days), round(v[10 + j * 3], 3),
                                             round(v[11 + j * 3], 3), round(v[12 + j * 3], 4))
                                 for j, (year, days) in enumerate(zip(self.years.years, self.years.periods))),
                    span=f'{dates[0]}..{dates[-1]}', n_periods=len(dates)), bool(v[-1]), risks[i])
                for i, v in enumerate(host)]

    def close(self) -> None:
        """Release this study's caches; runtime closes its execution resources."""
        self.cache.close()
        self.inputs.close()
        self.fields.clear()
        del self.spec, self.universe, self.slots, self.years, self.decision_ns, self.measurement_targets
