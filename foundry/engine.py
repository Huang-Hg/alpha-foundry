"""Batched candidate execution over an already assembled, immutable study.

The runtime supplies arrays, operators, execution kernels and resource ownership.
The engine owns expression/member caches and never selects or loads a market.
"""
from __future__ import annotations

from functools import wraps
from collections import OrderedDict
from dataclasses import replace
import hashlib

from foundry.contracts import Backend, LoadedStudy, ReturnBatch, StudySnapshot
from foundry.candidates import ArraySource, EqualWeight, Formula
from foundry.core import metrics as M
from foundry.expr.evaluate import evaluate
from foundry.expr.parse import validate
from foundry.research_types import ContextConflict, EvaluationRow, Metrics, RiskSummary, StatsRow, YearMetrics
from foundry.portfolio.allocation import cap_book, signal_to_book


NULL_LAG = (365, 1095)
INVALID = 'G1 数值无效:含无穷值或全为缺失'


def null_lag(expr: str) -> int:
    """Deterministic forward-only noise control; the sample end never wraps."""
    lo, hi = NULL_LAG
    return lo + int.from_bytes(hashlib.sha256(expr.encode()).digest()[:8], 'big') % (hi - lo + 1)


def resource_scope(method):
    @wraps(method)
    def scoped(self, *args, **kwargs):
        with self.resources.scope():
            return method(self, *args, **kwargs)
    return scoped


class Engine:
    def __init__(self, loaded: LoadedStudy, backend: Backend, executor, resources,
                 settings, context: StudySnapshot):
        self.study, self.market, self.panel = loaded.study, loaded.market, loaded.panel
        self.allocation = self.market.allocation
        self.backend, self.executor, self.resources = backend, executor, resources
        self.xp, self.to_host = backend.xp, backend.to_host
        self.settings = settings
        self.cuda_cfg, self.portfolio_cfg, self.expr_cfg = settings.cuda, settings.portfolio, settings.expr
        self.context = context
        self.fields = {k: self.panel[k] for k in self.panel.vocabulary}
        self.vocabulary = self.panel.vocabulary
        self.dates, self.start, self.ppy = loaded.window.dates, loaded.window.start, self.market.ppy
        self.universe = self.market.universe(self.panel)
        self.slots = self.xp.maximum(self.universe.sum(1, keepdims=True), 1).astype(self.xp.float64)
        self.spec = executor.device_spec(loaded.host_spec)
        self.years = M.Years(loaded.window.stat_dates, self.xp)
        self.recency_weights = M.recency(loaded.window.stat_dates, settings.portfolio.recent_halflife_days, self.xp)
        self.top_days = max(1, int((len(self.dates) - self.start) * 0.01))
        self.candidate_batch, self.trial_batch = resources.batch_sizes(settings.cuda, self.panel.shape)
        self.commit_batch = self.candidate_batch if executor.parallel_trials else 1
        self.precision = context.precision.effective
        self._members: dict[tuple, tuple] = {}
        self._arrays: OrderedDict[ArraySource, object] = OrderedDict()
        self._reports = OrderedDict()

    @resource_scope
    def validate_batch(self, batch):
        """Check external arrays once, before publishing any immutable artifact."""
        xp = self.xp
        if batch.kind not in ('signals', 'target_weights', 'target_exposure'):
            raise ValueError('batch kind must be signals, target_weights or target_exposure')
        if batch.kind == 'target_weights' and self.allocation.allow_short:
            raise ValueError('derivatives require explicit target_exposure, not cash target_weights')
        if batch.snapshot != self.context.id:
            raise ContextConflict('array batch belongs to a different snapshot')
        if batch.dates != tuple(str(d) for d in self.dates) or batch.assets != self.panel.assets:
            raise ValueError('batch dates/assets must exactly match the ordered study axes')
        if not batch.ids or len(set(batch.ids)) != len(batch.ids) or not all(isinstance(i, str) and i for i in batch.ids):
            raise ValueError('batch candidate IDs must be nonempty and unique')
        if not isinstance(batch.producer, str) or not batch.producer:
            raise ValueError('batch producer must name an immutable model/program version')
        values = batch.values
        B, T, S = len(batch.ids), *self.panel.shape
        if values.shape != (B, T, S) or values.dtype not in (xp.dtype('float32'), xp.dtype('float64')):
            raise ValueError('batch values must be float32/float64 with shape (B,T,S)')
        for name, cutoff in (('observed_through', batch.observed_through), ('fit_through', batch.fit_through)):
            if cutoff.shape != (B, T) or cutoff.dtype != xp.dtype('int64'):
                raise ValueError(f'{name} must be int64 with shape (B,T)')
        for lo in range(0, B, self.candidate_batch):
            hi = lo + self.candidate_batch
            for name, cutoffs in (('observed_through', batch.observed_through), ('fit_through', batch.fit_through)):
                cutoff = xp.asarray(cutoffs[lo:hi])
                if bool(((cutoff < -1) | (cutoff > xp.arange(T)[None, :])).any()):
                    raise ValueError(f'{name} contains a future cutoff or an index below -1')
            part = xp.asarray(values[lo:hi])
            if batch.kind == 'signals':
                if bool(xp.isinf(part).any()):
                    raise ValueError('signals may contain NaN, but not infinity')
            else:
                if not bool(xp.isfinite(part).all()):
                    raise ValueError('target allocation must be finite')
                if not self.allocation.allow_short and bool((part < 0).any()):
                    raise ValueError('cash target weights/exposures must be nonnegative')
                gross = (xp.abs(part) if self.allocation.allow_short else part).sum(2, dtype=xp.float64)
                if bool((gross > self.allocation.max_gross + self.portfolio_cfg.weight_sum_atol).any()):
                    raise ValueError('target allocation exceeds the market gross exposure limit')
                if bool(((part != 0) & ~self.universe).any()):
                    raise ValueError('target allocation must be zero for ineligible assets')

    @resource_scope
    def register_arrays(self, sources, values):
        """Own the device copy; no IO or content hashing inside computation."""
        for source, value in zip(sources, values):
            self._arrays[source] = self.xp.ascontiguousarray(self.xp.asarray(value)).copy()
            self._arrays.move_to_end(source)
            if len(self._arrays) > self.candidate_batch:
                self._arrays.popitem(last=False)
        self.resources.synchronize()

    def has_array(self, source):
        if source in self._arrays:
            self._arrays.move_to_end(source)
            return True
        return False

    @resource_scope
    def evaluate(self, exprs: list[str], report=True, null_shift=False) -> list[EvaluationRow]:
        """Evaluate candidates without publishing state; device books/returns belong to this runtime."""
        xp, precision = self.xp, self.precision
        self.resources.start_batch()
        streams = self.resources.streams
        caches = [{} for _ in streams]
        rows, ids, values = [], [], []
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
            k = len(ids) % len(streams)
            with streams[k]:
                values.append(evaluate(row.tree.root, self.fields, self.backend, caches[k], self.universe))
            ids.append(len(rows) - 1)
        self.resources.finish_batch()
        if not ids:
            return rows
        values = xp.stack(values)
        if null_shift:
            values = self._delayed(values, [rows[i].source.expression for i in ids], xp.nan)
        finite = (~xp.isinf(values).any(axis=(1, 2))) & xp.isfinite(values).any(axis=(1, 2))
        books = self.signal_to_book(values)
        return self._measure(rows, ids, books, finite, report, precision)

    def _delayed(self, values, identities, missing):
        xp = self.xp
        lag = xp.asarray([null_lag(identity) for identity in identities])
        at = xp.arange(values.shape[1])[None, :] - lag[:, None]
        return xp.where((at >= 0)[:, :, None],
                        values[xp.arange(len(identities))[:, None], xp.maximum(at, 0)], missing)

    def _measure(self, rows, ids, books, finite, report, precision):
        xp = self.xp
        if report:
            for i, stats, good in zip(ids, self.metrics(books), self.to_host(finite).tolist()):
                row = rows[i]
                row.ok, row.error = good, None if good else INVALID
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
            row.ok, row.error = good, None if good else INVALID
            row.book, row.returns, row.maxdd, row.valid_returns = result.books[j], result.returns[j], result.maxdd[j], fine
            row.risk = risks[j]
        return rows

    @resource_scope
    def evaluate_sources(self, sources, report=True, null_shift=False):
        """Snapshot-scoped measurements; returned buffers are borrowed read-only.

        Each caller gets its own mutable row. Portfolio verdicts are recomputed.
        """
        precision = self.precision
        if not report:
            return self._compute_sources(sources, report, precision, null_shift)
        keys = [(self.context.id, source, precision, null_shift) for source in sources]
        found = {key: self._reports[key] for key in keys if key in self._reports}
        missing = list(dict.fromkeys(key for key in keys if key not in found))
        if missing:
            computed = self._compute_sources([key[1] for key in missing], True, precision, null_shift)
            found.update(zip(missing, computed))
        for key in keys:
            self._reports[key] = found[key]
            self._reports.move_to_end(key)
            if len(self._reports) > self.candidate_batch:
                self._reports.popitem(last=False)
        return [replace(found[key]) for key in keys]

    def _compute_sources(self, sources, report, precision, null_shift):
        """Every source becomes a book before the same execution/metrics path."""
        rows = [EvaluationRow(source, True, None, precision) for source in sources]
        formulas = [i for i, source in enumerate(sources) if isinstance(source, Formula)]
        if formulas:
            computed = self.evaluate([sources[i].expression for i in formulas], report, null_shift)
            for i, row in zip(formulas, computed):
                rows[i] = row
        for kind in ('signals', 'target_weights', 'target_exposure'):
            ids = [i for i, source in enumerate(sources) if isinstance(source, ArraySource) and source.kind == kind]
            if not ids:
                continue
            values = self.xp.stack([self._arrays[sources[i]] for i in ids]).astype(self.xp.float64)
            if null_shift:
                # Full-array hashes include later values: using them as a noise
                # clock would let a future edit change historical delays.
                values = self._delayed(values, [sources[i].producer + ':' + kind for i in ids],
                                       self.xp.nan if kind == 'signals' else 0.)
            if kind == 'signals':
                finite = self.xp.isfinite(values).any(axis=(1, 2))
                books = self.signal_to_book(values)
            else:
                finite = self.xp.ones(len(ids), dtype=bool)
                books = cap_book(self.xp.where(self.universe, values, 0.), self.allocation, self.xp)
            self._measure(rows, ids, books, finite, report, precision)
        ids = [i for i, source in enumerate(sources) if isinstance(source, EqualWeight)]
        if ids:
            self._measure(rows, ids, self.xp.stack([self.equal_weight()] * len(ids)),
                          self.xp.ones(len(ids), dtype=bool), report, precision)
        return rows

    def members(self, sources, null_shift=False) -> list[tuple]:
        """Books and returns cached within this engine's study snapshot only."""
        missing = list(dict.fromkeys(source for source in sources if (source, null_shift) not in self._members))
        for lo in range(0, len(missing), self.candidate_batch):
            part = missing[lo:lo + self.candidate_batch]
            for source, row in zip(part, self.evaluate_sources(part, null_shift=null_shift)):
                assert row.ok, (source, row.error)
                self._members[(source, null_shift)] = (row.book, row.returns)
        return [self._members[(source, null_shift)] for source in sources]

    def retain_member(self, row, null_shift):
        """Keep an admitted book while its raw input can leave the bounded cache."""
        self._members[(row.source, null_shift)] = (row.book, row.returns)

    @resource_scope
    def signal_to_book(self, values):
        return signal_to_book(values, self.universe, self.slots, self.allocation, self.xp)

    @resource_scope
    def equal_weight(self):
        return self.universe / self.slots

    @resource_scope
    def simulate(self, W):
        return self.executor.simulate(W, self.spec, self.start, self.ppy, self.precision)

    @resource_scope
    def blend(self, W, U, owners, weights):
        return self.executor.blend(W, U, owners, weights, self.spec, self.start, self.ppy, self.precision, self.trial_batch)

    @resource_scope
    def fingerprints(self, W):
        return self.executor.fingerprints(W)

    def _stack(self, books):
        xp = self.xp
        W = xp.ascontiguousarray(books if isinstance(books, xp.ndarray)
                                else xp.stack([xp.asarray(b, dtype=xp.float64) for b in books]), dtype=xp.float64).copy()
        W[:, :self.start] = 0
        return W

    @resource_scope
    def returns(self, books):
        W = self._stack(books)
        result = self.simulate(W)
        rt = result.returns[:, self.start:]
        return ReturnBatch(W, rt, M.sound(rt, self.xp), result.statistics[:, 3], result.risk)

    def _risk_summaries(self, risk, size):
        if risk is None:
            return [None] * size
        xp = self.xp
        scalars = self.to_host(xp.stack((risk.funding[:, self.start:].sum(1),
            risk.trading_fees[:, self.start:].sum(1), risk.liquidation_fees[:, self.start:].sum(1),
            risk.isolated_loss_cap[:, self.start:].sum(1), risk.ending_equity, risk.max_gross_leverage), axis=1)).tolist()
        counts = self.to_host(risk.liquidations[:, self.start:].sum(1)).tolist()
        timestamps = self.to_host(risk.first_liquidation_ms).tolist()
        bankrupt = self.to_host(risk.bankrupt).tolist()
        return [RiskSummary(risk.model, v[0], v[1], v[2], v[3], int(count), int(timestamp), bool(failed), v[4], v[5])
                for v, count, timestamp, failed in zip(scalars, counts, timestamps, bankrupt)]

    @resource_scope
    def metrics(self, books, result=None) -> list[StatsRow]:
        """Compute one semantic report; transfer only report scalars to the host."""
        xp = self.xp
        W = self._stack(books)
        result = self.simulate(W) if result is None else result
        rr, st = result.returns, result.statistics
        rt, wt = rr[:, self.start:], W[:, self.start:]
        lg = xp.log1p(rt)
        gross = xp.abs(wt).sum(2)
        turnover = xp.abs(xp.diff(wt, axis=1)).sum(2).mean(1)
        ann = st[:, 1]
        sh = xp.where(st[:, 2] > 0, st[:, 0], 0)
        total = rt.sum(1)
        concentration = xp.where(total > 0, xp.sort(rt, axis=1)[:, -self.top_days:].sum(1)
                                 / xp.where(total > 0, total, 1), xp.inf)
        vals = xp.stack((sh, ann, st[:, 3], self.backend.measure(rt, self.recency_weights, self.ppy).cagr, st[:, 4], turnover,
                         gross.mean(1), ann / xp.maximum(turnover * self.ppy, 1e-12) * 1e4,
                         xp.abs(wt).max(axis=(1, 2)), concentration), axis=1)
        yearly = xp.stack((self.years.sharpe(rt, self.ppy), self.years.share(gross > 1e-9), self.years.total(lg)), axis=2)
        host = self.to_host(xp.concatenate((vals, yearly.reshape(len(W), -1), M.sound(rt, xp)[:, None]), axis=1)).tolist()
        dates = self.dates[self.start:]
        risks = self._risk_summaries(result.risk, len(W))
        return [StatsRow(W[i], rt[i], Metrics(*v[:10],
                    yearly=tuple(YearMetrics(int(year), int(days), round(v[10 + j * 3], 3),
                                             round(v[11 + j * 3], 3), round(v[12 + j * 3], 4))
                                 for j, (year, days) in enumerate(zip(self.years.years, self.years.days))),
                    span=f'{dates[0]}..{dates[-1]}', n_days=len(dates)), bool(v[-1]), risks[i])
                for i, v in enumerate(host)]

    def close(self):
        """Release this study's caches; runtime closes its execution resources."""
        self._members.clear()
        self._arrays.clear()
        self._reports.clear()
        self.fields.clear()
        self.panel.release()
        del self.spec, self.universe, self.slots, self.years
