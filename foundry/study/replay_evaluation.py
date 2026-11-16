"""Typed source-to-intent preparation and common reports for fresh account replay."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import fields, replace
from typing import TYPE_CHECKING

from foundry.numeric import metrics as M
from foundry.study.transforms import EXPOSURE, INVALID
from foundry.study.program_registry import ProgramRegistry
from foundry.study.replay import ReplayMember, ReplayResult, private_state_bytes, replay
from foundry.types import (
    AccountSnapshot, ArraySource, ArrayStore, EqualWeight, EvaluationRow, FloatArray, Formula,
    Metrics, OrderSnapshot, ProgramDescriptor, ProgramSource, RiskSummary, SimulationResult, Source, StatsRow, YearMetrics,
)

if TYPE_CHECKING:
    from foundry.study.resources import Runtime


class ReplayEvaluation:
    """Borrow immutable inputs; every query opens fresh program and account state."""

    def __init__(self, runtime: Runtime, arrays: ArrayStore, programs: ProgramRegistry):
        self.runtime, self.arrays, self.programs = runtime, arrays, programs
        self.pipeline = runtime.pipeline

    def _prepare(self, sources: Sequence[Source]
                 ) -> tuple[list[EvaluationRow], list[ReplayMember | None]]:
        pipeline, xp = self.pipeline, self.pipeline.xp
        precision = self.runtime.settings.execution.precision
        rows = [EvaluationRow(source, True, None, precision) for source in sources]
        members: list[ReplayMember | None] = [None] * len(sources)
        programs = self.programs.read_many(tuple(source for source in sources if isinstance(source, ProgramSource)))
        positions = [i for i, source in enumerate(sources) if isinstance(source, Formula)]
        if positions:
            mapped, ids, books, finite = pipeline.formula_intents([sources[i].expression for i in positions])
            valid = pipeline.to_host(finite).tolist()
            budgets = self._score_budgets(books)
            for i, row in zip(positions, mapped):
                rows[i] = row
                row.source = sources[i]
                row.precision = precision
            for j, at in enumerate(ids):
                i, row = positions[at], mapped[at]
                row.ok = bool(valid[j] and budgets[j])
                row.error = None if row.ok else EXPOSURE if not budgets[j] else INVALID
                if row.ok:
                    members[i] = ReplayMember(sources[i], values=books[j])
        for i, source in enumerate(sources):
            if isinstance(source, ArraySource):
                batch = pipeline.inputs.read(source) if source.scope else self.arrays.read(source)
                if not source.scope:
                    pipeline.validate_batch(batch)
                values = xp.asarray(batch.values)
                if source.kind == 'scores':
                    books = pipeline.score_to_book(values)
                    finite = bool(xp.isfinite(values).any())
                    budget = self._score_budgets(books)[0]
                    rows[i].ok = finite and budget
                    rows[i].error = None if rows[i].ok else EXPOSURE if not budget else INVALID
                    if rows[i].ok:
                        members[i] = ReplayMember(source, values=books[0])
                else:
                    members[i] = ReplayMember(source, values=values[0],
                        update_mask=None if batch.update_mask is None else xp.asarray(batch.update_mask[0]))
            elif isinstance(source, ProgramSource):
                members[i] = ReplayMember(source, program=programs[source])
            elif isinstance(source, EqualWeight):
                members[i] = ReplayMember(source, values=pipeline.equal_weight())
        return rows, members

    def _score_budgets(self, books: FloatArray) -> list[bool]:
        pipeline, xp = self.pipeline, self.pipeline.xp
        if pipeline.settings.sizing.rule != 'independent':
            return [True] * len(books)
        return pipeline.to_host((xp.abs(books).sum(axis=2, dtype=xp.float64)
            <= pipeline.allocation.max_gross + pipeline.portfolio_cfg.weight_sum_atol).all(axis=1)).tolist()

    def members(self, sources: Sequence[Source]) -> tuple[ReplayMember, ...]:
        """Prepare reusable immutable intents for one bounded portfolio query."""
        with self.runtime.resources.scope():
            members = []
            for part in self.pipeline.batches(sources):
                rows, prepared = self._prepare(part)
                invalid = [row for row in rows if not row.ok]
                if invalid:
                    raise ValueError(f'Portfolio source cannot produce valid intents: {invalid[0].source!r}: {invalid[0].error}')
                members.extend(member for member in prepared if member is not None)
            return tuple(members)

    def evaluate(self, sources: Sequence[Source], report: bool = True, *, delayed: bool = False
                 ) -> list[EvaluationRow]:
        """Reject invalid candidates as rows, without simulating invalid intents."""
        pipeline, xp = self.pipeline, self.pipeline.xp
        output = []
        with self.runtime.resources.scope():
            for part in pipeline.batches(sources):
                rows, members = self._prepare(part)
                for indices, member in self._evaluation_groups(members):
                    _, measured, result = self.simulate_members((member,), xp.ones((len(indices), 1)),
                        delayed=delayed, report=report)
                    if not report:
                        status = pipeline.to_host(xp.stack((M.sound(result.returns, xp), result.statistics[:, 3]), axis=1))
                        risks = self._risk_summaries(result)
                    for j, i in enumerate(indices):
                        row = rows[i]
                        row.returns = result.returns[j]
                        if report:
                            stats = measured[j]
                            row.valid_returns, row.maxdd, row.risk = stats.valid_returns, stats.metrics.maxdd, stats.risk
                            row.metrics = stats.metrics
                        else:
                            row.valid_returns, row.maxdd = bool(status[j, 0]), float(status[j, 1])
                            row.risk = risks[j]
                        row.behavior_digest = result.behavior_digests[j]
                output.extend(rows)
        return output

    def _evaluation_groups(self, members: Sequence[ReplayMember | None]
                           ) -> list[tuple[tuple[int, ...], ReplayMember]]:
        """Batch original variants in one owner slot without a quadratic owner matrix."""
        groups: list[tuple[tuple[int, ...], ReplayMember]] = []
        programs: dict[tuple[ProgramDescriptor, str, str], list[tuple[int, ReplayMember, int]]] = {}
        for i, member in enumerate(members):
            if member is None:
                continue
            source = member.source
            if isinstance(source, ProgramSource):
                programs.setdefault((source.descriptor, source.snapshot, source.scope), []).append((i, member, source.variant))
            else:
                groups.append(((i,), member))
        for selected in programs.values():
            variants = self.pipeline.xp.asarray([variant for _, _, variant in selected], dtype=self.pipeline.xp.int64)
            groups.append((tuple(i for i, _, _ in selected), replace(selected[0][1], variant_indices=variants)))
        return groups

    def simulate(self, sources: Sequence[Source], coefficients: FloatArray, *, delayed: bool = False, report: bool = True
                 ) -> tuple[SimulationResult, list[StatsRow], ReplayResult]:
        return self.simulate_members(self.members(sources), coefficients, delayed=delayed, report=report)

    def simulate_members(self, members: Sequence[ReplayMember], coefficients: FloatArray, *, delayed: bool = False,
                         report: bool = True
                         ) -> tuple[SimulationResult, list[StatsRow], ReplayResult]:
        """Return windowed simulation/metrics and bounded replay activity summaries."""
        with self.runtime.resources.scope():
            xp = self.pipeline.xp
            weights = xp.asarray(coefficients, dtype=xp.float64)
            if (weights.ndim != 2 or weights.shape[1] != len(members) or not len(weights)
                    or bool(xp.any(~xp.isfinite(weights) | (weights < 0)))):
                raise ValueError('Replay coefficients require finite nonnegative nonempty (B,M) coordinates')
            if self.runtime.settings.account.composition == 'independent' and bool(xp.any(weights.sum(1) > 1.)):
                raise ValueError('Independent capital allocations must sum to at most one')
            for member in members:
                if member.variant_indices is not None and (member.variant_indices.shape != (len(weights),)
                        or member.variant_indices.dtype != xp.dtype('int64')):
                    raise ValueError('Replay variant overrides require resident int64 (B,) coordinates')
            capacity = self.runtime.replay_batch_size(len(members), private_state_bytes(self.runtime, members))
            parts = []
            for start in range(0, len(weights), capacity):
                stop = start + capacity
                selected = tuple(replace(member, variant_indices=member.variant_indices[start:stop])
                    if member.variant_indices is not None else member for member in members)
                parts.append(replay(self.runtime, selected, weights[start:stop], delayed=delayed))
            result = parts[0] if len(parts) == 1 else self._join(parts)
            simulation = SimulationResult(result.returns, result.statistics, result.risk)
            return simulation, self._metrics(result) if report else [], result

    def _join(self, parts: Sequence[ReplayResult]) -> ReplayResult:
        """Join only measured outputs; live accounts are already deterministically closed."""
        xp = self.pipeline.xp
        risk = replace(parts[0].risk, **{item.name: xp.concatenate([getattr(part.risk, item.name) for part in parts])
            for item in fields(parts[0].risk) if item.name not in ('model', 'gross_sampling')})
        accounts = []
        for account_parts in zip(*(part.final_accounts for part in parts)):
            orders = OrderSnapshot(**{item.name: xp.concatenate([getattr(account.orders, item.name) for account in account_parts])
                for item in fields(account_parts[0].orders)})
            accounts.append(AccountSnapshot(**{item.name: orders if item.name == 'orders' else
                xp.concatenate([getattr(account, item.name) for account in account_parts]) for item in fields(account_parts[0])}))
        return replace(parts[0], **{item.name: xp.concatenate([getattr(part, item.name) for part in parts])
            for item in fields(parts[0]) if item.name not in ('behavior_digests', 'risk', 'final_accounts')},
            behavior_digests=tuple(value for part in parts for value in part.behavior_digests),
            risk=risk, final_accounts=tuple(accounts))

    def _risk_summaries(self, result: ReplayResult) -> list[RiskSummary]:
        pipeline, xp, risk = self.pipeline, self.pipeline.xp, result.risk
        amounts = pipeline.to_host(xp.stack((risk.funding.sum(1), risk.trading_fees.sum(1),
            risk.liquidation_fees.sum(1), risk.isolated_loss_cap.sum(1), risk.ending_equity,
            risk.max_gross_leverage), axis=1)).tolist()
        counts = pipeline.to_host(risk.liquidations.sum(1)).tolist()
        timestamps = pipeline.to_host(risk.first_liquidation_ns).tolist()
        bankrupt = pipeline.to_host(risk.bankrupt).tolist()
        return [RiskSummary(risk.model, v[0], v[1], v[2], v[3], int(count), int(timestamp),
            bool(failed), v[4], v[5], gross_sampling=risk.gross_sampling)
            for v, count, timestamp, failed in zip(amounts, counts, timestamps, bankrupt)]

    def _metrics(self, result: ReplayResult) -> list[StatsRow]:
        pipeline, xp = self.pipeline, self.pipeline.xp
        rt, st = result.returns, result.statistics
        lg = xp.log1p(rt)
        ann = st[:, 1]
        sharpe = xp.where(st[:, 2] > 0, st[:, 0], 0)
        total = rt.sum(1)
        concentration = xp.where(total > 0, xp.sort(rt, axis=1)[:, -pipeline.top_periods:].sum(1)
            / xp.where(total > 0, total, 1), xp.inf)
        values = xp.stack((sharpe, ann, st[:, 3],
            pipeline.backend.measure(rt, pipeline.recency_weights, pipeline.ppy).cagr,
            st[:, 4], result.turnover, result.mean_gross,
            ann / xp.maximum(result.turnover * pipeline.decision_ppy, 1e-12) * 1e4,
            result.max_abs_asset_exposure, concentration), axis=1)
        yearly = xp.stack((pipeline.years.sharpe(rt, pipeline.ppy),
            pipeline.years.share(result.active), pipeline.years.total(lg)), axis=2)
        host = pipeline.to_host(xp.concatenate((values, yearly.reshape(len(rt), -1),
            M.sound(rt, xp)[:, None]), axis=1)).tolist()
        risks = self._risk_summaries(result)
        dates = pipeline.measurement_times[pipeline.measurement_start:]
        return [StatsRow(None, rt[i], Metrics(*v[:10],
            yearly=tuple(YearMetrics(int(year), int(periods), round(v[10+j*3], 3),
                round(v[11+j*3], 3), round(v[12+j*3], 4)) for j, (year, periods) in
                enumerate(zip(pipeline.years.years, pipeline.years.periods))),
            span=f'{dates[0]}..{dates[-1]}', n_periods=len(dates), activity_basis='executed_fills'),
            bool(v[-1]), risks[i]) for i, v in enumerate(host)]
