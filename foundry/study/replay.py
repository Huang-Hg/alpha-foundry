"""Fresh chronological mixed-source account replay; no account trajectory cache."""
from __future__ import annotations

import hashlib
from collections.abc import Generator, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, fields, replace
from math import prod
from typing import TYPE_CHECKING, TypeVar, cast

import numpy as np

from foundry.numeric.semantics import ORDER_SEMANTIC_VERSION

from foundry.study.transforms import delay_lag
from foundry.study.programs import _descriptor, _empty, _event, _plan
from foundry.types import (
    AccountExecutionPort, AccountSnapshot, Array, ArraySource, BarExecution, BoolArray, ExecutionFeedback,
    ExecutionRiskBatch, FloatArray, Formula, IntArray, OrderCommands, OrderSnapshot,
    ProgramContext, ProgramRun, ProgramSource, Source, StrategyProgram,
)

if TYPE_CHECKING:
    from foundry.study.resources import Runtime


RowArray = TypeVar('RowArray', bound=Array)

MODEL = ORDER_SEMANTIC_VERSION
OWNER_POLICY = ('Signed owner contributions sum to the net account quantity. '
    'Spot sells consume only the issuing owner holdings. Isolated reduce-only '
    'must reduce both owner and net exposure; an ordinary owner close may '
    'increase net exposure. Aggregate liquidation extinguishes every owner '
    'contribution on that asset and allocates its fee by absolute contribution.')


@dataclass(frozen=True)
class ReplayMember:
    """One immutable source and its raw (D,S) intent or fresh-run factory.

    Formula and equal-weight values have already been mapped to equity fractions.
    Array values retain their declared unit and update mask. Shared features and
    caller-owned input arrays are borrowed; replay creates no full account book.
    """
    source: Source
    values: FloatArray | None = None
    update_mask: BoolArray | None = None
    program: StrategyProgram | None = None
    variant_indices: IntArray | None = None


@dataclass(frozen=True)
class ReplayResult:
    """Owned measured (B,V) outputs and bounded per-account summaries.

    Turnover is per-bar executed absolute notional/opening equity per research decision.
    Gross/net exposure are means over measured closes. Risk flows and terminal
    equity use the total configured initial account NAV as their unit. The empty
    risk liquidation_events axis is intentional: audit events are streamed.
    """
    returns: FloatArray
    statistics: FloatArray
    equity: FloatArray
    active: BoolArray
    turnover: FloatArray
    mean_gross: FloatArray
    mean_net: FloatArray
    max_abs_asset_exposure: FloatArray
    fill_count: FloatArray
    behavior_digests: tuple[str, ...]
    risk: ExecutionRiskBatch
    final_accounts: tuple[AccountSnapshot, ...]


@dataclass(frozen=True)
class _Run:
    member: ReplayMember
    run: ProgramRun
    rows: IntArray
    owner: int
    coefficients: FloatArray
    lag: int


def _rows(values: RowArray, rows: IntArray) -> RowArray:
    """Select private batch rows, preserving dtype and CPU read-only ownership."""
    selected = cast(RowArray, values[rows])
    if isinstance(selected, np.ndarray):
        selected.flags.writeable = False
    return selected


def _account_rows(account: AccountSnapshot, rows: IntArray) -> AccountSnapshot:
    orders = OrderSnapshot(**{item.name: _rows(getattr(account.orders, item.name), rows)
                             for item in fields(account.orders)})
    return AccountSnapshot(**{item.name: orders if item.name == 'orders'
        else _rows(getattr(account, item.name), rows) for item in fields(account)})


def _feedback_rows(feedback: ExecutionFeedback, rows: IntArray) -> ExecutionFeedback:
    return replace(feedback, account=_account_rows(feedback.account, rows),
        **{item.name: _rows(getattr(feedback, item.name), rows) for item in fields(feedback)
           if item.name not in ('account', 'event_ns', 'available_ns')})


def _commands(runtime: Runtime, batch: int, blocks: Sequence[OrderCommands]) -> OrderCommands:
    if not blocks:
        return _empty(runtime, batch)
    xp = runtime.backend.xp
    columns = {item.name: xp.concatenate([getattr(block, item.name) for block in blocks], axis=1)
               for item in fields(blocks[0])}
    active = columns['action'] != 0
    capacity = runtime.settings.execution.command_capacity
    if bool(xp.any(active.sum(1) > capacity)):
        raise ValueError('Mixed programs exceed the shared per-account command_capacity')
    # Stable compaction preserves owner/member command order even when different
    # trial rows emit in disjoint columns. Padding does not consume capacity.
    key = (~active).astype(xp.int64)*active.shape[1] + xp.arange(active.shape[1], dtype=xp.int64)[None, :]
    order = xp.argsort(key, axis=1)[:, :capacity]
    return OrderCommands(**{name: xp.take_along_axis(value, order, axis=1)
                            for name, value in columns.items()})


def _scaled_commands(runtime: Runtime, executor: AccountExecutionPort, state: _Run,
                     commands: OrderCommands) -> OrderCommands:
    """Validate callbacks once before scattering their bounded rows to accounts."""
    xp = runtime.backend.xp
    if not isinstance(commands, OrderCommands):
        raise TypeError('Program on_event must return OrderCommands')
    shape = commands.action.shape
    if (len(shape) != 2 or shape[0] != len(state.rows)
            or shape[1] > runtime.settings.execution.command_capacity):
        raise ValueError('Program commands require bounded effective (B,K) coordinates')
    columns = {}
    for item in fields(commands):
        value = getattr(commands, item.name)
        if value.shape != shape:
            raise ValueError('Program command columns must share effective (B,K) coordinates')
        if runtime.resources.placement.kind == 'cpu':
            if not isinstance(value, np.ndarray):
                raise TypeError('CPU program command columns must remain on the host')
        elif not hasattr(value, '__cuda_array_interface__'):
            raise TypeError('CUDA program command columns must remain on device')
        if item.name == 'reduce_only':
            valid_dtype = value.dtype == xp.dtype('bool')
        elif item.name in ('quantity', 'price', 'reservation_price'):
            valid_dtype = xp.issubdtype(value.dtype, xp.floating)
        else:
            valid_dtype = xp.issubdtype(value.dtype, xp.signedinteger)
        if not valid_dtype:
            raise TypeError('Program commands require declared signed integer, floating and bool dtypes')
        column = xp.zeros((executor.batch, shape[1]), dtype=value.dtype)
        column[state.rows] = value
        columns[item.name] = column
    active = commands.action != 0
    if bool(xp.any(active & (commands.owner != state.owner))):
        raise ValueError('Programs may issue commands only in their assigned owner namespace')
    columns['quantity'][state.rows] *= state.coefficients[:, None]
    # A zero coefficient never opens a run, so its padding cannot create orders.
    return OrderCommands(**columns)


def _source_key(member: ReplayMember) -> str:
    source = member.source
    if isinstance(source, Formula):
        return source.expression
    if isinstance(source, ProgramSource):
        return source.descriptor.id
    if isinstance(source, ArraySource):
        return source.producer + ':' + source.kind
    return repr(source)


def _steps(runtime: Runtime, members: Sequence[ReplayMember], coefficients: FloatArray,
           capital: FloatArray, delayed: bool) -> Generator[tuple[int, AccountExecutionPort], None, ReplayResult]:
    xp, pipeline = runtime.backend.xp, runtime.pipeline
    B, M = coefficients.shape
    spec = cast(BarExecution, runtime.loaded.program_spec)
    runnable = np.flatnonzero(spec.schedule.decision_index >= pipeline.decision_start)
    if not len(runnable):
        raise ValueError('Account replay requires an executable research decision')
    first_event = int(runnable[0])
    with ExitStack() as cleanup:
        executor = runtime.open_program_account(B, runtime.settings.execution.precision, first_event,
            owner_count=max(M, 1), initial_cash=capital)
        cleanup.callback(executor.close)
        units = tuple(member.source.unit if isinstance(member.source, ArraySource)
                      and member.source.kind == 'targets' else 'equity_fraction' for member in members)
        executor.configure_targets(units if M else ('equity_fraction',), coefficients if M else xp.ones((B, 1)))
        runs = []
        for owner, member in enumerate(members):
            if member.program is None:
                continue
            descriptor = _descriptor(member.program)
            planned = _plan(runtime, descriptor)
            rows = xp.flatnonzero((coefficients[:, owner] > 0) & (capital > 0))
            if not len(rows):
                continue
            planned = replace(planned, batch_size=len(rows),
                state_bytes=planned.state_bytes//descriptor.batch_size*len(rows),
                feedback_capacity=executor.feedback_capacity, model=MODEL)
            source = cast(ProgramSource, member.source)
            stream = int(xp.cuda.get_current_stream().ptr) if planned.device.kind == 'cuda' else None
            scale = coefficients[rows, owner].copy()
            context = ProgramContext(descriptor, tuple(runtime.panel.assets), planned, xp, stream,
                capital[rows].copy(), len(rows),
                xp.full(len(rows), source.variant, dtype=xp.int64) if member.variant_indices is None
                else member.variant_indices[rows].copy(), scale.copy(), owner)
            run = member.program.open(context)
            if not isinstance(run, ProgramRun):
                raise TypeError('program.open must return a ProgramRun')
            cleanup.callback(run.close)
            runs.append(_Run(member, run, rows, owner, scale,
                             delay_lag(_source_key(member)) if delayed else 0))
        target_values = xp.zeros((B, max(M, 1), len(runtime.panel.assets)), dtype=xp.float64)
        target_updates = xp.zeros(target_values.shape, dtype=xp.bool_)
        lags = [delay_lag(_source_key(member)) if delayed else 0 for member in members]
        account = executor.snapshot()
        for event in range(first_event, len(spec.schedule.event_ns)):
            decision = int(spec.schedule.decision_index[event])
            blocks = []
            target_updates.fill(False)
            if decision >= pipeline.decision_start:
                for owner, member in enumerate(members):
                    if member.values is None:
                        continue
                    at = decision-lags[owner]
                    if at < 0:
                        continue
                    target_values[:, owner] = member.values[at]
                    target_updates[:, owner] = True if member.update_mask is None else member.update_mask[at]
                    target_updates[:, owner] &= coefficients[:, owner, None] > 0
                for state in runs:
                    observed = decision-state.lag
                    if observed < 0:
                        continue
                    owner_quantity = _rows(executor.owner_quantities(state.owner), state.rows)
                    visible = _event(runtime, state.member.program.descriptor, observed,
                                     _account_rows(account, state.rows), owner_quantity)
                    visible = replace(visible, decision_ns=int(pipeline.clock.decision_ns[decision]))
                    emitted = state.run.on_event(visible)
                    blocks.append(_scaled_commands(runtime, executor, state, emitted))
            feedback = executor.advance(event, _commands(runtime, B, blocks),
                targets=target_values, update_mask=target_updates)
            account = feedback.account
            for state in runs:
                if state.run.on_feedback(_feedback_rows(feedback, state.rows)) is not None:
                    raise TypeError('Program on_feedback must return None')
            yield event, executor
        start = pipeline.measurement_start
        measured = executor.activity()
        count = xp.maximum(measured[:, 5], 1.)
        decisions = len(pipeline.clock.decision_ns)-pipeline.decision_start
        domain = repr((MODEL, runtime.settings.execution, runtime.settings.account, spec.snapshot)).encode()
        digests = tuple(hashlib.sha256(domain+bytes.fromhex(value)).hexdigest()
                        for value in executor.behavior_digests())
        risk = executor.risk(runtime.settings.account.initial_cash)
        risk = replace(risk, **{name: getattr(risk, name)[:, start:] for name in
            ('funding', 'trading_fees', 'liquidation_fees', 'isolated_loss_cap', 'liquidations')})
        returns = executor.returns()[:, start:].copy()
        return ReplayResult(returns, runtime.order_statistics(returns), executor.equity()[:, start:].copy(),
            executor.active()[:, start:].copy(), measured[:, 0]/max(decisions, 1), measured[:, 1]/count,
            measured[:, 2]/count, measured[:, 3], measured[:, 4], digests, risk, (executor.snapshot(),))


def _shared(runtime: Runtime, members: Sequence[ReplayMember], coefficients: FloatArray,
            capital: FloatArray, delayed: bool) -> ReplayResult:
    steps = _steps(runtime, members, coefficients, capital, delayed)
    try:
        while True:
            next(steps)
    except StopIteration as completed:
        return completed.value
    finally:
        steps.close()


def private_state_bytes(runtime: Runtime, members: Sequence[ReplayMember]) -> int:
    """Conservatively include declared private tensors and copied causal windows."""
    total = 0
    for member in members:
        if member.program is not None:
            descriptor = member.program.descriptor
            total += sum(np.dtype(item.dtype).itemsize*prod(item.shape) for item in descriptor.state_schema)
            total += descriptor.history*(8 + len(runtime.panel.assets)*(1 + sum(
                runtime.panel[name].dtype.itemsize for name in descriptor.features)))
    return total


def replay(runtime: Runtime, members: Sequence[ReplayMember], coefficients: FloatArray, *,
           delayed: bool = False) -> ReplayResult:
    """Replay new accounts for every trial; shared coefficients scale intent once.

    Independent coefficients allocate actual initial capital, sum to at most one,
    and never scale the member intent a second time. Unallocated capital remains
    constant cash. The delayed control shifts causal observations/target intents;
    realized returns and account feedback are always generated by a fresh replay.
    """
    with runtime.resources.scope():
        xp = runtime.backend.xp
        if runtime.settings.execution.model != 'orders':
            raise ValueError('Mixed account replay requires execution.model=orders fixed at open')
        if runtime.loaded.program_spec is None:
            raise ValueError('Order account replay requires compiled Bar execution capabilities')
        weights = xp.asarray(coefficients, dtype=xp.float64)
        if (weights.ndim != 2 or weights.shape[1] != len(members) or weights.shape[0] < 1
                or weights.shape[0] > max(runtime.pipeline.trial_batch, runtime.pipeline.candidate_batch)
                or bool(xp.any(~xp.isfinite(weights) | (weights < 0)))):
            raise ValueError('Replay coefficients require bounded finite nonnegative (B,M) values')
        for member in members:
            if (member.program is None) == (member.values is None):
                raise ValueError('Every replay member requires exactly one target input or program')
            if member.values is not None and member.values.shape != runtime.panel.shape:
                raise ValueError('Replay targets must align with the decision and asset axes')
            if member.update_mask is not None and (member.values is None
                    or member.update_mask.shape != member.values.shape
                    or member.update_mask.dtype != xp.dtype('bool')):
                raise ValueError('Replay target masks require bool decision/asset coordinates')
        B = len(weights)
        effective_cap = runtime.replay_batch_size(len(members), private_state_bytes(runtime, members))
        if B > effective_cap:
            raise ValueError(f'Replay trial count exceeds the effective allocation cap {effective_cap}; split the query')
        for member in members:
            if member.variant_indices is not None:
                if (member.program is None or member.variant_indices.shape != (B,)
                        or member.variant_indices.dtype != xp.dtype('int64')
                        or bool(xp.any((member.variant_indices < 0)
                            | (member.variant_indices >= member.program.descriptor.batch_size)))):
                    raise ValueError('Program variant overrides require resident int64 (B,) original batch indices')
        capital = xp.full(B, runtime.settings.account.initial_cash, dtype=xp.float64)
        if runtime.settings.account.composition == 'shared':
            return _shared(runtime, members, weights, capital, delayed)
        if bool(xp.any(weights.sum(1) > 1.+runtime.settings.portfolio.weight_sum_atol)):
            raise ValueError('Independent account capital allocations cannot sum above one')
        # Inputs within tolerance cannot create cash: the final positive excess is
        # rejected rather than normalized or silently financed.
        if bool(xp.any(weights.sum(1) > 1.)):
            raise ValueError('Independent capital allocations must sum to at most one')
        if not members:
            return _shared(runtime, (), weights, capital, delayed)
        cash = capital*(1.-weights.sum(1))
        spec = cast(BarExecution, runtime.loaded.program_spec)
        statistics_state = xp.zeros((B, 7), dtype=xp.float64)
        statistics_state[:, 6] = capital
        active = xp.zeros((B, spec.schedule.measurement_count-runtime.pipeline.measurement_start), dtype=xp.bool_)
        equity = xp.broadcast_to(capital[:, None], active.shape).copy()
        returns = xp.zeros_like(equity)
        parts = []
        with ExitStack() as cleanup:
            streams = [_steps(runtime, (member,), xp.ones((B, 1)), capital*weights[:, i], delayed)
                       for i, member in enumerate(members)]
            for stream in streams:
                cleanup.callback(stream.close)
            while True:
                current = []
                for stream in streams:
                    try:
                        current.append(next(stream))
                    except StopIteration as completed:
                        parts.append(completed.value)
                if not current:
                    break
                if len(current) != len(streams) or len({event for event, _ in current}) != 1:
                    raise RuntimeError('Independent account event clocks diverged')
                event = current[0][0]
                runtime.independent_turnover(xp.stack([executor.bar_activity() for _, executor in current], axis=1),
                                             cash, statistics_state)
                measurement = int(spec.schedule.measurement_index[event])
                measured = event+1 == len(spec.schedule.event_ns) or spec.schedule.measurement_index[event+1] != measurement
                if measured and measurement >= runtime.pipeline.measurement_start:
                    snapshots = [executor.snapshot() for _, executor in current]
                    sample = runtime.sample_independent(xp.stack([account.quantity for account in snapshots], axis=1),
                        xp.stack([account.equity for account in snapshots], axis=1), cash, event, statistics_state)
                    index = measurement-runtime.pipeline.measurement_start
                    active[:, index] = sample[:, 4] != 0
                    equity[:, index] = sample[:, 0]
                    returns[:, index] = sample[:, 5]
        risk = parts[0].risk
        risk = replace(risk, **{name: sum((getattr(part.risk, name) for part in parts), xp.zeros_like(getattr(risk, name)))
            for name in ('funding', 'trading_fees', 'liquidation_fees', 'isolated_loss_cap', 'liquidations')},
            first_liquidation_ns=xp.min(xp.stack([xp.where(part.risk.first_liquidation_ns >= 0,
                part.risk.first_liquidation_ns, np.iinfo(np.int64).max) for part in parts]), axis=0),
            bankrupt=equity[:, -1] <= 0, ending_equity=equity[:, -1]/runtime.settings.account.initial_cash,
            max_gross_leverage=statistics_state[:, 4], model='independent_actual_capital_v1',
            gross_sampling='statistics_closes')
        risk = replace(risk, first_liquidation_ns=xp.where(risk.first_liquidation_ns == np.iinfo(np.int64).max,
                                                        -1, risk.first_liquidation_ns))
        digests = tuple(hashlib.sha256(b'independent-actual-capital-v1'+b''.join(
            bytes.fromhex(part.behavior_digests[b]) for part in parts)).hexdigest() for b in range(B))
        return ReplayResult(returns, runtime.order_statistics(returns), equity, active, statistics_state[:, 5]/max(len(runtime.pipeline.clock.decision_ns)-runtime.pipeline.decision_start, 1),
            statistics_state[:, 0]/xp.maximum(statistics_state[:, 3], 1.),
            statistics_state[:, 1]/xp.maximum(statistics_state[:, 3], 1.), statistics_state[:, 2],
            sum((part.fill_count for part in parts), xp.zeros(B)), digests, risk,
            tuple(account for part in parts for account in part.final_accounts))
