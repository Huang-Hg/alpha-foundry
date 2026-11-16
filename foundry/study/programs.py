"""Bounded causal orchestration of private strategy runs and actual accounts."""
from __future__ import annotations

import hashlib
from contextlib import ExitStack
from math import prod
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

import numpy as np

from foundry.types import (
    AccountSnapshot,
    BarExecution,
    FloatArray,
    OrderCommands,
    ProgramContext,
    ProgramDescriptor,
    ProgramEvent,
    ProgramPlan,
    ProgramRun,
    ProgramSimulation,
    ResearchContext,
    StrategyProgram,
)

if TYPE_CHECKING:
    from foundry.study.resources import Runtime


MODEL = 'bar_orders_actual_account_v1'
LATENCY = ('Orders are emitted only at the configured decision clock and become '
           'eligible at its compiled next base-bar open. Feedback follows every '
           'completed base bar and cannot emit orders; protection based on a '
           'new fill waits until the next decision.')


def _descriptor(program: StrategyProgram) -> ProgramDescriptor:
    if not isinstance(program, StrategyProgram):
        raise TypeError('program must implement descriptor and open(context)')
    descriptor = program.descriptor
    if not isinstance(descriptor, ProgramDescriptor):
        raise TypeError('program descriptor must be a ProgramDescriptor')
    return descriptor


def plan(runtime: Runtime, program: StrategyProgram) -> ProgramPlan:
    """Validate external capabilities once, before creating any private run."""
    return _plan(runtime, _descriptor(program))


def _plan(runtime: Runtime, descriptor: ProgramDescriptor) -> ProgramPlan:
    spec = runtime.loaded.program_spec
    if spec is None or spec.market not in ('crypto_spot', 'crypto_perpetual'):
        raise ValueError('Program orders require compiled crypto Bar execution; equity corporate actions and settlement are unavailable')
    device = runtime.resources.placement
    if device.kind not in descriptor.devices:
        raise ValueError(f'Program declares devices {descriptor.devices}; open an explicitly supported backend for {device.kind}')
    pipeline = runtime.pipeline
    if descriptor.batch_size > pipeline.candidate_batch:
        raise ValueError('Program batch_size exceeds the resident candidate_batch limit')
    if not set(descriptor.features) <= set(runtime.panel.fields):
        raise ValueError('Program features must name fields available in this study')
    executable = spec.schedule.decision_index
    runnable = executable[executable >= pipeline.decision_start]
    if not len(runnable):
        raise ValueError('Program requires a decision with a subsequent executable base bar in the research window')
    first = int(pipeline.clock.decision_ns[int(runnable[0])])
    if descriptor.fit_at > first:
        raise ValueError('Program fitting information must be visible before its first decision')
    updates = executable >= 0
    cutoffs = pipeline.clock.decision_ns[executable[updates]]
    if np.any(cutoffs > spec.schedule.event_ns[updates]):
        raise ValueError('Program decisions cannot follow their compiled execution opens')
    prior = np.flatnonzero(updates)
    prior = prior[prior > 0]
    if np.any(spec.event_close_ns[prior - 1] > pipeline.clock.decision_ns[executable[prior]]):
        raise ValueError('Program account feedback would include a base bar unavailable at the decision')
    execution = runtime.settings.execution
    precision = execution.precision
    B, H, S = descriptor.batch_size, descriptor.history, len(runtime.panel.assets)
    state_bytes = B * sum(np.dtype(item.dtype).itemsize * prod(item.shape)
                          for item in descriptor.state_schema)
    window_bytes = H * S * sum(runtime.panel[name].dtype.itemsize for name in descriptor.features)
    window_bytes += H * S * np.dtype('bool').itemsize + H * np.dtype('int64').itemsize
    identity = hashlib.sha256(repr((MODEL, pipeline.context.id, spec.snapshot, descriptor,
        runtime.settings.account, execution)).encode()).hexdigest()
    return ProgramPlan(device, f'explicit_{device.kind}_program_and_bar_execution', precision,
        spec.market, MODEL, execution.bar_path, execution.participation, B, H,
        execution.order_capacity, execution.command_capacity, state_bytes, window_bytes,
        execution.command_capacity + 4 * execution.order_capacity + S,
        ('market', 'limit', 'stop', 'cancel', 'amend'), LATENCY,
        'Missing or nonpositive crypto OHLC quotes cannot execute orders; spot valuation carries the last valid close, with zero before first listing.',
        identity)


def _empty(runtime: Runtime, batch: int) -> OrderCommands:
    """A zero-column command block leaves outstanding orders unchanged."""
    xp = runtime.backend.xp
    integer = xp.empty((batch, 0), dtype=xp.int64)
    numeric = xp.empty((batch, 0), dtype=xp.float64)
    boolean = xp.empty((batch, 0), dtype=xp.bool_)
    return OrderCommands(integer, integer, integer, integer, integer, numeric,
                         integer, numeric, numeric, integer, boolean)


def _event(runtime: Runtime, descriptor: ProgramDescriptor, decision: int,
           account: AccountSnapshot) -> ProgramEvent:
    """Copy a bounded visible window; no slice retains a future panel base."""
    xp, pipeline = runtime.backend.xp, runtime.pipeline
    start, stop = max(0, decision + 1 - descriptor.history), decision + 1
    history = np.array(pipeline.clock.decision_ns[start:stop], copy=True, order='C')
    history.flags.writeable = False
    features = {name: xp.array(runtime.panel[name][start:stop], copy=True, order='C')
                for name in descriptor.features}
    eligibility = xp.array(pipeline.universe[start:stop], copy=True, order='C')
    if runtime.resources.placement.kind == 'cpu':
        for value in features.values():
            value.flags.writeable = False
        eligibility.flags.writeable = False
    return ProgramEvent(int(pipeline.clock.decision_ns[decision]), history,
                        MappingProxyType(features), eligibility, account)


def simulate(runtime: Runtime, context: ResearchContext,
             program: StrategyProgram) -> ProgramSimulation:
    """Run once with fresh private state; never publish candidates or accounts."""
    with runtime.resources.scope():
        descriptor, xp = _descriptor(program), runtime.backend.xp
        planned = _plan(runtime, descriptor)
        spec = cast(BarExecution, runtime.loaded.program_spec)
        first_event = int(np.flatnonzero(spec.schedule.decision_index >= runtime.pipeline.decision_start)[0])
        # plan() establishes the compiled capability before resource creation.
        with ExitStack() as cleanup:
            executor = runtime.open_program_account(descriptor.batch_size, planned.precision, first_event)
            cleanup.callback(executor.close)
            if planned.device.kind == 'cuda':
                stream = int(xp.cuda.get_current_stream().ptr)
            else:
                stream = None
            run = program.open(ProgramContext(descriptor, tuple(runtime.panel.assets), planned,
                xp, stream, runtime.settings.account.initial_cash))
            if not isinstance(run, ProgramRun):
                raise TypeError('program.open must return a ProgramRun with on_event, on_feedback and close')
            cleanup.callback(run.close)
            empty = _empty(runtime, descriptor.batch_size)
            account = executor.snapshot()
            for event in range(first_event, len(spec.schedule.event_ns)):
                decision = int(spec.schedule.decision_index[event])
                commands = (run.on_event(_event(runtime, descriptor, decision, account))
                            if decision >= runtime.pipeline.decision_start else empty)
                feedback = executor.advance(event, commands)
                account = feedback.account
                if run.on_feedback(feedback) is not None:
                    raise TypeError('Program on_feedback must return None; orders require a decision callback')
            # Copies detach the returned result from the runtime's final cleanup.
            returns: FloatArray = executor.returns()[:, runtime.pipeline.measurement_start:].copy()
            equity: FloatArray = executor.equity()[:, runtime.pipeline.measurement_start:].copy()
            final = executor.snapshot()
            times = np.array(runtime.pipeline.measurement_times[runtime.pipeline.measurement_start:],
                             dtype='datetime64[ns]').astype(np.int64)
            times.flags.writeable = False
            if planned.device.kind == 'cpu':
                returns.flags.writeable = False
                equity.flags.writeable = False
            return ProgramSimulation(context, descriptor, planned, times, returns, equity, final)
