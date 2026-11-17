"""Explicit startup assembly. Each runtime owns its compute resources and study."""
from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
from types import ModuleType, TracebackType
import os
from typing import TYPE_CHECKING, Literal, Self, cast

import numpy as np

from foundry.config import (
    AShareConfig,
    CryptoSpotConfig,
    CpuConfig,
    CudaConfig,
    Settings,
    UsEquityConfig,
)
from foundry.backtest import orders, orders_cuda
from foundry.market.a_share import AShare
from foundry.market.crypto_spot import CryptoSpot
from foundry.market.data import load_data
from foundry.market.perpetual import CryptoPerpetual
from foundry.market.us_equity import UsEquity
from foundry.ops.registry import MAX_WINDOW
from foundry.ops.evaluate import evaluate
from foundry.numeric.parallel import HostWorkers
from foundry.study.pipeline import DeviceResources, Pipeline
from foundry.study.store import study_snapshot
from foundry.study.identity import DeviceHasher, host_leaves
from foundry.types import (
    AccountExecutionPort,
    ComputeDevice,
    Array,
    HostArray,
    Backend,
    BoolArray,
    BarExecution,
    FloatArray,
    ExecutionSpec,
    ParsedFormula,
    PerpetualSpec,
    PrecisionPlan,
    ResourceStatus,
    Study,
    Window,
)

if TYPE_CHECKING:
    from foundry.ops.ast import Node
    from foundry.market.panel import Panel
    from foundry.study.pipeline import ExecutionBackend, StudyData


def _all_supported(tree: ParsedFormula) -> tuple[str, ...]:
    return ()


def _close_host_backend() -> None:
    """Host operator tables contain no owned execution resources."""


def prepare_backend(name: str) -> Backend:
    """Initialize one explicitly chosen implementation; unsupported names fail."""
    if name == 'cuda':
        import cupy as cp

        from foundry.numeric.cuda import Moments
        from foundry.ops import cuda
        moments = Moments()
        operators = cuda.prepare()
        return Backend(name, cp, operators.IMPL, operators.unsupported, operators.max_window,
                       operators.max_pair_window, cp.asnumpy, moments.measure,
                       lambda: (operators.close(), moments.close()))
    if name == 'cpu':
        from foundry.numeric.host import measure
        from foundry.ops import cpu
        return Backend(name, np, cpu.IMPL, _all_supported, MAX_WINDOW, MAX_WINDOW,
                       np.asarray, measure, _close_host_backend)
    if name == 'portable':
        from foundry.numeric.host import measure
        from foundry.ops import portable
        return Backend(name, np, portable.IMPL, _all_supported, MAX_WINDOW, MAX_WINDOW,
                       np.asarray, measure, _close_host_backend)
    raise ValueError(f'Unknown execution backend: {name}')


def configured_study(settings: Settings) -> Study:
    cfg = settings.market
    return Study(settings.market.name, cfg.start, cfg.end)


def load_study(settings: Settings, study: Study, backend: Backend) -> StudyData:
    """Read one artifact snapshot, then compile its market rules without IO."""
    cfg = settings.market
    if isinstance(cfg, AShareConfig):
        market = AShare(cfg.commission, cfg.transfer_fee, cfg.stamp_tax_sell, settings.account.gross_limit, settings.clock, settings.execution)
    elif isinstance(cfg, UsEquityConfig):
        market = UsEquity(cfg.cost_rate, cfg.daily_borrow, settings.account.gross_limit, settings.clock, settings.execution)
    elif isinstance(cfg, CryptoSpotConfig):
        market = CryptoSpot(cfg.maker_fee, cfg.taker_fee, settings.account.gross_limit, settings.clock, settings.execution)
    else:
        market = CryptoPerpetual(settings.account.leverage, cfg.maker_fee, cfg.taker_fee, cfg.maintenance_rate,
                                 cfg.liquidation_fee, settings.account.gross_limit, settings.clock, settings.execution)
    return market.compile(load_data(settings, study), backend, study)


class HostResources:
    def __init__(self, cfg: CpuConfig, initializer: Callable[[], None] | None = None) -> None:
        self.placement = ComputeDevice('cpu')
        self.cfg = cfg
        self.workers = HostWorkers(cfg.workers, initializer)
        self.streams = (nullcontext(),)

    def formulas(self, roots: Sequence[Node], fields: Mapping[str, FloatArray],
                 backend: Backend, universe: BoolArray) -> list[FloatArray]:
        return list(self.workers.map(lambda root: evaluate(root, fields, backend, {}, universe), roots))

    def scope(self) -> nullcontext[None]:
        return nullcontext()

    def start_batch(self) -> None:
        pass

    def finish_batch(self) -> None:
        pass

    def synchronize(self) -> None:
        pass

    def stage_host(self, values: Array) -> HostArray:
        return np.ascontiguousarray(values)

    def content_leaves(self, values: Array) -> HostArray:
        return host_leaves(values)

    def batch_sizes(self, cfg: CudaConfig, shape: tuple[int, int], events: int,
                    measurements: int) -> tuple[int, int]:
        return self.cfg.candidate_batch, self.cfg.workers

    def measurement_budget(self, cfg: CudaConfig) -> int:
        return self.cfg.measurement_cache_mib * 1048576

    def status(self) -> ResourceStatus:
        return ResourceStatus('CPU', 0)

    def close(self) -> None:
        self.workers.close()
        self.streams = ()


class CudaResources:
    def __init__(self, cp: ModuleType, stream_count: int):
        self.cp = cp
        self.device = cp.cuda.Device()
        self.placement = ComputeDevice('cuda', self.device.id)
        self.pool = cp.cuda.MemoryPool()
        self.hasher = DeviceHasher(cp)
        self.streams = tuple(cp.cuda.Stream(non_blocking=True) for _ in range(stream_count))

    def formulas(self, roots: Sequence[Node], fields: Mapping[str, FloatArray],
                 backend: Backend, universe: BoolArray) -> list[FloatArray]:
        self.start_batch()
        caches = [{} for _ in self.streams]
        values = []
        for i, root in enumerate(roots):
            k = i % len(self.streams)
            with self.streams[k]:
                values.append(evaluate(root, fields, backend, caches[k], universe))
        self.finish_batch()
        return values

    @contextmanager
    def scope(self) -> Iterator[None]:
        with self.device, self.cp.cuda.using_allocator(self.pool.malloc):
            yield

    def start_batch(self) -> None:
        ready = self.cp.cuda.Event()
        ready.record()
        for stream in self.streams:
            stream.wait_event(ready)

    def finish_batch(self) -> None:
        for stream in self.streams:
            stream.synchronize()

    def synchronize(self) -> None:
        with self.device:
            self.cp.cuda.get_current_stream().synchronize()

    def stage_host(self, values: Array) -> HostArray:
        """Queue one bounded D2H copy; the caller synchronizes before host use.

        A direct pinned allocation expires with the NumPy view, outside any
        process-global pinned pool. Existing host metadata needs no transfer.
        """
        if isinstance(values, np.ndarray):
            return np.ascontiguousarray(values)
        memory = self.cp.cuda.PinnedMemoryPointer(self.cp.cuda.PinnedMemory(values.nbytes), 0)
        host = np.ndarray(values.shape, dtype=values.dtype, buffer=memory)
        values.get(out=host, blocking=False)
        return host

    def content_leaves(self, values: Array) -> HostArray:
        return self.hasher.leaves(values)

    def batch_sizes(self, cfg: CudaConfig, shape: tuple[int, int], events: int,
                    measurements: int) -> tuple[int, int]:
        free, _ = self.cp.cuda.runtime.memGetInfo()
        D, S = shape
        candidate_bytes = D * S * 8 * 20 + events * S + measurements * 8 * 10
        trial_bytes = D * S * 8 * 2 + events * S + measurements * 8 * 10
        limit = max(1, int(free * .65 / candidate_bytes))
        return (min(cfg.candidate_batch, 1 << (limit.bit_length() - 1)),
                min(cfg.trial_batch, max(1, int(free * .10 / trial_bytes))))

    def measurement_budget(self, cfg: CudaConfig) -> int:
        free, _ = self.cp.cuda.runtime.memGetInfo()
        return min(cfg.measurement_cache_mib * 1048576, int(free * .20))

    def status(self) -> ResourceStatus:
        with self.device:
            free, total = self.cp.cuda.runtime.memGetInfo()
            name = self.cp.cuda.runtime.getDeviceProperties(self.device.id)['name'].decode()
        return ResourceStatus(name, round((total - free) / 1048576))

    def close(self) -> None:
        with self.device:
            self.finish_batch()
            self.device.synchronize()
            self.streams = ()
            del self.hasher
            self.pool.free_all_blocks()


@dataclass
class Runtime:
    settings: Settings
    loaded: StudyData
    backend: Backend
    executor: ExecutionBackend
    resources: DeviceResources
    pipeline: Pipeline
    program_data: orders.PreparedData | None
    program_backend: orders_cuda.PreparedBackend | None
    program_native: ModuleType | None

    @property
    def market(self) -> AShare | UsEquity | CryptoSpot | CryptoPerpetual:
        return self.loaded.market

    @property
    def panel(self) -> Panel:
        return self.loaded.panel

    @property
    def host_spec(self) -> ExecutionSpec | PerpetualSpec:
        return self.loaded.host_spec

    @property
    def window(self) -> Window:
        return self.loaded.window

    def open_program_account(self, batch: int, precision: Literal['mixed', 'paired'],
                             start_event: int, *, owner_count: int = 1,
                             initial_cash: FloatArray | None = None) -> AccountExecutionPort:
        """Create a private account from this runtime's explicitly prepared backend."""
        spec = cast(BarExecution, self.loaded.program_spec)
        data = cast(orders.PreparedData, self.program_data)
        if self.resources.placement.kind == 'cuda':
            return orders_cuda.OrderRuntime(spec, self.settings.account, self.settings.execution,
                batch=batch, precision=precision, array_module=self.backend.xp,
                prepared=cast(orders_cuda.PreparedBackend, self.program_backend),
                prepared_data=data, start_event=start_event, owner_count=owner_count, initial_cash=initial_cash)
        return orders.OrderRuntime(spec, self.settings.account, self.settings.execution,
            batch=batch, precision=precision, native=cast(ModuleType, self.program_native),
            prepared_data=data, start_event=start_event, owner_count=owner_count, initial_cash=initial_cash)

    def replay_batch_size(self, members: int, state_bytes: int = 0, statistics: int | None = None) -> int:
        """Expose a conservative actual-owner replay cap before allocating trials.

        state_bytes is the sum of declared private tensor/window bytes per trial.
        Three account-sized allocations cover live state, detached callback/audit
        views and returned measured buffers. Shared market/input panels are already
        resident and excluded. CUDA uses the same ten-percent free-memory trial
        allowance as the target planner; host uses ten percent of available pages.
        """
        if type(members) is not int or members < 0 or type(state_bytes) is not int or state_bytes < 0:
            raise ValueError('Replay capacity requires nonnegative member and private-state counts')
        V = self.pipeline.schedule.measurement_count if statistics is None else statistics
        if type(V) is not int or V < 1:
            raise ValueError('Replay statistics count must be positive')
        S = len(self.panel.assets)
        Q, K = self.settings.execution.order_capacity, self.settings.execution.command_capacity
        independent = self.settings.account.composition == 'independent'
        M = 1 if independent else max(members, 1)
        accounts = max(members, 1) if independent else 1
        F = K + M*S + 4*Q + (M+1)*(Q+S)
        account_bytes = (S*40 + 14*8 + 5*8 + M*S*26 + M*24 + Q*120 + K*88
                         + F*96 + 8 + V*65 + 40 + 48 + 104)
        per_trial = 3*accounts*account_bytes + state_bytes + 7*8 + V*32
        if self.resources.placement.kind == 'cuda':
            free, _ = self.backend.xp.cuda.runtime.memGetInfo()
            reusable = cast(CudaResources, self.resources).pool.free_bytes()
            budget = int((free+reusable)*.10)
        else:
            budget = int(os.sysconf('SC_AVPHYS_PAGES')*os.sysconf('SC_PAGE_SIZE')*.10)
        limit = budget//per_trial
        if limit < 1:
            raise RuntimeError(f'One replay trial requires {per_trial} bytes, above its {budget}-byte allocation budget')
        return min(max(self.pipeline.candidate_batch, self.pipeline.trial_batch), limit)

    def independent_turnover(self, values: FloatArray, unallocated: FloatArray, state: FloatArray) -> None:
        """Accumulate actual quote turnover over the aggregate bar-opening NAV."""
        xp = self.backend.xp
        packed = xp.ascontiguousarray(values, dtype=xp.float64)
        if self.resources.placement.kind == 'cuda':
            module = cast(orders_cuda.PreparedBackend, self.program_backend).module
            module.get_function('orders_independent_turnover')(((len(packed)+31)//32,), (32,),
                (packed, unallocated, state, np.int64(len(packed)), np.int64(packed.shape[1])))
        else:
            cast(ModuleType, self.program_native).independent_turnover(packed, unallocated, state)

    def sample_independent(self, quantities: FloatArray, equities: FloatArray, unallocated: FloatArray,
                           event: int, state: FloatArray) -> FloatArray:
        """Accumulate a real aggregate valuation with the prepared paired backend."""
        xp = self.backend.xp
        q, e = xp.ascontiguousarray(quantities, dtype=xp.float64), xp.ascontiguousarray(equities, dtype=xp.float64)
        marks = cast(orders.PreparedData, self.program_data).panels[event, 8]
        if self.resources.placement.kind == 'cuda':
            output = xp.empty((len(q), 6), dtype=xp.float64)
            module = cast(orders_cuda.PreparedBackend, self.program_backend).module
            module.get_function('orders_sample_independent')(((len(q)+31)//32,), (32,),
                (q, e, marks, unallocated, state, output, np.int64(len(q)), np.int64(q.shape[1]), np.int64(q.shape[2])))
            return output
        return cast(ModuleType, self.program_native).sample_independent(q, e, marks, unallocated, state)

    def order_statistics(self, returns: FloatArray) -> FloatArray:
        """Use the explicitly prepared paired statistics path for actual accounts."""
        values = self.backend.xp.ascontiguousarray(returns, dtype=self.backend.xp.float64)
        if self.resources.placement.kind == 'cuda':
            output = self.backend.xp.empty((len(values), 5), dtype=self.backend.xp.float64)
            module = cast(orders_cuda.PreparedBackend, self.program_backend).module
            module.get_function('k_statistics_paired')((len(values),), (128,),
                (values, np.int32(values.shape[1]), np.int32(0), np.float64(self.pipeline.ppy), output))
            return output
        return cast(ModuleType, self.program_native).statistics(values, self.pipeline.ppy)

    def close(self) -> None:
        self.pipeline.close()
        self.loaded.panel.release()
        self.executor.close()
        self.program_data = None
        self.program_backend = None
        self.program_native = None
        self.backend.close()
        self.resources.close()

    def status(self) -> ResourceStatus:
        return self.resources.status()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 tb: TracebackType | None) -> None:
        self.close()


def plan_precision(requested: str, supported: frozenset[str]) -> PrecisionPlan:
    effective = 'native' if supported == frozenset({'native'}) else requested
    if effective not in supported:
        raise ValueError(f'unsupported precision policy {requested}: {sorted(supported)}')
    reason = 'validated_native_path' if effective == 'native' else 'fixed_study_policy'
    return PrecisionPlan(requested, effective, reason, tuple(sorted(supported)))


def open_runtime(settings: Settings, study: Study) -> Runtime:
    """Open exactly the backend configured for this explicit study snapshot."""
    name = settings.backend
    with ExitStack() as cleanup:
        if name == 'cuda':
            import cupy as cp

            from foundry.backtest import cuda
            resources = CudaResources(cp, settings.cuda.streams)
            prepare_execution = lambda: cuda.prepare(settings.account.gross_limit)
            if study.market == 'crypto_perpetual':
                from foundry.backtest import perpetual_cpu, perpetual_cuda
                prepare_execution = lambda: perpetual_cuda.prepare(perpetual_cpu.prepare(), settings.cuda.perpetual_host_cells)
        elif name in ('cpu', 'portable'):
            from foundry.backtest import cpu
            from foundry.ops import _ops
            resources = HostResources(settings.cpu, _ops.worker_init)
            prepare_execution = lambda: cpu.prepare(resources.workers, settings.account.gross_limit)
            if study.market == 'crypto_perpetual':
                from foundry.backtest import perpetual_cpu
                prepare_execution = lambda: perpetual_cpu.prepare(resources.workers)
        else:
            raise ValueError(f'Unknown execution backend: {name}')
        cleanup.callback(resources.close)
        with resources.scope():
            backend = prepare_backend(name)
            cleanup.callback(backend.close)
            executor = prepare_execution()
            cleanup.callback(executor.close)
            loaded = load_study(settings, study, backend)
            cleanup.callback(loaded.panel.release)
            if settings.execution.model == 'orders' and loaded.program_spec is None:
                raise ValueError('The orders model requires a market-bars-v2 artifact with explicit execution rules')
            program_data, program_backend, program_native = None, None, None
            if loaded.program_spec is not None:
                program_data = orders.prepare_data(loaded.program_spec, backend.xp)
                if name == 'cuda':
                    program_backend = orders_cuda.prepare(backend.xp)
                else:
                    from foundry.backtest import _orders
                    program_native = _orders
            precisions = frozenset({'mixed', 'paired'}) if settings.execution.model == 'orders' else executor.precisions
            context = study_snapshot(loaded, settings, plan_precision(settings.execution.precision, precisions))
            pipeline = Pipeline(loaded, backend, executor, resources, settings, context)
            cleanup.callback(pipeline.close)
        runtime = Runtime(settings, loaded, backend, executor, resources, pipeline,
                          program_data, program_backend, program_native)
        cleanup.pop_all()  # Ownership now belongs to Runtime.close.
        return runtime
