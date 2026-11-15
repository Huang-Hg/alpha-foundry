"""Explicit startup assembly. Each runtime owns its compute resources and study."""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
from types import ModuleType, TracebackType
from typing import TYPE_CHECKING, Self

import numpy as np

from foundry.config import (
    AShareConfig,
    CryptoSpotConfig,
    CudaConfig,
    Settings,
    UsEquityConfig,
)
from foundry.market.a_share import AShare
from foundry.market.crypto_spot import CryptoSpot
from foundry.market.data import load_data
from foundry.market.perpetual import CryptoPerpetual
from foundry.market.us_equity import UsEquity
from foundry.ops.registry import MAX_WINDOW
from foundry.study.pipeline import DeviceResources, Pipeline, ResourceStatus
from foundry.study.store import study_snapshot
from foundry.types import (
    Backend,
    ExecutionSpec,
    ParsedFormula,
    PerpetualSpec,
    PrecisionPlan,
    Study,
    Window,
)

if TYPE_CHECKING:
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
        market = AShare(cfg.commission, cfg.transfer_fee, cfg.stamp_tax_sell)
    elif isinstance(cfg, UsEquityConfig):
        market = UsEquity(cfg.cost_rate, cfg.daily_borrow)
    elif isinstance(cfg, CryptoSpotConfig):
        market = CryptoSpot(cfg.cost_rate)
    else:
        market = CryptoPerpetual(cfg.leverage, cfg.cost_rate, cfg.maintenance_rate, cfg.liquidation_fee)
    return market.compile(load_data(settings, study), backend, study)


class HostResources:
    def __init__(self) -> None:
        self.streams = (nullcontext(),)

    def scope(self) -> nullcontext[None]:
        return nullcontext()

    def start_batch(self) -> None:
        pass

    def finish_batch(self) -> None:
        pass

    def synchronize(self) -> None:
        pass

    def batch_sizes(self, cfg: CudaConfig, shape: tuple[int, int]) -> tuple[int, int]:
        return cfg.candidate_batch, cfg.trial_batch

    def status(self) -> ResourceStatus:
        return ResourceStatus('CPU', 0)

    def close(self) -> None:
        self.streams = ()


class CudaResources:
    def __init__(self, cp: ModuleType, stream_count: int):
        self.cp = cp
        self.device = cp.cuda.Device()
        self.pool = cp.cuda.MemoryPool()
        self.streams = tuple(cp.cuda.Stream(non_blocking=True) for _ in range(stream_count))

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

    def batch_sizes(self, cfg: CudaConfig, shape: tuple[int, int]) -> tuple[int, int]:
        free, _ = self.cp.cuda.runtime.memGetInfo()
        T, S = shape
        limit = max(1, int(free * .65 / (T * S * 8 * 20)))
        return (min(cfg.candidate_batch, 1 << (limit.bit_length() - 1)),
                min(cfg.trial_batch, max(1, int(free * .10 / (T * 8 * 10)))))

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
            self.pool.free_all_blocks()


@dataclass
class Runtime:
    settings: Settings
    loaded: StudyData
    backend: Backend
    executor: ExecutionBackend
    resources: DeviceResources
    pipeline: Pipeline

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

    def close(self) -> None:
        self.pipeline.close()
        self.loaded.panel.release()
        self.executor.close()
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
    name = settings.execution.backend
    with ExitStack() as cleanup:
        if name == 'cuda':
            import cupy as cp

            from foundry.backtest import cuda
            resources = CudaResources(cp, settings.cuda.streams)
            prepare_execution = cuda.prepare
            if study.market == 'crypto_perpetual':
                from foundry.backtest import perpetual_cpu, perpetual_cuda
                prepare_execution = lambda: perpetual_cuda.prepare(perpetual_cpu.prepare(), settings.cuda.perpetual_host_cells)
        elif name in ('cpu', 'portable'):
            from foundry.backtest import cpu
            resources = HostResources()
            prepare_execution = cpu.prepare
            if study.market == 'crypto_perpetual':
                from foundry.backtest import perpetual_cpu
                prepare_execution = perpetual_cpu.prepare
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
            context = study_snapshot(loaded, settings, plan_precision(settings.cuda.precision,
                                      executor.precisions))
            pipeline = Pipeline(loaded, backend, executor, resources, settings, context)
            cleanup.callback(pipeline.close)
        runtime = Runtime(settings, loaded, backend, executor, resources, pipeline)
        cleanup.pop_all()  # Ownership now belongs to Runtime.close.
        return runtime
