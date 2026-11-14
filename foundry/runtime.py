"""Explicit startup assembly. Each runtime owns its compute resources and study."""
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass

import numpy as np

from foundry.adapters.data import load_data
from foundry.adapters.provenance import study_snapshot
from foundry.config import Settings
from foundry.contracts import Backend, LoadedStudy, PrecisionPlan, Study
from foundry.engine import Engine
from foundry.market.a_share import AShare
from foundry.market.crypto_spot import CryptoSpot
from foundry.market.crypto_perpetual import CryptoPerpetual
from foundry.market.us_equity import UsEquity
from foundry.ops.registry import MAX_WINDOW


def _all_supported(tree):
    return ()


def _close_host_backend():
    """Host operator tables contain no owned execution resources."""


def prepare_backend(name: str) -> Backend:
    """Initialize one explicitly chosen implementation; unsupported names fail."""
    if name == 'cuda':
        import cupy as cp
        from foundry.ops import cuda
        operators = cuda.prepare()
        return Backend(name, cp, operators.IMPL, operators.unsupported, operators.max_window,
                       operators.max_pair_window, cp.asnumpy, operators.close)
    if name == 'cpu':
        from foundry.ops import cpu
        return Backend(name, np, cpu.IMPL, _all_supported, MAX_WINDOW, MAX_WINDOW,
                       np.asarray, _close_host_backend)
    if name == 'portable':
        from foundry.ops import portable
        return Backend(name, np, portable.IMPL, _all_supported, MAX_WINDOW, MAX_WINDOW,
                       np.asarray, _close_host_backend)
    raise ValueError(f'Unknown execution backend: {name}')


def configured_study(settings: Settings) -> Study:
    cfg = {'a_share': settings.a_share, 'us_equity': settings.us_equity,
           'crypto_spot': settings.crypto_spot, 'crypto_perpetual': settings.crypto_perpetual}[settings.market.name]
    return Study(settings.market.name, cfg.start, cfg.end)


def load_study(settings: Settings, study: Study, backend: Backend) -> LoadedStudy:
    """Read one artifact snapshot, then compile its market rules without IO."""
    markets = {'a_share': lambda: AShare(settings.a_share.commission, settings.a_share.transfer_fee,
                                         settings.a_share.stamp_tax_sell),
               'us_equity': lambda: UsEquity(settings.us_equity.cost_rate, settings.us_equity.daily_borrow),
               'crypto_spot': lambda: CryptoSpot(settings.crypto_spot.cost_rate),
               'crypto_perpetual': lambda: CryptoPerpetual(settings.crypto_perpetual.leverage,
                    settings.crypto_perpetual.cost_rate, settings.crypto_perpetual.maintenance_rate,
                    settings.crypto_perpetual.liquidation_fee)}
    market = markets[study.market]()
    return market.compile(load_data(settings, study), backend, study)


class HostResources:
    def __init__(self):
        self.streams = (nullcontext(),)

    def scope(self):
        return nullcontext()

    def start_batch(self):
        pass

    def finish_batch(self):
        pass

    def synchronize(self):
        pass

    def batch_sizes(self, cfg, shape):
        return cfg.candidate_batch, cfg.trial_batch

    def status(self):
        return ResourceStatus('CPU', 0)

    def close(self):
        self.streams = ()


class CudaResources:
    def __init__(self, cp, stream_count):
        self.cp = cp
        self.device = cp.cuda.Device()
        self.pool = cp.cuda.MemoryPool()
        self.streams = tuple(cp.cuda.Stream(non_blocking=True) for _ in range(stream_count))

    @contextmanager
    def scope(self):
        with self.device, self.cp.cuda.using_allocator(self.pool.malloc):
            yield

    def start_batch(self):
        ready = self.cp.cuda.Event()
        ready.record()
        for stream in self.streams:
            stream.wait_event(ready)

    def finish_batch(self):
        for stream in self.streams:
            stream.synchronize()

    def synchronize(self):
        with self.device:
            self.cp.cuda.get_current_stream().synchronize()

    def batch_sizes(self, cfg, shape):
        free, _ = self.cp.cuda.runtime.memGetInfo()
        T, S = shape
        limit = max(1, int(free * .65 / (T * S * 8 * 20)))
        return (min(cfg.candidate_batch, 1 << (limit.bit_length() - 1)),
                min(cfg.trial_batch, max(1, int(free * .10 / (T * 8 * 10)))))

    def status(self):
        with self.device:
            free, total = self.cp.cuda.runtime.memGetInfo()
            name = self.cp.cuda.runtime.getDeviceProperties(self.device.id)['name'].decode()
        return ResourceStatus(name, round((total - free) / 1048576))

    def close(self):
        with self.device:
            self.finish_batch()
            self.device.synchronize()
            self.streams = ()
            self.pool.free_all_blocks()


@dataclass(frozen=True)
class ResourceStatus:
    device: str
    used_mib: int


@dataclass
class Runtime:
    settings: Settings
    loaded: LoadedStudy
    backend: Backend
    executor: object
    resources: object
    engine: Engine

    @property
    def market(self):
        return self.loaded.market

    @property
    def panel(self):
        return self.loaded.panel

    @property
    def host_spec(self):
        return self.loaded.host_spec

    @property
    def window(self):
        return self.loaded.window

    def close(self):
        self.engine.close()
        self.executor.close()
        self.backend.close()
        self.resources.close()

    def status(self) -> ResourceStatus:
        return self.resources.status()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def plan_precision(requested, supported):
    required = {'auto': {'f64', 'mixed'}, 'paired': {'f64', 'paired'}, 'f64': {'f64'}}[requested]
    if required <= supported:
        return PrecisionPlan(requested, requested, 'requested_policy_supported', tuple(sorted(supported)))
    if supported == frozenset({'f64'}):
        return PrecisionPlan(requested, 'f64', 'executor_f64_only', ('f64',))
    raise ValueError(f'unsupported precision policy {requested}: executor supports {sorted(supported)}')


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
                from foundry.backtest import perpetual_cuda
                prepare_execution = perpetual_cuda.prepare
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
            context = study_snapshot(loaded, settings, plan_precision(settings.cuda.precision, executor.precisions))
            engine = Engine(loaded, backend, executor, resources, settings, context)
            cleanup.callback(engine.close)
        runtime = Runtime(settings, loaded, backend, executor, resources, engine)
        cleanup.pop_all()  # Ownership now belongs to Runtime.close.
        return runtime
