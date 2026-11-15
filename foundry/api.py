"""Explicit public entry points; importing Foundry opens no resources."""
from pathlib import Path
from typing import Unpack

from foundry.config import Parameters, configure


def open(portfolio: str | Path | None = None, **parameters: Unpack[Parameters]):
    """Open an explicitly parameterized session; an omitted portfolio stays in memory."""
    from foundry.study.session import open_session, DiscoveryServices
    from foundry.search.diagnostics import diagnose
    from foundry.search.robustness import judge_family
    from foundry.adapters.serialization import describe
    return open_session(configure(**parameters), portfolio, describe, DiscoveryServices(diagnose, judge_family))


def connect(address: tuple[str, int] | None = None):
    """Connect to an existing service address or the explicitly attached endpoint."""
    from foundry.adapters.remote_session import RemoteBackend
    from foundry.study.session import Session
    return Session(RemoteBackend(address))


def serve(portfolio: str | Path = 'production/library.json', **parameters: Unpack[Parameters]):
    """Serve a parameterized research session until explicitly stopped."""
    from foundry.service.daemon import Server
    with open(portfolio, **parameters) as session:
        backend = session._backend
        return Server(backend.runtime, backend.research, backend.judge_family).run()


def export(report, output: str | Path):
    """Write a structured report as strict JSON at an explicit output path."""
    from dataclasses import asdict
    import json
    from foundry.study.store import publish
    from foundry.adapters.serialization import json_value
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    publish(target, json.dumps(json_value(asdict(report)), ensure_ascii=False, allow_nan=False,
                                    indent=2).encode('utf-8'))


def schema():
    """Discover the public machine protocol without loading data or a backend."""
    from foundry.service.schema import describe
    return describe()


def mine(output: str | Path, *, session=None, proposers=None, **parameters: Unpack[Parameters]):
    """Run Candidate-native discovery with owned or caller-supplied resources."""
    from dataclasses import asdict
    import hashlib
    import json
    from contextlib import nullcontext
    from foundry.config import ROOT
    from foundry.study.session import open_session, DiscoveryServices
    from foundry.study.store import FileCheckpointStore, FileTelemetrySink, RunLease
    from foundry.search.diagnostics import CONTEXT_NAMES, diagnose
    from foundry.search.robustness import judge_family
    from foundry.search.driver import MiningDriver
    from foundry.search.proposer.gfn import GfnProposer
    from foundry.search.proposer.random import RandomProposer
    from foundry.adapters.serialization import describe, checkpoint_bytes, checkpoint_from_bytes, telemetry_bytes
    directory = Path(output)
    with RunLease(directory):
        checkpoints = FileCheckpointStore(directory / 'state.json', checkpoint_bytes, checkpoint_from_bytes)
        checkpoint = checkpoints.read()
        if session is not None:
            if parameters:
                raise TypeError('a supplied session already owns its resolved parameters')
            owner = nullcontext(session)
        else:
            settings = configure(**parameters)
            cfg = settings.search
            portfolio = None if cfg.seed_portfolio == 'equal' else ROOT / 'production' / 'library.json'
            owner = open_session(settings, portfolio, describe, DiscoveryServices(diagnose, judge_family),
                initial=None if checkpoint is None else checkpoint.portfolio, persist=cfg.accept)
        with owner as active:
            port = active.resident
            cfg = port.settings.search
            if proposers is None:
                factory = GfnProposer if cfg.mode == 'gfn' else RandomProposer
                proposers = (factory(port.vocabulary, cfg.seed, cfg.max_tokens, port.settings.expr.max_cost,
                    cfg.max_ts_depth, port.window_caps, len(CONTEXT_NAMES)),)
            signature = hashlib.sha256(json.dumps(dict(snapshot=port.context.snapshot.id,
                parameters=asdict(port.settings)), sort_keys=True).encode()).hexdigest()
            telemetry = FileTelemetrySink(directory, ROOT / 'var' / 'monitor' / 'live.json', telemetry_bytes)
            return MiningDriver(active, proposers, checkpoints, telemetry, signature=signature).run()


def import_perpetual(output: str | Path, **parameters: Unpack[Parameters]):
    """Create a verified perpetual dataset from explicit symbols and dates."""
    from foundry.market.perpetual import import_perpetual as acquire
    from foundry.study.store import publish
    return acquire(Path(output), configure(market='crypto_perpetual', **parameters).market, publish)


def benchmark(output: str | Path, *, sizes=(1, 64, 4096), **parameters: Unpack[Parameters]):
    """Run validation and CPU/CUDA benchmarks, then generate the report bundle."""
    from validation.benchmarks.run import run
    return run(output, sizes=sizes, **parameters)


def walk_forward(session, fit, predict, **parameters):
    """Fit causal folds, register predictions and report their realized sign stability."""
    from foundry.study.walk_forward import walk_forward as execute
    return execute(session, fit, predict, **parameters)
