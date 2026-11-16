"""Explicit public entry points; importing Foundry opens no resources."""
from pathlib import Path
from typing import Unpack

from foundry.config import Parameters, configure


def open(portfolio: str | Path | None = None, **parameters: Unpack[Parameters]):
    """Open an explicitly parameterized session; an omitted portfolio stays in memory."""
    from foundry.study.session import open_session, DiscoveryServices
    from foundry.search.diagnostics import diagnose_current
    from foundry.search.robustness import judge_family
    from foundry.rpc.codec import describe
    return open_session(configure(**parameters), portfolio, describe, DiscoveryServices(diagnose_current, judge_family))


def connect(address: tuple[str, int] | None = None):
    """Connect to an existing service address or the explicitly attached endpoint."""
    from foundry.rpc.client import RemoteBackend
    from foundry.study.session import Session
    from contextlib import ExitStack
    with ExitStack() as cleanup:
        backend = RemoteBackend(address)
        cleanup.callback(backend.close)
        session = Session(backend)
        cleanup.pop_all()
        return session


def serve(portfolio: str | Path = 'production/library.json', **parameters: Unpack[Parameters]):
    """Serve a parameterized research session until explicitly stopped."""
    from foundry.rpc.server import Server
    with open(portfolio, **parameters) as session:
        backend = session._backend
        return Server(backend).run()


def export(report, output: str | Path):
    """Write a structured report as strict JSON at an explicit output path."""
    from dataclasses import asdict
    import json
    from foundry.study.store import publish
    from foundry.rpc.codec import json_value
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    publish(target, json.dumps(json_value(asdict(report)), ensure_ascii=False, allow_nan=False,
                                    indent=2).encode('utf-8'))


def schema():
    """Discover the public machine protocol without loading data or a backend."""
    from foundry.rpc.schema import describe
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
    from foundry.search.diagnostics import CONTEXT_NAMES, diagnose_current
    from foundry.search.robustness import judge_family
    from foundry.search.driver import MiningDriver
    from foundry.search.proposer.gfn import GfnProposer
    from foundry.search.proposer.random import RandomProposer
    from foundry.rpc.codec import describe, checkpoint_bytes, checkpoint_from_bytes, telemetry_bytes
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
            owner = open_session(settings, portfolio, describe, DiscoveryServices(diagnose_current, judge_family),
                initial=None if checkpoint is None else checkpoint.portfolio, persist=cfg.accept)
        with owner as active:
            port = active.resident
            cfg = port.settings.search
            if proposers is None:
                factory = GfnProposer if cfg.mode == 'gfn' else RandomProposer
                learning = dict(probes=cfg.probes, steps=cfg.train_steps, learning_rate=cfg.learning_rate,
                                opd_strength=cfg.opd_strength) if cfg.mode == 'gfn' else {}
                proposers = (factory(port.vocabulary, cfg.seed, cfg.max_tokens, port.settings.expr.max_cost,
                    cfg.max_ts_depth, port.window_caps, len(CONTEXT_NAMES), **learning),)
            signature = hashlib.sha256(json.dumps(dict(snapshot=port.context.snapshot.id,
                parameters=asdict(port.settings)), sort_keys=True).encode()).hexdigest()
            telemetry = FileTelemetrySink(directory, ROOT / 'var' / 'monitor' / 'live.json', telemetry_bytes)
            return MiningDriver(active, proposers, checkpoints, telemetry, signature=signature).run()


def import_perpetual(output: str | Path, **parameters: Unpack[Parameters]):
    """Create a verified perpetual dataset from explicit symbols and dates."""
    from foundry.market.perpetual import import_perpetual as acquire
    from foundry.study.store import publish
    return acquire(Path(output), configure(market='crypto_perpetual', **parameters).market, publish)


def benchmark(output: str | Path, *, suite='foundry', sizes=None, assets=8, backends=('cpu', 'cuda'),
              spectre_python=None, **parameters: Unpack[Parameters]):
    """Run the selected native benchmark suite and generate its report bundle."""
    if suite == 'nonformula':
        from validation.benchmarks.nonformula import run
        if tuple(backends) != ('cuda',) or sizes is not None or assets != 8 or spectre_python is not None:
            raise ValueError('nonformula requires backends=("cuda",); set scale through Search parameters')
        return run(output, **parameters)
    if suite == 'discovery':
        from validation.benchmarks.discovery import run
        if tuple(backends) != ('cuda',) or sizes is not None or assets != 8 or spectre_python is not None:
            raise ValueError('discovery requires backends=("cuda",); set scale through Search parameters')
        return run(output, **parameters)
    if suite == 'frameworks':
        import sys
        from validation.benchmarks.competition.run import run
        if parameters:
            raise ValueError('frameworks uses its published common trading contract; omit research parameters')
        if tuple(backends) != ('cpu', 'cuda'):
            raise ValueError('backends applies only to the foundry suite')
        return run(output, sizes=(1, 16, 64) if sizes is None else tuple(sizes), assets=assets,
                   spectre_python=sys.executable if spectre_python is None else spectre_python)
    if suite != 'foundry':
        raise ValueError('benchmark suite must be foundry, frameworks, discovery or nonformula')
    if assets != 8 or spectre_python is not None:
        raise ValueError('assets and spectre_python apply only to the frameworks suite')
    from validation.benchmarks.run import run
    return run(output, sizes=(1, 64, 4096) if sizes is None else tuple(sizes), backends=tuple(backends), **parameters)
