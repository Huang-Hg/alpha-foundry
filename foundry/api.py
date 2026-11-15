"""Explicit public entry points; importing Foundry opens no resources."""
from pathlib import Path
from typing import Unpack

from foundry.config import Parameters, configure


def open(portfolio: str | Path | None = None, **parameters: Unpack[Parameters]):
    """Open an explicitly parameterized session; an omitted portfolio stays in memory."""
    from foundry.study.session import open_session
    from foundry.adapters.serialization import describe
    return open_session(configure(**parameters), portfolio, describe)


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
        return Server(backend.runtime, backend.research).run()


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


def mine(output: str | Path, **parameters: Unpack[Parameters]):
    """Run a miner with explicit research and search parameters."""
    from foundry.search.run import run
    return run(Path(output), configure(**parameters))


def import_perpetual(output: str | Path, **parameters: Unpack[Parameters]):
    """Create a verified perpetual dataset from explicit symbols and dates."""
    from foundry.market.perpetual import import_perpetual as acquire
    from foundry.study.store import publish
    return acquire(Path(output), configure(market='crypto_perpetual', **parameters).market, publish)


def benchmark(output: str | Path, *, sizes=(1, 64, 4096), **parameters: Unpack[Parameters]):
    """Run validation and CPU/CUDA benchmarks, then generate the report bundle."""
    from validation.benchmarks.run import run
    return run(output, sizes=sizes, **parameters)
