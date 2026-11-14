"""Cloud integration fixtures with explicit studies and no production-library reads."""
from dataclasses import replace

from foundry.config import load_config
from foundry.contracts import Study
from foundry.runtime import open_runtime
from foundry.adapters.library import MemoryLibraryStore
from foundry.adapters.arrays import MemoryArrayStore
from foundry.portfolio.library import Library
from foundry.research import Research


EXPRESSIONS = ('gt(ma_dist_20,0)', 'gt(ma_dist_60,0.05)',
               'gt(ma_dist_20,0.1)', 'gt(ma_dist_60,-0.1)')


def settings_for(market='us_equity', backend='cuda'):
    settings = load_config()
    return replace(settings, market=replace(settings.market, name=market),
                   execution=replace(settings.execution, backend=backend))


def study_for(settings, market):
    cfg = {'us_equity': settings.us_equity, 'a_share': settings.a_share,
           'crypto_spot': settings.crypto_spot, 'crypto_perpetual': settings.crypto_perpetual}[market]
    return Study(market, cfg.start, cfg.end)


def runtime_for(market='us_equity', backend='cuda'):
    settings = settings_for(market, backend)
    return open_runtime(settings, study_for(settings, market))


def research_for(runtime, members=()):
    library = Library(tuple(members), runtime.engine.study)
    return Research(runtime.engine, MemoryLibraryStore(library), MemoryArrayStore())
