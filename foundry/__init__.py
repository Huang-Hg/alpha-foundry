"""CUDA-accelerated research with one public entry point for people and programs."""
from foundry.api import benchmark, connect, export, import_perpetual, mine, open, schema, serve
from foundry.config import (CudaConfig as CUDA, ExprConfig as Operators, MonitorConfig as Monitor,
                            PortfolioConfig as Admission, SearchConfig as Search, SingleConfig as Single)
from foundry.study.walk_forward import walk_forward
from foundry.search.objective import Discovery
from foundry.search.proposer.external import ExternalProposer
from foundry.search.proposer.gfn import GfnProposer
from foundry.search.proposer.random import RandomProposer
from foundry.types import Candidate, CandidateGrid, DataGrid, Fold, Formula

__all__ = ('open', 'connect', 'serve', 'export', 'schema', 'mine', 'import_perpetual', 'benchmark', 'DataGrid',
           'Discovery', 'ExternalProposer', 'GfnProposer', 'RandomProposer', 'Candidate', 'CandidateGrid', 'Formula', 'Fold', 'walk_forward', 'CUDA', 'Admission', 'Operators', 'Search', 'Single', 'Monitor')
