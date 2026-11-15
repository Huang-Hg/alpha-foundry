"""Public parameter values and market presets; no configuration files or implicit IO."""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal, TypedDict, Unpack

ROOT = Path(__file__).resolve().parents[1]

def path(value: str) -> Path:
    """A configured path, relative to the repository root unless absolute."""
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


@dataclass(frozen=True)
class ExecutionConfig:
    backend: str = 'cuda'


@dataclass(frozen=True)
class UsEquityConfig:
    name: Literal['us_equity'] = field(default='us_equity', init=False)
    data_dir: str = './data/us_equity_daily'
    start: str = '2025-07-01'
    end: str = '2026-06-30'
    cost_rate: float = 0.0005
    daily_borrow: float = 0.0


@dataclass(frozen=True)
class AShareConfig:
    name: Literal['a_share'] = field(default='a_share', init=False)
    data_dir: str = './data/a_share_pit'
    start: str = '2024-03-01'
    end: str = '2026-06-30'
    commission: float = 0.00025
    stamp_tax_sell: float = 0.0005
    transfer_fee: float = 1e-05


@dataclass(frozen=True)
class CryptoSpotConfig:
    name: Literal['crypto_spot'] = field(default='crypto_spot', init=False)
    data_dir: str = './data/crypto_spot_daily'
    start: str = '2024-01-01'
    end: str = '2025-12-31'
    cost_rate: float = 0.001


@dataclass(frozen=True)
class CryptoPerpetualConfig:
    name: Literal['crypto_perpetual'] = field(default='crypto_perpetual', init=False)
    data_dir: str = './data/crypto_perpetual'
    history_start: str = '2025-01-01'
    start: str = '2025-07-01'
    end: str = '2025-12-31'
    symbols: str = 'BTCUSDT ETHUSDT SOLUSDT'
    leverage: float = 5.0
    cost_rate: float = 0.0004
    maintenance_rate: float = 0.005
    liquidation_fee: float = 0.0005

    def __post_init__(self):
        if not (math.isfinite(self.leverage) and self.leverage >= 1
                and 0 <= self.cost_rate < 1 / self.leverage
                and 0 < self.maintenance_rate < 1 / self.leverage
                and 0 <= self.liquidation_fee < self.maintenance_rate):
            raise ValueError('Invalid isolated-perpetual margin or fee parameters')


@dataclass(frozen=True)
class ExprConfig:
    max_cost: int = 25


@dataclass(frozen=True)
class CudaConfig:
    streams: int = 8
    candidate_batch: int = 256
    trial_batch: int = 16384
    precision: str = 'paired'
    return_atol: float = 1e-06
    perpetual_host_cells: int = 100000

    def __post_init__(self):
        if self.streams <= 0 or self.candidate_batch <= 0 or self.trial_batch <= 0:
            raise ValueError('CUDA parameters stream count and batch sizes must be positive')
        if self.precision not in ('mixed', 'paired'):
            raise ValueError('CUDA parameters precision must be mixed or paired')
        if not (math.isfinite(self.return_atol) and self.return_atol > 0):
            raise ValueError('CUDA parameters return error budget must be positive')
        if self.perpetual_host_cells < 0:
            raise ValueError('CUDA parameters perpetual_host_cells must be nonnegative')


@dataclass(frozen=True)
class PortfolioConfig:
    artifact_dir: str = './var/candidates'
    weight_sum_atol: float = 1e-07
    metric_atol: float = 1e-10
    min_sharpe_gain: float = 0.01
    weight_steps: int = 20
    recent_halflife_days: int = 1095
    max_members: int = 40
    sharpe_scale: float = 0.1
    growth_scale: float = 0.02


    def __post_init__(self):
        if not all(math.isfinite(x) and x > 0 for x in (
                self.metric_atol, self.weight_sum_atol, self.sharpe_scale, self.growth_scale)):
            raise ValueError('Admission parameters utility scales must be positive and finite')


@dataclass(frozen=True)
class SearchConfig:
    violation_scale_wsharpe: float = 0.081
    violation_scale_sharpe: float = 0.063
    violation_scale_cagr: float = 0.079
    violation_scale_growth: float = 0.024
    mode: str = 'gfn'
    seed_portfolio: str = 'equal'
    objective: str = 'portfolio'
    noise_control: bool = False
    accept: bool = False
    seed: int = 25
    epochs: int = 5
    batch_size: int = 128
    probes: int = 2
    max_tokens: int = 13
    max_ts_depth: int = 2
    train_steps: int = 4
    learning_rate: float = 0.002
    opd_strength: float = 0.1
    rho_max: float = 0.9
    archive_size: int = 32
    reweight_iterations: int = 3
    reweight_batch: int = 32
    reweight_every: int = 5

    def __post_init__(self):
        if not all(math.isfinite(x) and x > 0 for x in (
                self.violation_scale_wsharpe, self.violation_scale_sharpe,
                self.violation_scale_cagr, self.violation_scale_growth)):
            raise ValueError('Search violation scales must be positive and finite')
        if not (self.mode in ('gfn','random') and self.seed_portfolio in ('current','equal')
                and self.objective in ('portfolio','single')
                and 1 <= self.batch_size <= 8192 and 0 <= self.probes <= self.batch_size
                and self.epochs > 0 and 3 <= self.max_tokens <= 64 and 0 <= self.max_ts_depth <= 64
                and 1 <= self.train_steps <= 100 and 0 <= self.archive_size <= 128
                and 0 <= self.reweight_iterations <= 100 and 2 <= self.reweight_batch <= 8192 and self.reweight_every > 0
                and self.seed >= 0 and 0 < self.learning_rate <= .1
                and 0 <= self.opd_strength <= 10 and 0 < self.rho_max <= 1):
            raise ValueError('Invalid Search parameters configuration')


@dataclass(frozen=True)
class SingleConfig:
    calibration_sample: int = 8192
    anchor_correlation: float = 0.7
    anchor_sharpe: float = 1.5
    recent_years: int = 2
    recent_year_sharpe: float = 1.0
    weight_sharpe: float = 0.6
    weight_drawdown: float = 0.2
    weight_return: float = 0.2
    sharpe_scale: float = 2.0
    drawdown_full: float = 0.6
    return_scale: float = 1.0
    violation_scale_sharpe: float = 1.3
    violation_scale_year: float = 3.35

    def __post_init__(self):
        weights = (self.weight_sharpe, self.weight_drawdown, self.weight_return)
        if not (64 <= self.calibration_sample <= 65536 and 0 <= self.anchor_correlation < 1 and self.anchor_sharpe > 0
                and self.recent_years >= 0 and min(weights) >= 0
                and abs(sum(weights) - 1) < 1e-9 and self.weight_sharpe >= max(weights)
                and min(self.sharpe_scale, self.drawdown_full, self.return_scale,
                        self.violation_scale_sharpe, self.violation_scale_year) > 0):
            raise ValueError('Invalid Single-strategy parameters configuration')


@dataclass(frozen=True)
class MonitorConfig:
    frame_seconds: float = 0.1
    push_seconds: float = 2.0


MarketConfig = UsEquityConfig | AShareConfig | CryptoSpotConfig | CryptoPerpetualConfig


@dataclass(frozen=True)
class Settings:
    market: MarketConfig = field(default_factory=UsEquityConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    expr: ExprConfig = field(default_factory=ExprConfig)
    cuda: CudaConfig = field(default_factory=CudaConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    single: SingleConfig = field(default_factory=SingleConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)


class Parameters(TypedDict, total=False):
    """Keyword parameters shared by public sessions, miners and service startup."""
    market: Literal['us_equity', 'a_share', 'crypto_spot', 'crypto_perpetual']
    backend: Literal['cpu', 'cuda', 'portable']
    data_dir: str
    start: str
    end: str
    cost_rate: float
    daily_borrow: float
    commission: float
    stamp_tax_sell: float
    transfer_fee: float
    history_start: str
    symbols: str
    leverage: float
    maintenance_rate: float
    liquidation_fee: float
    cuda: CudaConfig
    admission: PortfolioConfig
    operators: ExprConfig
    search: SearchConfig
    single: SingleConfig
    monitor: MonitorConfig


PARAMETER_GROUPS = dict(cuda=('cuda', CudaConfig), admission=('portfolio', PortfolioConfig),
                        operators=('expr', ExprConfig), search=('search', SearchConfig),
                        single=('single', SingleConfig), monitor=('monitor', MonitorConfig))


def configure(**parameters: Unpack[Parameters]) -> Settings:
    """Resolve one explicit parameter set against the selected market's defaults."""
    settings = Settings()
    name = parameters.pop('market', settings.market.name)
    factories = {'us_equity': UsEquityConfig, 'a_share': AShareConfig,
                 'crypto_spot': CryptoSpotConfig, 'crypto_perpetual': CryptoPerpetualConfig}
    if name not in factories:
        raise ValueError(f'Unknown market: {name}')
    backend = parameters.pop('backend', settings.execution.backend)
    if backend not in ('cpu', 'cuda', 'portable'):
        raise ValueError(f'Unknown execution backend: {backend}')
    groups = {field_name: parameters.pop(public_name, getattr(settings, field_name))
              for public_name, (field_name, _) in PARAMETER_GROUPS.items()}
    for public_name, (field_name, kind) in PARAMETER_GROUPS.items():
        if not isinstance(groups[field_name], kind):
            raise TypeError(f'{public_name} requires {kind.__name__}')
    selected = factories[name](**parameters)
    return replace(settings, market=selected, execution=ExecutionConfig(backend), **groups)



def parameters_from_json(record: dict) -> dict:
    """Decode nested public parameter groups at a JSON request boundary."""
    return {name: PARAMETER_GROUPS[name][1](**value) if name in PARAMETER_GROUPS else value
            for name, value in record.items()}
