"""Public parameter values and market presets; no configuration files or implicit IO."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal, TypedDict, Unpack

ROOT = Path(__file__).resolve().parents[1]

def path(value: str) -> Path:
    """A configured path, relative to the repository root unless absolute."""
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def frequency_nanoseconds(value: str) -> int:
    """Resolve a fixed UTC duration; calendar months are not fixed durations."""
    match = re.fullmatch(r'([1-9][0-9]*)(s|min|h|D)', value) if isinstance(value, str) else None
    if match is None:
        raise ValueError('Clock frequencies require a positive integer followed by s, min, h or D')
    result = int(match[1]) * {'s': 1, 'min': 60, 'h': 3600, 'D': 86400}[match[2]] * 1000000000
    if result > 9223372036854775807:
        raise ValueError('Clock frequency exceeds the UTC nanosecond range')
    return result


@dataclass(frozen=True)
class Clock:
    """Independent fixed decision and valuation periods on the market clock."""
    decision: str = '1D'
    statistics: str = '1D'

    def __post_init__(self) -> None:
        frequency_nanoseconds(self.decision)
        frequency_nanoseconds(self.statistics)


@dataclass(frozen=True)
class Sizing:
    """Score mapping; independent targets use scale without asset-count dilution."""
    rule: Literal['equal_slots', 'independent'] = 'equal_slots'
    scale: float = 1.

    def __post_init__(self) -> None:
        if self.rule not in ('equal_slots', 'independent'):
            raise ValueError('Sizing rule must be equal_slots or independent')
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError('Sizing scale must be positive and finite')
        if self.rule == 'equal_slots' and self.scale != 1.:
            raise ValueError('equal_slots requires scale=1; choose independent for explicit scaling')


@dataclass(frozen=True)
class Account:
    """Collateral leverage and allowed gross notional/equity are separate limits."""
    margin: Literal['cash', 'isolated'] = 'cash'
    leverage: float = 1.
    gross_limit: float = 1.
    initial_cash: float = 1.

    def __post_init__(self) -> None:
        if self.margin not in ('cash', 'isolated'):
            raise ValueError('Account margin must be cash or isolated')
        if not (math.isfinite(self.leverage) and self.leverage >= 1
                and math.isfinite(self.gross_limit) and 0 < self.gross_limit <= self.leverage):
            raise ValueError('Account requires finite leverage >= 1 and 0 < gross_limit <= leverage')
        if self.margin == 'cash' and self.leverage != 1:
            raise ValueError('Cash accounts require leverage=1')
        if not math.isfinite(self.initial_cash) or self.initial_cash <= 0:
            raise ValueError('Initial quote cash must be positive and finite')


@dataclass(frozen=True)
class Execution:
    """Bar fill rule; next_open means the next base bar after the decision."""
    fill: Literal['next_open'] = 'next_open'
    precision: Literal['paired', 'mixed'] = 'paired'
    bar_path: Literal['open_low_high_close', 'open_high_low_close'] = 'open_low_high_close'
    participation: float = 1.
    order_capacity: int = 32
    command_capacity: int = 32
    quantity_step: float = 0.
    price_tick: float = 0.

    def __post_init__(self) -> None:
        if self.fill != 'next_open':
            raise ValueError('The target execution plan supports only next_open fills')
        if self.precision not in ('paired', 'mixed'):
            raise ValueError('Execution precision must be paired or mixed')
        if self.bar_path not in ('open_low_high_close', 'open_high_low_close'):
            raise ValueError('Unknown deterministic Bar path')
        if not math.isfinite(self.participation) or not 0 < self.participation <= 1:
            raise ValueError('Volume participation must lie in (0,1]')
        if any(type(value) is not int or value < 1 for value in (self.order_capacity, self.command_capacity)):
            raise ValueError('Order and command capacities must be positive integers')
        if any(not math.isfinite(value) or value < 0 for value in (self.quantity_step, self.price_tick)):
            raise ValueError('Quantity and price steps must be finite and nonnegative; zero means continuous')


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
    cost_rate: float = 0.0004
    maintenance_rate: float = 0.005
    liquidation_fee: float = 0.0005

    def __post_init__(self):
        if not (0 <= self.cost_rate < 1
                and 0 < self.maintenance_rate < 1
                and 0 <= self.liquidation_fee < self.maintenance_rate):
            raise ValueError('Invalid isolated-perpetual margin or fee parameters')


@dataclass(frozen=True)
class ExprConfig:
    max_cost: int = 25


@dataclass(frozen=True)
class CpuConfig:
    workers: int = 8
    candidate_batch: int = 32
    measurement_cache_mib: int = 32

    def __post_init__(self):
        if any(type(x) is not int or x < 1 for x in (self.workers, self.candidate_batch)):
            raise ValueError('CPU workers and candidate_batch must be positive integers')
        if type(self.measurement_cache_mib) is not int or self.measurement_cache_mib < 0:
            raise ValueError('CPU measurement_cache_mib must be a nonnegative integer')


@dataclass(frozen=True)
class CudaConfig:
    streams: int = 8
    candidate_batch: int = 256
    trial_batch: int = 16384
    return_atol: float = 1e-06
    perpetual_host_cells: int = 100000
    measurement_cache_mib: int = 32

    def __post_init__(self):
        if self.streams <= 0 or self.candidate_batch <= 0 or self.trial_batch <= 0:
            raise ValueError('CUDA parameters stream count and batch sizes must be positive')
        if not (math.isfinite(self.return_atol) and self.return_atol > 0):
            raise ValueError('CUDA parameters return error budget must be positive')
        if self.perpetual_host_cells < 0:
            raise ValueError('CUDA parameters perpetual_host_cells must be nonnegative')
        if type(self.measurement_cache_mib) is not int or self.measurement_cache_mib < 0:
            raise ValueError('CUDA parameters measurement_cache_mib must be a nonnegative integer')


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
    backend: Literal['cpu', 'cuda', 'portable'] = 'cuda'
    clock: Clock = field(default_factory=Clock)
    sizing: Sizing = field(default_factory=Sizing)
    account: Account = field(default_factory=Account)
    execution: Execution = field(default_factory=Execution)
    expr: ExprConfig = field(default_factory=ExprConfig)
    cuda: CudaConfig = field(default_factory=CudaConfig)
    cpu: CpuConfig = field(default_factory=CpuConfig)
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
    maintenance_rate: float
    liquidation_fee: float
    clock: Clock
    sizing: Sizing
    account: Account
    execution: Execution
    cuda: CudaConfig
    cpu: CpuConfig
    admission: PortfolioConfig
    operators: ExprConfig
    search: SearchConfig
    single: SingleConfig
    monitor: MonitorConfig


PARAMETER_GROUPS = dict(clock=('clock', Clock), sizing=('sizing', Sizing), account=('account', Account),
                        execution=('execution', Execution),
                        cuda=('cuda', CudaConfig), cpu=('cpu', CpuConfig), admission=('portfolio', PortfolioConfig),
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
    backend = parameters.pop('backend', settings.backend)
    if backend not in ('cpu', 'cuda', 'portable'):
        raise ValueError(f'Unknown execution backend: {backend}')
    account = Account('isolated', 5., 5.) if name == 'crypto_perpetual' else Account()
    settings = replace(settings, account=account)
    groups = {field_name: parameters.pop(public_name, getattr(settings, field_name))
              for public_name, (field_name, _) in PARAMETER_GROUPS.items()}
    for public_name, (field_name, kind) in PARAMETER_GROUPS.items():
        if not isinstance(groups[field_name], kind):
            raise TypeError(f'{public_name} requires {kind.__name__}')
    selected = factories[name](**parameters)
    account = groups['account']
    if name in ('us_equity', 'a_share') and groups['clock'] != Clock():
        raise ValueError('Equity data and execution plans currently require daily decision and statistics clocks')
    if (name == 'crypto_perpetual') != (account.margin == 'isolated'):
        raise ValueError('Isolated accounts require crypto_perpetual; other markets require cash accounts')
    if isinstance(selected, CryptoPerpetualConfig) and not (
            selected.cost_rate < 1 / account.leverage
            and selected.maintenance_rate < 1 / account.leverage):
        raise ValueError('Perpetual trading and maintenance rates must be below isolated collateral fraction')
    return replace(settings, market=selected, backend=backend, **groups)



def parameters_from_json(record: dict) -> dict:
    """Decode nested public parameter groups at a JSON request boundary."""
    return {name: PARAMETER_GROUPS[name][1](**value) if name in PARAMETER_GROUPS else value
            for name, value in record.items()}
