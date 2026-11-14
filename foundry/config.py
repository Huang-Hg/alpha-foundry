"""Configuration: one INI file (config/config.ini), one dataclass per section.

`ini(section, key, default)` declares a field. Only load_config reads the INI,
at explicit application startup; domain and compute objects receive its values.
"""
from __future__ import annotations

import configparser
import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TypeVar

ROOT = Path(__file__).resolve().parents[1]

T = TypeVar('T')


def ini(section: str, key: str, default: T) -> T:
    return field(default=default, metadata=dict(section=section, key=key))


def path(value: str) -> Path:
    """A configured path, relative to the repository root unless absolute."""
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


@dataclass(frozen=True)
class MarketConfig:
    name: str = ini('market', 'name', 'us_equity')


@dataclass(frozen=True)
class ExecutionConfig:
    backend: str = ini('execution', 'backend', 'cuda')


@dataclass(frozen=True)
class UsEquityConfig:
    data_dir: str = ini('us_equity', 'data_dir', './data/us_equity_daily')
    start: str = ini('us_equity', 'start', '2025-07-01')
    end: str = ini('us_equity', 'end', '2026-06-30')
    cost_rate: float = ini('us_equity', 'cost_rate', 0.0005)
    daily_borrow: float = ini('us_equity', 'daily_borrow', 0.0)


@dataclass(frozen=True)
class AShareConfig:
    data_dir: str = ini('a_share', 'data_dir', './data/a_share_pit')
    start: str = ini('a_share', 'start', '2024-03-01')
    end: str = ini('a_share', 'end', '2026-06-30')
    commission: float = ini('a_share', 'commission', 0.00025)
    stamp_tax_sell: float = ini('a_share', 'stamp_tax_sell', 0.0005)
    transfer_fee: float = ini('a_share', 'transfer_fee', 0.00001)


@dataclass(frozen=True)
class CryptoSpotConfig:
    data_dir: str = ini('crypto_spot', 'data_dir', './data/crypto_spot_daily')
    start: str = ini('crypto_spot', 'start', '2024-01-01')
    end: str = ini('crypto_spot', 'end', '2025-12-31')
    cost_rate: float = ini('crypto_spot', 'cost_rate', 0.001)


@dataclass(frozen=True)
class CryptoPerpetualConfig:
    data_dir: str = ini('crypto_perpetual', 'data_dir', './data/crypto_perpetual')
    history_start: str = ini('crypto_perpetual', 'history_start', '2025-01-01')
    start: str = ini('crypto_perpetual', 'start', '2025-07-01')
    end: str = ini('crypto_perpetual', 'end', '2025-12-31')
    symbols: str = ini('crypto_perpetual', 'symbols', 'BTCUSDT ETHUSDT SOLUSDT')
    leverage: float = ini('crypto_perpetual', 'leverage', 5.)
    cost_rate: float = ini('crypto_perpetual', 'cost_rate', .0004)
    maintenance_rate: float = ini('crypto_perpetual', 'maintenance_rate', .005)
    liquidation_fee: float = ini('crypto_perpetual', 'liquidation_fee', .0005)

    def __post_init__(self):
        if not (math.isfinite(self.leverage) and self.leverage >= 1
                and 0 <= self.cost_rate < 1 / self.leverage
                and 0 < self.maintenance_rate < 1 / self.leverage
                and 0 <= self.liquidation_fee < self.maintenance_rate):
            raise ValueError('Invalid isolated-perpetual margin or fee parameters')


@dataclass(frozen=True)
class ExprConfig:
    max_cost: int = ini('expr', 'max_cost', 25)


@dataclass(frozen=True)
class CudaConfig:
    streams: int = ini('cuda', 'streams', 8)
    candidate_batch: int = ini('cuda', 'candidate_batch', 256)
    trial_batch: int = ini('cuda', 'trial_batch', 16384)
    precision: str = ini('cuda', 'precision', 'paired')
    return_atol: float = ini('cuda', 'return_atol', 1e-6)
    perpetual_host_cells: int = ini('cuda', 'perpetual_host_cells', 100000)

    def __post_init__(self):
        if self.streams <= 0 or self.candidate_batch <= 0 or self.trial_batch <= 0:
            raise ValueError('[cuda] stream count and batch sizes must be positive')
        if self.precision not in ('mixed', 'paired'):
            raise ValueError('[cuda] precision must be mixed or paired')
        if not (math.isfinite(self.return_atol) and self.return_atol > 0):
            raise ValueError('[cuda] return error budget must be positive')
        if self.perpetual_host_cells < 0:
            raise ValueError('[cuda] perpetual_host_cells must be nonnegative')


@dataclass(frozen=True)
class PortfolioConfig:
    artifact_dir: str = ini('portfolio', 'artifact_dir', './data/candidates')
    weight_sum_atol: float = ini('portfolio', 'weight_sum_atol', 1e-7)
    metric_atol: float = ini('portfolio', 'metric_atol', 1e-10)
    min_sharpe_gain: float = ini('portfolio', 'min_sharpe_gain', 0.01)
    weight_steps: int = ini('portfolio', 'weight_steps', 20)
    recent_halflife_days: int = ini('portfolio', 'recent_halflife_days', 1095)
    max_members: int = ini('portfolio', 'max_members', 40)
    sharpe_scale: float = ini('portfolio', 'sharpe_scale', 0.1)
    growth_scale: float = ini('portfolio', 'growth_scale', 0.02)
    violation_scale_wsharpe: float = ini('portfolio', 'violation_scale_wsharpe', 0.1)
    violation_scale_sharpe: float = ini('portfolio', 'violation_scale_sharpe', 0.1)
    violation_scale_cagr: float = ini('portfolio', 'violation_scale_cagr', 0.02)
    violation_scale_growth: float = ini('portfolio', 'violation_scale_growth', 0.02)

    def __post_init__(self):
        if not all(math.isfinite(x) and x > 0 for x in (
                self.metric_atol, self.weight_sum_atol, self.sharpe_scale, self.growth_scale, self.violation_scale_wsharpe, self.violation_scale_sharpe,
                self.violation_scale_cagr, self.violation_scale_growth)):
            raise ValueError('[portfolio] utility and violation scales must be positive and finite')


@dataclass(frozen=True)
class SearchConfig:
    mode: str = ini('search', 'mode', 'gfn')
    seed_portfolio: str = ini('search', 'seed_portfolio', 'equal')
    objective: str = ini('search', 'objective', 'portfolio')
    null_shift: bool = ini('search', 'null_shift', False)
    accept: bool = ini('search', 'accept', False)
    seed: int = ini('search', 'seed', 25)
    epochs: int = ini('search', 'epochs', 20)
    batch_size: int = ini('search', 'batch_size', 128)
    probes: int = ini('search', 'probes', 2)
    max_tokens: int = ini('search', 'max_tokens', 13)
    max_ts_depth: int = ini('search', 'max_ts_depth', 2)
    train_steps: int = ini('search', 'train_steps', 4)
    learning_rate: float = ini('search', 'learning_rate', .002)
    opd_strength: float = ini('search', 'opd_strength', .1)
    rho_max: float = ini('search', 'rho_max', .9)
    archive_size: int = ini('search', 'archive_size', 32)
    reweight_iterations: int = ini('search', 'reweight_iterations', 3)
    reweight_batch: int = ini('search', 'reweight_batch', 32)
    reweight_every: int = ini('search', 'reweight_every', 5)

    def __post_init__(self):
        if not (self.mode in ('gfn','random') and self.seed_portfolio in ('current','equal')
                and self.objective in ('portfolio','single')
                and 1 <= self.batch_size <= 8192 and 0 <= self.probes <= self.batch_size
                and self.epochs > 0 and 3 <= self.max_tokens <= 64 and 0 <= self.max_ts_depth <= 64
                and 1 <= self.train_steps <= 100 and 0 <= self.archive_size <= 128
                and 0 <= self.reweight_iterations <= 100 and 2 <= self.reweight_batch <= 8192 and self.reweight_every > 0
                and self.seed >= 0 and 0 < self.learning_rate <= .1
                and 0 <= self.opd_strength <= 10 and 0 < self.rho_max <= 1):
            raise ValueError('Invalid [search] configuration')


@dataclass(frozen=True)
class SingleConfig:
    calibration_sample: int = ini('single', 'calibration_sample', 8192)
    anchor_correlation: float = ini('single', 'anchor_correlation', .7)
    anchor_sharpe: float = ini('single', 'anchor_sharpe', 1.5)
    recent_years: int = ini('single', 'recent_years', 2)
    recent_year_sharpe: float = ini('single', 'recent_year_sharpe', 1.)
    weight_sharpe: float = ini('single', 'weight_sharpe', .6)
    weight_drawdown: float = ini('single', 'weight_drawdown', .2)
    weight_return: float = ini('single', 'weight_return', .2)
    sharpe_scale: float = ini('single', 'sharpe_scale', 2.)
    drawdown_full: float = ini('single', 'drawdown_full', .6)
    return_scale: float = ini('single', 'return_scale', 1.)
    violation_scale_sharpe: float = ini('single', 'violation_scale_sharpe', 1.)
    violation_scale_year: float = ini('single', 'violation_scale_year', 1.)

    def __post_init__(self):
        weights = (self.weight_sharpe, self.weight_drawdown, self.weight_return)
        if not (64 <= self.calibration_sample <= 65536 and 0 <= self.anchor_correlation < 1 and self.anchor_sharpe > 0
                and self.recent_years >= 0 and min(weights) >= 0
                and abs(sum(weights) - 1) < 1e-9 and self.weight_sharpe >= max(weights)
                and min(self.sharpe_scale, self.drawdown_full, self.return_scale,
                        self.violation_scale_sharpe, self.violation_scale_year) > 0):
            raise ValueError('Invalid [single] configuration')


@dataclass(frozen=True)
class MonitorConfig:
    frame_seconds: float = ini('monitor', 'frame_seconds', .1)
    push_seconds: float = ini('monitor', 'push_seconds', 2.)


@dataclass(frozen=True)
class Settings:
    market: MarketConfig
    execution: ExecutionConfig
    a_share: AShareConfig
    us_equity: UsEquityConfig
    crypto_spot: CryptoSpotConfig
    crypto_perpetual: CryptoPerpetualConfig
    expr: ExprConfig
    cuda: CudaConfig
    portfolio: PortfolioConfig
    search: SearchConfig
    single: SingleConfig
    monitor: MonitorConfig


def load_config() -> Settings:
    """Read the one configuration file once, returning an immutable snapshot."""
    parser = configparser.ConfigParser(inline_comment_prefixes=(';', '#'))
    with (ROOT / 'config' / 'config.ini').open(encoding='utf-8') as source:
        parser.read_file(source)
    types = dict(market=MarketConfig, execution=ExecutionConfig, a_share=AShareConfig,
                 us_equity=UsEquityConfig, crypto_spot=CryptoSpotConfig, crypto_perpetual=CryptoPerpetualConfig,
                 expr=ExprConfig, cuda=CudaConfig,
                 portfolio=PortfolioConfig, search=SearchConfig, single=SingleConfig,
                 monitor=MonitorConfig)
    sections = {}
    for name, kind in types.items():
        values = {}
        for definition in fields(kind):
            section, key = definition.metadata['section'], definition.metadata['key']
            value_type = type(definition.default)
            values[definition.name] = (parser.getboolean(section, key) if value_type is bool
                                       else value_type(parser[section][key]))
        sections[name] = kind(**values)
    return Settings(**sections)
