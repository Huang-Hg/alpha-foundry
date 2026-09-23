"""Configuration: one INI file (config/config.ini), one dataclass per section.

`ini(section, key, default)` reads a value and casts it to the type of the default; the INI is read once at import,
so a dataclass instance carries the values of the file as it was when the process started.
"""
from __future__ import annotations

import configparser
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

ROOT = Path(__file__).resolve().parents[1]
_INI = configparser.ConfigParser(inline_comment_prefixes=(';', '#'))
_INI.read(ROOT / 'config' / 'config.ini', encoding='utf-8')

T = TypeVar('T')


def ini(section: str, key: str, default: T) -> T:
    raw = _INI[section][key].strip() if section in _INI and key in _INI[section] else ''
    if not raw:
        return default
    kind = type(default)
    if kind is bool:
        return raw.lower() in ('true', 'yes', '1', 'on')
    return kind(raw)


def path(value: str) -> Path:
    """A configured path, relative to the repository root unless absolute."""
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


@dataclass(frozen=True)
class MarketConfig:
    name: str = ini('market', 'name', 'crypto')


@dataclass(frozen=True)
class CryptoConfig:
    data_dir: str = ini('crypto', 'data_dir', './data/qnt_crypto')
    data_start: str = ini('crypto', 'data_start', '2015-01-01')
    start: str = ini('crypto', 'start', '2016-01-01')
    end: str = ini('crypto', 'end', '2026-09-01')
    pls_components: int = ini('crypto', 'pls_components', 4)
    pls_refit_days: int = ini('crypto', 'pls_refit_days', 30)
    pls_warmup: int = ini('crypto', 'pls_warmup', 365)
    beta_ewma_lambda: float = ini('crypto', 'beta_ewma_lambda', 0.97)


@dataclass(frozen=True)
class UsEquityConfig:
    data_dir: str = ini('us_equity', 'data_dir', './data/us_equity_daily')
    start: str = ini('us_equity', 'start', '2025-07-01')
    end: str = ini('us_equity', 'end', '2026-06-30')
    cost_rate: float = ini('us_equity', 'cost_rate', 0.0005)
    daily_borrow: float = ini('us_equity', 'daily_borrow', 0.0)


@dataclass(frozen=True)
class AShareConfig:
    data_dir: str = ini('a_share', 'data_dir', './data/a_share_daily')
    start: str = ini('a_share', 'start', '2024-03-01')
    end: str = ini('a_share', 'end', '2026-06-30')
    commission: float = ini('a_share', 'commission', 0.00025)
    stamp_tax_sell: float = ini('a_share', 'stamp_tax_sell', 0.0005)
    transfer_fee: float = ini('a_share', 'transfer_fee', 0.00001)


@dataclass(frozen=True)
class ExprConfig:
    max_cost: int = ini('expr', 'max_cost', 25)


@dataclass(frozen=True)
class CudaConfig:
    candidate_batch: int = ini('cuda', 'candidate_batch', 256)
    trial_batch: int = ini('cuda', 'trial_batch', 16384)
    precision: str = ini('cuda', 'precision', 'auto')
    mixed_return_atol: float = ini('cuda', 'mixed_return_atol', 1e-6)
    mixed_error_safety: float = ini('cuda', 'mixed_error_safety', 16.)

    def __post_init__(self):
        if self.candidate_batch <= 0 or self.trial_batch <= 0:
            raise ValueError('[cuda] batch sizes must be positive')
        if self.precision not in ('auto', 'f64'):
            raise ValueError('[cuda] precision must be auto or f64')
        if not (math.isfinite(self.mixed_return_atol) and self.mixed_return_atol > 0 and self.mixed_error_safety >= 1):
            raise ValueError('[cuda] error budget must be positive with safety >= 1')


@dataclass(frozen=True)
class PortfolioConfig:
    min_sharpe_gain: float = ini('portfolio', 'min_sharpe_gain', 0.01)
    weight_steps: int = ini('portfolio', 'weight_steps', 20)
    recent_halflife_days: int = ini('portfolio', 'recent_halflife_days', 1095)
    submission_sharpe: float = ini('portfolio', 'submission_sharpe', 1.58)
    max_members: int = ini('portfolio', 'max_members', 40)
    sharpe_scale: float = ini('portfolio', 'sharpe_scale', 0.1)
    growth_scale: float = ini('portfolio', 'growth_scale', 0.02)
    violation_scale_wsharpe: float = ini('portfolio', 'violation_scale_wsharpe', 0.1)
    violation_scale_sharpe: float = ini('portfolio', 'violation_scale_sharpe', 0.1)
    violation_scale_cagr: float = ini('portfolio', 'violation_scale_cagr', 0.02)
    violation_scale_growth: float = ini('portfolio', 'violation_scale_growth', 0.02)

    def __post_init__(self):
        if not all(math.isfinite(x) and x > 0 for x in (
                self.sharpe_scale, self.growth_scale, self.violation_scale_wsharpe, self.violation_scale_sharpe,
                self.violation_scale_cagr, self.violation_scale_growth)):
            raise ValueError('[portfolio] utility and violation scales must be positive and finite')


@dataclass(frozen=True)
class SearchConfig:
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
        if not (1 <= self.batch_size <= 8192 and 0 <= self.probes <= self.batch_size
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
