"""配置 — 单一入口:读 `config/config.ini`。

只暴露一个助手 `ini(section, key, default)`:类型按 default 自动推断,INI 缺失或留空
时回落 default。子模块在 dataclass 字段上直接调用,例:

    @dataclass
    class FactorMiningConfig:
        pool_capacity: int = ini('factor_mining', 'pool_capacity', 12)

INI 在 import 阶段读一次,值固化为字段默认。
"""

from __future__ import annotations

import configparser
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar


_INI = configparser.ConfigParser(inline_comment_prefixes=(';', '#'))
_INI.read(Path(__file__).resolve().parent / 'config.ini', encoding='utf-8')


T = TypeVar('T')


def ini(section: str, key: str, default: T) -> T:
    """从 config.ini[section][key] 读值;类型按 default 自动转换。
    缺失 / 留空 → 返回 default。"""
    if section not in _INI or key not in _INI[section]:
        return default
    raw = _INI[section][key].strip()
    if not raw:
        return default
    t = type(default)
    if t is bool:
        return raw.lower() in ('true', 'yes', '1', 'on')   # type: ignore[return-value]
    if t is int:
        return int(raw)                                    # type: ignore[return-value]
    if t is float:
        return float(raw)                                  # type: ignore[return-value]
    return raw                                             # type: ignore[return-value]


# ============================================================================
# 顶层 orchestration 配置
# ============================================================================

@dataclass
class FactorMiningConfig:
    qnt_data_dir:    str = ini('factor_mining', 'qnt_data_dir', './data/qnt_crypto')
    data_start:      str = ini('factor_mining', 'data_start', '2015-01-01')
    pool_capacity:   int = ini('factor_mining', 'pool_capacity', 12)
    # 挖矿窗口(gp/alphasage 共用;Q25 crypto_daily 口径):test = 竞赛 live,天然 held out
    train_start:     str = ini('factor_mining', 'train_start', '2016-01-01')
    train_end:       str = ini('factor_mining', 'train_end', '2023-06-30')
    val_start:       str = ini('factor_mining', 'val_start', '2023-07-01')
    val_end:         str = ini('factor_mining', 'val_end', '2026-09-01')


@dataclass
class FeaturesConfig:
    """OHLCV 提取分解特征层(scripts/build_q25_dataset.py → features.parquet;
    2026-07-05:池审计 N_eff=1.2 修复 —— scale-free 原语 + 因果滚动 PLS 正交监督复合)。"""
    pls_components:   int   = ini('features', 'pls_components', 4)
    refit_days:       int   = ini('features', 'refit_days', 30)
    pls_warmup:       int   = ini('features', 'pls_warmup', 365)
    beta_ewma_lambda: float = ini('features', 'beta_ewma_lambda', 0.97)


@dataclass
class Q25ComboConfig:
    min_sharpe_gain: float = ini('q25_combo', 'min_sharpe_gain', 0.01)
    weight_steps: int = ini('q25_combo', 'weight_steps', 20)
    recent_halflife_days: int = ini('q25_combo', 'recent_halflife_days', 1095)
    submission_sharpe: float = ini('q25_combo', 'submission_sharpe', 1.58)
    max_members: int = ini('q25_combo', 'max_members', 40)
    sharpe_scale: float = ini('q25_combo', 'sharpe_scale', 0.1)
    growth_scale: float = ini('q25_combo', 'growth_scale', 0.02)
    violation_scale_wsharpe: float = ini('q25_combo', 'violation_scale_wsharpe', 0.1)
    violation_scale_sharpe: float = ini('q25_combo', 'violation_scale_sharpe', 0.1)
    violation_scale_cagr: float = ini('q25_combo', 'violation_scale_cagr', 0.02)
    violation_scale_growth: float = ini('q25_combo', 'violation_scale_growth', 0.02)

    def __post_init__(self):
        if not all(math.isfinite(x) and x > 0 for x in (
                self.sharpe_scale, self.growth_scale, self.violation_scale_wsharpe, self.violation_scale_sharpe,
                self.violation_scale_cagr, self.violation_scale_growth)):
            raise ValueError('Portfolio utility and violation scales must be positive and finite')


@dataclass
class Q25CudaConfig:
    candidate_batch: int = ini('q25_cuda', 'candidate_batch', 256)
    trial_batch: int = ini('q25_cuda', 'trial_batch', 16384)
    precision: str = ini('q25_cuda', 'precision', 'auto')
    mixed_return_atol: float = ini('q25_cuda', 'mixed_return_atol', 1e-6)
    mixed_error_safety: float = ini('q25_cuda', 'mixed_error_safety', 16.)

    def __post_init__(self):
        if self.candidate_batch <= 0 or self.trial_batch <= 0:
            raise ValueError('Q25 CUDA batch sizes must be positive')
        if self.precision not in ('auto','f64'):
            raise ValueError('Production Q25 precision must be auto or f64')
        if (not math.isfinite(self.mixed_return_atol) or not math.isfinite(self.mixed_error_safety)
                or self.mixed_return_atol <= 0 or self.mixed_error_safety < 1):
            raise ValueError('Q25 precision error budgets must be positive with safety >= 1')


@dataclass
class Q25SingleConfig:
    calibration_sample: int = ini('q25_single', 'calibration_sample', 8192)
    anchor_correlation: float = ini('q25_single', 'anchor_correlation', .7)
    anchor_sharpe: float = ini('q25_single', 'anchor_sharpe', 1.5)
    recent_years: int = ini('q25_single', 'recent_years', 2)
    recent_year_sharpe: float = ini('q25_single', 'recent_year_sharpe', 1.)
    weight_sharpe: float = ini('q25_single', 'weight_sharpe', .6)
    weight_drawdown: float = ini('q25_single', 'weight_drawdown', .2)
    weight_return: float = ini('q25_single', 'weight_return', .2)
    sharpe_scale: float = ini('q25_single', 'sharpe_scale', 2.)
    drawdown_full: float = ini('q25_single', 'drawdown_full', .6)
    return_scale: float = ini('q25_single', 'return_scale', 1.)
    violation_scale_sharpe: float = ini('q25_single', 'violation_scale_sharpe', 1.)
    violation_scale_year: float = ini('q25_single', 'violation_scale_year', 1.)

    def __post_init__(self):
        weights = (self.weight_sharpe, self.weight_drawdown, self.weight_return)
        if not (64 <= self.calibration_sample <= 65536 and 0 <= self.anchor_correlation < 1 and self.anchor_sharpe > 0
                and self.recent_years >= 0 and min(weights) >= 0
                and abs(sum(weights)-1) < 1e-9 and self.weight_sharpe >= max(weights)
                and min(self.sharpe_scale, self.drawdown_full, self.return_scale,
                        self.violation_scale_sharpe, self.violation_scale_year) > 0):
            raise ValueError('Invalid Q25 single-alpha configuration')


@dataclass
class Q25MonitorConfig:
    frame_seconds: float = ini('q25_monitor', 'frame_seconds', .1)
    push_seconds: float = ini('q25_monitor', 'push_seconds', 2.)


@dataclass
class Q25SearchConfig:
    seed: int = ini('q25_search', 'seed', 25)
    epochs: int = ini('q25_search', 'epochs', 20)
    batch_size: int = ini('q25_search', 'batch_size', 128)
    probes: int = ini('q25_search', 'probes', 2)
    max_tokens: int = ini('q25_search', 'max_tokens', 13)
    max_cost: int = ini('q25_search', 'max_cost', 25)
    max_ts_depth: int = ini('q25_search', 'max_ts_depth', 2)
    train_steps: int = ini('q25_search', 'train_steps', 4)
    learning_rate: float = ini('q25_search', 'learning_rate', .002)
    opd_strength: float = ini('q25_search', 'opd_strength', .1)
    rho_max: float = ini('q25_search', 'rho_max', .9)
    archive_size: int = ini('q25_search', 'archive_size', 32)
    reweight_iterations: int = ini('q25_search', 'reweight_iterations', 3)
    reweight_batch: int = ini('q25_search', 'reweight_batch', 32)
    reweight_every: int = ini('q25_search', 'reweight_every', 5)

    def __post_init__(self):
        if not (1 <= self.batch_size <= 8192 and 0 <= self.probes <= self.batch_size
                and self.epochs > 0 and 3 <= self.max_tokens <= 64 and 3 <= self.max_cost <= 25
                and 0 <= self.max_ts_depth <= 64
                and 1 <= self.train_steps <= 100 and 0 <= self.archive_size <= 128
                and 0 <= self.reweight_iterations <= 100 and 2 <= self.reweight_batch <= 8192 and self.reweight_every > 0
                and self.seed >= 0 and 0 < self.learning_rate <= .1
                and 0 <= self.opd_strength <= 10 and 0 < self.rho_max <= 1):
            raise ValueError('Invalid Q25 native search configuration')


@dataclass
class TsBookConfig:
    """时序 long-only 书(crypto_daily 挖矿口径):2026-07-24 自由书转向 —— 搜索/评估 =
    公式直出敞口(rl.sizing.qnt_free_exposure),reward = 书净 Sharpe − EW 锚不变;
    state_window 退位为部署层原语参数(现役 ts_arg_min 提交栈复现)。"""
    state_window: int   = ini('ts_book', 'state_window', 120)
    gross_scale:  float = ini('ts_book', 'gross_scale', 1.0)


@dataclass
class AlphaSAGEConfig:
    """AlphaSAGE GFlowNet alpha-mining baseline(main.py train-alphasage 驱动,独立于 GP)。
    数据路径/pool_capacity 复用 [factor_mining];GFN/reward 超参在 [alphasage]。"""
    pool_capacity:       int   = ini('factor_mining', 'pool_capacity', 12)
    hidden_dim:          int   = ini('alphasage', 'hidden_dim', 64)
    n_layers:            int   = ini('alphasage', 'n_layers', 2)
    lr:                  float = ini('alphasage', 'lr', 0.001)
    logz_lr:             float = ini('alphasage', 'logz_lr', 0.1)       # logZ 单独高 lr(GFlowNet 铁律,防 TB 发散)
    batch_size:          int   = ini('alphasage', 'batch_size', 16)
    n_episodes:          int   = ini('alphasage', 'n_episodes', 1000)
    nov_weight:          float = ini('alphasage', 'nov_weight', 0.1)
    entropy_coef:        float = ini('alphasage', 'entropy_coef', 0.01)
    entropy_temperature: float = ini('alphasage', 'entropy_temperature', 1.0)
    r_sa_weight:         float = ini('alphasage', 'r_sa_weight', 0.5)     # λ(0):R_SA 结构稠密奖励初值(退火→0)
    r_sa_k:              int   = ini('alphasage', 'r_sa_k', 16)           # R_SA kNN 结构邻居数(池内)
    r_sa_tau:            float = ini('alphasage', 'r_sa_tau', 1.0)        # R_SA 结构相似度 softmax 温度
