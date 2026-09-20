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


@dataclass
class Q25CudaConfig:
    candidate_batch: int = ini('q25_cuda', 'candidate_batch', 256)
    trial_batch: int = ini('q25_cuda', 'trial_batch', 16384)

    def __post_init__(self):
        if self.candidate_batch <= 0 or self.trial_batch <= 0:
            raise ValueError('Q25 CUDA batch sizes must be positive')


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
