"""
组合级 backtest 配置与一次性面板预处理(市场 config + C 内核 Python 包装)。

    - BacktestRewardConfig + run_bt:Q25 crypto_daily(Quantiacs qnt 1:1 内核)
    - EquityCostsConfig + run_equity_bt / AShareCostsConfig + run_ashare_bt:权益日线
    - prepare_qnt_static:qnt 回测权重无关静态量(ATR14 滑点 / ffill 价 / unlocked)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from backtest import _bt as _bt_kernel
from config.config import ini
from markets.profile import get_profile


class _MarketCostConfig:
    """市场成本配置基类:每个市场的成本/口径 config 绑定一个 MarketProfile(子类设 profile_name),
    warmup / 年化全由该 profile 单源派生 —— 取代旧的「crypto 读全局 ACTIVE、权益各自 get_profile」
    双通道 + 各市场重复定义的 property。新增市场 = 子类 + profile_name,不再重复 warmup/年化。"""
    profile_name: str = ''                              # 子类覆盖为具体画像名

    @property
    def skip_warmup_bars(self) -> int:
        return get_profile(self.profile_name).warmup_bars

    @property
    def bars_per_year(self) -> float:
        return get_profile(self.profile_name).calendar.bars_per_year


@dataclass
class BacktestRewardConfig(_MarketCostConfig):
    """crypto 回测配置。2026-07-04 起 crypto 回测 = Quantiacs qnt 1:1(日线,ATR14×0.04 滑点为
    唯一成本,open 建仓 / close 盯市 / arithmetic 年化 ppy=365,内核 backtest/engine/qnt_bt.cpp)。
    旧永续 5m PerpPolicy 路径(binding + 引擎 + 配套预处理)2026-07-25 已彻底删除。"""
    profile_name = 'crypto_daily'                      # bars_per_year=365 由 profile 单源派生
    # top_k 2026-07-05 已删:reward 书换部署同款 softmax 截面(top-5 从未被证据支持),
    # top-K 机制全线退役(evaluator/_topk_net_np/_gm.topk_net/topk_long_weights 同日删)。


@dataclass
class EquityCostsConfig(_MarketCostConfig):
    """US 权益回测口径(EquityPolicy:佣金+半点差单边对称 cost_rate、空腿日借券费;
    无 funding / 强平 / sqrt-impact)。MTM 与 rebalance 全走**复权 OHLC**(单一总收益口径,
    与 EquityPanel.fwd_ret 一致;不混原始价以免 split/分红跳空双计数)。年化走 us_equity_daily=252。"""
    profile_name = 'us_equity_daily'
    initial_cash:      float = 100_000.0
    cost_rate:         float = ini('equity_costs', 'cost_rate',    0.0005)   # 佣金+半点差,单边 fraction
    daily_borrow:      float = ini('equity_costs', 'daily_borrow', 0.0)      # 空腿日借券费(年化/252);0=关


def run_equity_bt(weights_TS: np.ndarray,
                  adj_close_TS: np.ndarray, adj_open_TS: np.ndarray,
                  ecfg: EquityCostsConfig,
                  raw_weights: bool = True,
                  with_equity: bool = False) -> Dict[str, float]:
    """US 权益组合回测(EquityPolicy)→ 字典 (total_return / sharpe / mean_turnover [/ equity])。
    weights_TS:(T,S) 最终目标权重(market-neutral long-short,Σw≈0;raw_weights=True 直接用)。
    adj_close/adj_open:(T,S) **复权** OHLC(总收益口径)——MTM 用 adj_close,rebalance @ adj_open。
    成本 = cost_rate×|Δnotional|(买卖对称);空腿借券 = daily_borrow×|notional|;无 funding/强平。
    Sharpe 年化 = √bars_per_year(日线 US = √252)。"""
    res = _bt_kernel.portfolio_bt_equity(
        np.ascontiguousarray(weights_TS, dtype=np.float64),
        np.ascontiguousarray(adj_close_TS, dtype=np.float64),
        np.ascontiguousarray(adj_open_TS, dtype=np.float64),
        ecfg.initial_cash, int(ecfg.skip_warmup_bars),
        ecfg.cost_rate, ecfg.daily_borrow,
        float(ecfg.bars_per_year),         # 必传:profile 单源年化(252)
        1 if raw_weights else 0,
    )
    out = {
        'total_return':  float(res['total_return']),
        'sharpe':        float(res['sharpe']),
        'mean_turnover': float(res['mean_turnover']),
    }
    if with_equity:
        out['equity'] = np.asarray(res['equity'], dtype=np.float64)
    return out


@dataclass
class AShareCostsConfig(_MarketCostConfig):
    """中国 A 股回测口径(ASharePolicy:佣金双边 + 印花税卖出单边 + 过户费双边;**long-only**,无 funding/
    强平/impact)。MTM/rebalance 全走 qfq 复权 OHLC(总收益)。年化 √242;daily cadence 天然满足 T+1。"""
    profile_name = 'a_share_daily'
    initial_cash:      float = 100_000.0
    commission:        float = ini('ashare_costs', 'commission',       0.00025)  # 佣金单边(万2.5)
    stamp_tax_sell:    float = ini('ashare_costs', 'stamp_tax_sell',   0.0005)   # 印花税卖出单边(万5,2023 减半)
    transfer_fee:      float = ini('ashare_costs', 'transfer_fee',     0.00001)  # 过户费双边(万0.1)
    price_limit_pct:   float = ini('ashare_costs', 'price_limit_pct',  0.10)     # 涨跌停幅度(沪深300 主板 ±10%)


def ashare_trade_block(adj_close_TS: np.ndarray, adj_open_TS: np.ndarray,
                       raw_volume_TS: np.ndarray, price_limit_pct: float) -> np.ndarray:
    """涨跌停 / 停牌方向冻结掩码 (T,S) int8 —— 喂 run_ashare_bt 的 C 引擎(因果,只用 ≤t 信息)。
    位语义:bit0(=1)开盘即涨停→禁买、bit1(=2)开盘即跌停→禁卖、3(=1|2)停牌→双禁。
    判据(open-fill 口径):ratio=adj_open[t]/adj_close[t-1](复权抹除除权跳空 → 真涨跌幅);
      涨停 ratio≥1+pct−1e-3、跌停 ratio≤1−pct+1e-3(1e-3 容 0.01 元 tick 取整误差);
      停牌 = open 非有限 ∨ 当日 raw 成交量≤0。**用原始(含 NaN)面板**派生,非 ffill 后的。"""
    T, S = adj_close_TS.shape
    block = np.zeros((T, S), dtype=np.int8)
    ratio = np.full((T, S), np.nan)
    ratio[1:] = adj_open_TS[1:] / adj_close_TS[:-1]
    block[ratio >= (1.0 + price_limit_pct) - 1e-3] |= 1           # 开盘涨停 → 禁买
    block[ratio <= (1.0 - price_limit_pct) + 1e-3] |= 2           # 开盘跌停 → 禁卖
    susp = ~np.isfinite(adj_open_TS) | ~np.isfinite(raw_volume_TS) | (raw_volume_TS <= 0.0)
    block[susp] |= 3                                              # 停牌 → 双禁
    return np.ascontiguousarray(block)


def run_ashare_bt(weights_TS: np.ndarray,
                  adj_close_TS: np.ndarray, adj_open_TS: np.ndarray,
                  acfg: AShareCostsConfig,
                  raw_weights: bool = True,
                  trade_block: np.ndarray | None = None,
                  with_equity: bool = False) -> Dict[str, float]:
    """中国 A 股组合回测(ASharePolicy)→ 字典 (total_return / sharpe / mean_turnover [/ equity])。
    weights_TS:(T,S) 最终目标权重(**long-only**,w≥0;A 股不可做空)。adj_*:qfq 复权 OHLC。
    成本 = 佣金×|Δ|(双边)+ **印花税×卖出 notional(单边)** + 过户费×|Δ|;无 funding/强平。Sharpe √242。
    trade_block((T,S) int8,可选):涨跌停 / 停牌方向冻结掩码(见 ashare_trade_block);受限方向不成交。"""
    res = _bt_kernel.portfolio_bt_ashare(
        np.ascontiguousarray(weights_TS, dtype=np.float64),
        np.ascontiguousarray(adj_close_TS, dtype=np.float64),
        np.ascontiguousarray(adj_open_TS, dtype=np.float64),
        acfg.initial_cash, int(acfg.skip_warmup_bars),
        acfg.commission, acfg.stamp_tax_sell, acfg.transfer_fee,
        float(acfg.bars_per_year),         # 必传:profile 单源年化(242)
        1 if raw_weights else 0,
        None if trade_block is None else np.ascontiguousarray(trade_block, dtype=np.int8),
    )
    out = {
        'total_return':  float(res['total_return']),
        'sharpe':        float(res['sharpe']),
        'mean_turnover': float(res['mean_turnover']),
    }
    if with_equity:
        out['equity'] = np.asarray(res['equity'], dtype=np.float64)
    return out


# qnt 平台滑点常量(SL_CRYPTODAILY=0.04):动它即失去与 Quantiacs 跑分的 1:1 → 不进 INI。
QNT_SLIPPAGE_FRACT = 0.04


def prepare_qnt_static(open_raw: np.ndarray, high_raw: np.ndarray,
                       low_raw: np.ndarray, close_raw: np.ndarray,
                       fract: float = QNT_SLIPPAGE_FRACT) -> Dict[str, np.ndarray]:
    """qnt 回测的权重无关静态量(每数据集只算一次,喂 portfolio_bt_qnt):
    ATR14×fract 滑点 / ffill 价 / prev_close / unlocked 掩码。
    ffill 语义 = qnt(前导 NaN **不** bfill;ffill 后残留 NaN → 0)。
    全部输出 (T,S) C-连续 f64(unlocked uint8)。"""
    from numpy.lib.stride_tricks import sliding_window_view
    open_raw, high_raw, low_raw, close_raw = (
        np.ascontiguousarray(x, dtype=np.float64) for x in (open_raw, high_raw, low_raw, close_raw))
    T, S = open_raw.shape

    def ffill_qnt(x: np.ndarray) -> np.ndarray:
        idx = np.where(np.isfinite(x), np.arange(T, dtype=np.int64)[:, None], 0)
        np.maximum.accumulate(idx, axis=0, out=idx)
        return x[idx, np.arange(S, dtype=np.int64)[None, :]]   # 前导取 x[0](NaN)→ 保 NaN

    cl_prev = np.vstack([np.full((1, S), np.nan), close_raw[:-1]])
    d1 = high_raw - low_raw
    d2 = np.abs(high_raw - cl_prev)
    d3 = np.abs(cl_prev - low_raw)
    tr = np.where(np.isnan(d1) | np.isnan(d2) | np.isnan(d3), np.nan,
                  np.fmax(d1, np.fmax(d2, d3)))                # xr.concat.max(skipna=False)
    p = min(T, 14)
    atr = np.full((T, S), np.nan)
    atr[p - 1:] = sliding_window_view(tr, p, axis=0).mean(axis=2)  # rolling(p, min_periods=p, skipna=False)
    slippage = ffill_qnt(atr) * fract

    open_ff = ffill_qnt(open_raw)
    open_ff0 = np.where(np.isnan(open_ff), 0.0, open_ff)
    close_ff = ffill_qnt(close_raw)
    close_ff0 = np.where(np.isnan(close_ff), 0.0, close_ff)
    prev_close = ffill_qnt(cl_prev)                            # close.shift(1).ffill(前导保 NaN)
    unlocked = (np.isfinite(open_raw) & np.isfinite(close_raw)
                & np.isfinite(slippage) & (open_ff0 > 1e-7))   # qnt EPS;不含 is_liquid(qnt 引擎层语义)
    c = np.ascontiguousarray
    return dict(open_raw=c(open_raw), prev_close=c(prev_close),
                open_ff0=c(open_ff0), close_ff0=c(close_ff0),
                slippage=c(slippage), unlocked=c(unlocked.astype(np.uint8)))


def run_bt(weights_TS: np.ndarray,
           qnt_static: Dict[str, np.ndarray],
           stat_start: int,
           bcfg: BacktestRewardConfig,
           with_equity: bool = False) -> Dict[str, float]:
    """crypto 组合回测 = Quantiacs qnt 1:1(_bt_kernel.portfolio_bt_qnt)。
    weights_TS:(T,S) 策略**原始**权重(qnt 语义:内核自做 shift-1 / 双重归一 / open 价
    调整,勿外部预 shift)。qnt_static:prepare_qnt_static 产物。stat_start:统计窗起点
    行号(竞赛 IS 口径 = 2016-01-01 行;此前权重须为 0,见 qnt_bt.hpp)。
    → (total_return / sharpe / mean_return / volatility / max_drawdown);
    with_equity=True 附 'equity' + 'rr' (T,) 序列。CUDA 批量镜像:ops_cuda.metrics.qnt_bt。"""
    res = _bt_kernel.portfolio_bt_qnt(
        np.ascontiguousarray(weights_TS, dtype=np.float64),
        qnt_static['open_raw'], qnt_static['prev_close'],
        qnt_static['open_ff0'], qnt_static['close_ff0'],
        qnt_static['slippage'], qnt_static['unlocked'],
        int(stat_start), float(bcfg.bars_per_year))
    out = {k: float(res[k])
           for k in ('total_return', 'sharpe', 'mean_return', 'volatility', 'max_drawdown')}
    if with_equity:
        out['equity'] = np.asarray(res['equity'], dtype=np.float64)
        out['rr'] = np.asarray(res['rr'], dtype=np.float64)
    return out
