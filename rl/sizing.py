"""确定性 sizing — Q25 时序 long-only 书(2026-07-22 挖矿口径转向:截面选币 → 逐币时序状态)。

动机(记忆 project-q25-mining-stat-ceiling + 2026-07-22 文献深检):S≈10 截面 rank 信号统计
不可辨识(日 IC 噪声 sd≈0.34,train 最优 IR=纯噪声极值);大市值 crypto 净成本存活的结构
几乎全是时序性(TSMOM/多窗趋势,Han-Kang-Ryu (28,5) 净 Sharpe 1.51 全市值分位存活;
Zarattini SFI 25-80 long-only top-20 净 Sharpe 1.57)。书结构:
  state = ts_rank(factor, state_window)        逐币因果滚动分位 ∈[0,1](对逐币单调变换不变
                                               → mul(正市场序列, x) 之类包装等价类自动坍缩)
  gate  = max(2·state − 1, 0)                  高于自身中位数才持仓,线性信念
  W     = gate / n_liq                         Σ|W| = liquid 内 gate 均值 ≤ 1:
                                               信念不足自动留现金(qnt 内核 Σw<1 余额=现金),
                                               全体满信念 = 满仓 EW
截面选币退位为副产品:收益来源 = 每币对自身历史的趋势状态 + 组合仓位择时。

2026-07-24 自由书口径(用户指令:扫除搜索器自由度限制):搜索/评估侧不再强制
rank-gate 投影,公式输出**直出敞口** qnt_free_exposure(clip[0,1]),配 GT/LT 恢复后
searcher 可显式表达 regime 条件门、信号×门乘法双层、1/σ 缩放;旧投影 = 公式里显式写
ts_rank 的特例,自动成为可选而非枷锁。qnt_ts_state/qnt_ts_gate/qnt_ts_weights 保留为
**部署层原语**(现役 ts_arg_min(btc_rv,240) 提交栈与 diag 脚本仍按旧口径复现)。

旧截面栈(signal_to_weight / topk_ls / softmax 书 / β·crowding 中性化 / aff_fuse /
vol-target overlay)2026-07-22 随口径转向全线退役删除(零调用;重启截面研究从 git 历史恢复)。
"""
from __future__ import annotations

import numpy as np

from backtest import ops


def qnt_ts_state(signal: np.ndarray, window: int) -> np.ndarray:
    """逐币因果时序状态:ts_rank(signal, window) ∈ [0,1](当前值在自身近 window 日的分位;
    warmup / 窗内含 NaN → NaN)。C 内核(W≥32 走 bucketized BIT)。"""
    return ops.ts_rank(np.ascontiguousarray(signal, dtype=np.float64), int(window))


def qnt_ts_gate(state: np.ndarray, liquid: np.ndarray) -> np.ndarray:
    """state (T,S) → gate (T,S) ∈[0,1]:max(2·state−1, 0);state NaN / 非 liquid → 0。"""
    return np.where(liquid & np.isfinite(state), np.clip(2.0 * state - 1.0, 0.0, 1.0), 0.0)


def qnt_gate_book(gate: np.ndarray, liquid: np.ndarray, gross_scale: float = 1.0) -> np.ndarray:
    """gate (T,S) → 权重 (T,S) = gross_scale·gate / n_liq(行内 liquid 数,≥1)。池 ensemble =
    成员 gate 等权平均后走同一入口(两级等权定论)。gross_scale=λ 组合层 gross 放大
    ([ts_book].gross_scale;Σ|w|>1 由 qnt 内核顶格归一 = 单纯形投影,保相对配比):
    挖矿 reward 恒 λ=1(Sharpe 对常数杠杆不变 → 排序等价,信号形状不受 λ 污染);
    λ 只在 eval_pool_val / 部署书生效,是用户对 DD 预算的显式决策。"""
    nliq = np.maximum(liquid.sum(1, keepdims=True), 1).astype(np.float64)
    return gross_scale * gate / nliq


def qnt_ts_weights(signal: np.ndarray, liquid: np.ndarray, state_window: int) -> np.ndarray:
    """因子 (T,S) → 时序 long-only 权重 (T,S)(state → gate → book 三段合一)。"""
    return qnt_gate_book(qnt_ts_gate(qnt_ts_state(signal, state_window), liquid), liquid)


def qnt_free_exposure(values: np.ndarray, liquid: np.ndarray) -> np.ndarray:
    """自由书敞口(2026-07-24 搜索口径):公式输出 (T,S) 直接当敞口,clip 到 [0,1];
    非有限(warmup NaN)/ 非 liquid → 0(无仓语义)。归一走同一 qnt_gate_book(/n_liq,
    现金自动)。有界门由 searcher 自己构造(GT/LT / ts_rank / ts_chanpos / tanh…);
    尺度失控的因子会饱和成恒 0/恒 1 书,由 G-EXP 暴露带结构拒。"""
    return np.where(liquid & np.isfinite(values), np.clip(values, 0.0, 1.0), 0.0)


def unlock_downweight(W: np.ndarray, d2unlock: np.ndarray,
                      pre_days: float = 5.0, post_days: float = 3.0) -> np.ndarray:
    """解锁 overhang 事件降权(**部署/交易层**;研究读数一律裸口径,勿叠进 eval)。
    d2unlock (T,S) = 距最近一次已公开 cliff 解锁的带号天数(正=事前,负=事后;NaN=无计划,
    由 scripts/build_unlock_events.py 从公开 vesting 日程构建 → 决策时可知,无 lookahead)。
    [−post_days, +pre_days] 窗内该币权重 → 0(Kim SSRN 6632838:52 事件 88.5% 负 72h;
    Keyrock/Tokenomist 236 事件 ~90% 负 30d,驱动=预期抢跑)。释放的权重不重归一(转现金),
    避免把事件信息泄给其余名的相对配置。"""
    inwin = np.isfinite(d2unlock) & (d2unlock <= pre_days) & (d2unlock >= -post_days)
    return np.where(inwin, 0.0, W)
