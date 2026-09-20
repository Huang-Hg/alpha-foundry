"""
Alpha 评估 — crypto_daily **时序 long-only 书**口径(2026-07-22 截面→时序转向)。

转向依据(记忆 project-q25-mining-stat-ceiling + 2026-07-22 文献深检):S≈10 截面 rank 信号
统计不可辨识(日 IC 噪声 sd≈0.34;train 最优 IR=纯噪声极值 1.3-1.6 带内);大市值 crypto
净成本存活结构几乎全是时序性(TSMOM/多窗趋势)。故:
  质量 q = **书净 Sharpe − EW 锚净 Sharpe**(qnt 官方口径,同引擎同统计窗;EW 锚 = 流动名
  逐日等权书,set_qnt_context 注入时算一次)

2026-07-22 冗余机制清除(跨引擎族判决后):4 折时序一致性 fold_pos 删(实证击穿 —— GP 唯一
4/4 折成员 val 最差,train edge 与 val 近似反序,表达式级挡不住选择偏差;真正的鉴别在族级
参数高原/跨引擎复制,不在 train 内切折);rank_ic 参考量 + y_future 标签管线删(截面口径遗留,
零决策);FailCache/G2 嵌入去重删(policy net 06-27 删除后全管线 embed_fn=None,从未生效);
零读者 profiling 删。

门通路(任一拒 → _bad;below-gate → per_t_pnl=None,gp 侧落 _REWARD_FLOOR 不收集):
  G0        无 operand / raw-price-wrapper(无 ts_*/pair 且仅 OHLCV leaf)结构静态拒
  G1        finite_ratio 数值有效性(**截面 std 门退役**:截面常数因子=合法市场择时书)
  G-EXP     书暴露带 [min_exposure, max_exposure]:恒空(死书)/恒满仓(EW 克隆)结构拒

流程(2026-07-24 自由书口径,用户指令扫除搜索器自由度限制):eval_tree(**CPU 单路径**
—— 日线 panel 极小,单候选 C 内核毫秒级)→ qnt_free_exposure(公式输出直出敞口 clip[0,1],
NaN/非 liquid=0;旧 rank-gate 投影 = 公式显式写 ts_rank 的特例)→ W = 敞口/n_liq(信念
不足自动留现金)→ run_bt(qnt 1:1)→ edge / exposure / 书换手 / ex 超额流(供池 R²/Δ/多样性)。

事件融合:事件面板(events.parquet:d2unlock/unlockpct/unlockwin)经 prepare_gp_data 入词表
→ 公式可自由与趋势状态复合;部署/交易层另有确定性 unlock_downweight(rl.sizing)。

非 crypto_daily 市场口径已退役,重启时从 git 历史恢复。
输出 dict:net_sharpe(=edge)/ exposure / turnover(书换手)/ per_t_pnl(= rr_book − rr_ew
超额流,供池 R²/Δ/多样性)。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from config.config import ini
from backtest import ops
from evaluation import grammar as _grammar
from evaluation.cache import EvalCache
from evaluation.ast import AlphaTree, Node
from evaluation.expression import evaluate as eval_tree
from markets.profile import ACTIVE as _PROFILE
from rl.backtest_reward import BacktestRewardConfig, run_bt
from rl.sizing import qnt_free_exposure, qnt_gate_book, qnt_ts_weights

# crypto 回测 = qnt 1:1:active profile 为 crypto_daily 时走时序书 reward。
_QNT_REWARD = _PROFILE.name == 'crypto_daily'
_QNT_CTX: Optional[Dict] = None

# alphasage trainer 的 torch device 派发仍读此值(GFN 网络在 GPU;评估恒 CPU)。
_DEVICE = ini('evaluator', 'device', 'cpu')

_REWARD_FLOOR = -2.0      # gp_baseline 无效个体(门拒 / cost 越界)的 fitness 下界


# R4 滚动折数(2026-07-25 接入):train 7.5y → ~18mo/折 ≈ 半个 crypto 周期,5 折跨
# 2 牛 2 熊 1 冬;固定实现常量非 knob。折 edge 中位数 = reward/池准入统计量(cv_edge),
# 时间轴尖峰(单 regime 走运树)在取出统计量处就不给分;>0 ⇔ ≥3/5 折胜锚。
_CV_FOLDS = 5


def _fold_sharpe(x: np.ndarray) -> float:
    """折内官方算术 Sharpe(ppy=365)。全现金折(std=0)按约定 Sharpe=0。"""
    s = float(x.std())
    return float(x.mean()) / s * 365.0 ** 0.5 if s > 0.0 else 0.0


def set_qnt_context(static: Dict[str, np.ndarray], liquid: np.ndarray, row0: int) -> None:
    """crypto_daily 挖矿奖励上下文(main.prepare_gp_data 注入一次):
    static = prepare_qnt_static 产物(网格 = 数据头..train_end,与 train panels 同长);
    liquid (T_grid,S) bool = is_liquid 因果掩码;row0 = train_start 行号(统计窗起点,
    此前权重恒 0)。附带算一次流动名逐日等权书的 rr 作 EW 锚(rr_bench),
    并预算 R4 滚动折界 + 锚折内 Sharpe(cv_edge 零增量回测成本)。"""
    global _QNT_CTX
    bcfg = BacktestRewardConfig()
    W_ew = np.zeros(liquid.shape, dtype=np.float64)
    L = liquid[row0:].astype(np.float64)
    W_ew[row0:] = L / np.maximum(L.sum(1, keepdims=True), 1.0)
    res = run_bt(W_ew, static, int(row0), bcfg, with_equity=True)
    bounds = np.linspace(row0, liquid.shape[0], _CV_FOLDS + 1).astype(int)
    _QNT_CTX = dict(static=static, liquid=liquid, row0=int(row0), bcfg=bcfg,
                    rr_bench=res['rr'], bench_sharpe=float(res['sharpe']),
                    cv_bounds=bounds,
                    bench_fold_sharpes=tuple(_fold_sharpe(res['rr'][a:b])
                                             for a, b in zip(bounds[:-1], bounds[1:])))


def qnt_bench_sharpe() -> float:
    """EW 锚 Sharpe(set_qnt_context 注入后有效)。Sharpe 一律 = engine run_bt 的 qnt 竞赛
    官方口径(calc_stat 1:1),evaluator 不自算;日志读数 = 内部 edge + 锚。"""
    return _QNT_CTX['bench_sharpe']


def _qnt_ts_net(values: np.ndarray):
    """时序书 reward:公式输出 (T_grid,S) → qnt_free_exposure(clip[0,1])→ W(敞口/n_liq,
    现金自动)→ run_bt → (edge, cv_edge, exposure, book_turnover, ex)。
    edge = 书 Sharpe − EW 锚 Sharpe(均为 engine qnt 官方口径,同统计窗);cv_edge = R4
    滚动折 edge 中位数(同一 rr 切 _CV_FOLDS 折,零增量回测);exposure = 平均
    Σ|W|(暴露带门用);ex = rr_book − rr_ew 全网格超额流(池正交度量)。"""
    ctx = _QNT_CTX
    row0 = ctx['row0']
    E = qnt_free_exposure(values, ctx['liquid'])
    W = qnt_gate_book(E, ctx['liquid'])
    W[:row0] = 0.0
    res = run_bt(np.ascontiguousarray(W), ctx['static'], row0, ctx['bcfg'],
                 with_equity=True)
    edge = float(res['sharpe']) - ctx['bench_sharpe']
    bounds = ctx['cv_bounds']
    cv_edge = float(np.median([_fold_sharpe(res['rr'][a:b]) - bs
                               for a, b, bs in zip(bounds[:-1], bounds[1:],
                                                   ctx['bench_fold_sharpes'])]))
    exposure = float(np.abs(W[row0:]).sum(1).mean())
    book_to = float(np.abs(np.diff(W[row0:], axis=0)).sum(1).mean())
    ex = np.ascontiguousarray(res['rr'] - ctx['rr_bench'], dtype=np.float64)
    return edge, cv_edge, exposure, book_to, ex


@dataclass
class EvalConfig:
    # 数值有效性 / 书暴露带(固定实现常量)
    min_finite_ratio: float = 0.10
    min_exposure:     float = 0.02    # 平均 Σ|W| 下限:恒空 = 死书
    max_exposure:     float = 0.98    # 上限:恒满仓 = EW 克隆(edge≈0 纯占坑)


def evaluate_alpha(
    tree: AlphaTree,
    panels: Dict[str, np.ndarray],
    cfg: EvalConfig = EvalConfig(),
    cache: Optional[EvalCache] = None,
) -> Dict[str, float]:
    """时序书因子评估 path。

    返回 dict 字段:
        net_sharpe — 全窗 edge = 书 Sharpe − EW 锚 Sharpe(日志/电池 M3 沿用口径)
        cv_edge    — R4 滚动折 edge 中位数(2026-07-25 起 GFN reward / GP obj1 /
                     池准入质量幅度统一换此刀;>0 ⇔ ≥3/5 折胜锚)
        exposure   — 平均 Σ|W|(书在场比例)
        turnover   — 书日均换手 Σ|ΔW|(成本直觉参考)
        per_t_pnl  — (T_grid,) rr_book − rr_ew 超额流(above-gate 才暴露;供池 R²/Δ/多样性)
    """
    if not _QNT_REWARD:
        raise NotImplementedError('评估仅存 crypto_daily 时序书口径;权益系重启时从 git 历史恢复')
    if not tree.has_operand():
        return _bad()
    # G0: raw-price-wrapper 结构静态拒(无 ts_*/pair 且仅 OHLCV leaf)
    if tree.is_raw_price_wrapper():
        return _bad()

    values = eval_tree(tree, panels, cache)
    values_c = np.ascontiguousarray(values, dtype=np.float64)

    # G1: finite_ratio 数值有效性(截面 std 门退役 —— 截面常数 = 合法市场择时书)
    finite_ratio, _ = ops.cs_finite_validstd(values_c, 1e-9)
    if finite_ratio < cfg.min_finite_ratio:
        return _bad()

    # 时序书 reward:敞口直出 → W → run_bt(qnt 1:1)→ edge / cv_edge / exposure / 换手 / ex 流
    edge, cv_edge, exposure, book_to, ex = _qnt_ts_net(values_c)

    # G-EXP: 书暴露带 — 恒空(死书)/ 恒满仓(EW 克隆)结构拒
    if not (cfg.min_exposure <= exposure <= cfg.max_exposure):
        return _bad(exposure=exposure)

    return {
        'net_sharpe': float(edge),
        'cv_edge':    float(cv_edge),
        'exposure':   float(exposure),
        'turnover':   float(book_to),
        'per_t_pnl':  ex,
    }


def _bad(**fields) -> Dict[str, float]:
    # 坏结构(G0/G1/G-EXP 拒):per_t_pnl=None → gp 侧 fitness 落 _REWARD_FLOOR、不收集。
    out = {'net_sharpe': 0.0, 'cv_edge': 0.0, 'exposure': 0.0, 'turnover': 0.0,
           'per_t_pnl': None}
    out.update(fields)
    return out


# ============================================================================
# 多矩复合门(2026-07-24 安装):train-only 稳健性电池,val 触碰的前置滤器
# ============================================================================

def _window_neighbor_trees(tree: AlphaTree):
    """M3 参数高原邻域:树中每个 ts/pair 窗替换成窗口网格相邻档(边档单邻),
    一次只动一个窗 → 变体树列表;无窗树 → 空。"""
    grid = sorted(_grammar.ACTIVE_WINDOWS)
    variants = []

    def rebuild(n: Node, path: tuple, target: tuple, new_w: int) -> Node:
        children = tuple(rebuild(c, path + (i,), target, new_w)
                         for i, c in enumerate(n.children))
        w = new_w if path == target else n.window
        return Node.make(n.kind, n.op, w, children)

    def walk(n: Node, path: tuple) -> None:
        if n.kind in ('ts', 'pair'):
            gi = grid.index(n.window)
            for gj in (gi - 1, gi + 1):
                if 0 <= gj < len(grid):
                    variants.append(AlphaTree(rebuild(tree.root, (), path, grid[gj])))
        for i, c in enumerate(n.children):
            walk(c, path + (i,))

    walk(tree.root, ())
    return variants


def moment_battery(tree: AlphaTree, panels: Dict[str, np.ndarray],
                   cfg: EvalConfig = EvalConfig()) -> Dict[str, float]:
    """四矩 train-only 稳健性电池(维度互异,非时间切片 —— 折一致性 07-22 证否勿复活):
      M1_lag     执行滞后:信号延迟 1 天,edge>0(脆弱时点依赖=噪声签名)
      M2_weekly  周频持仓:信号每 7 行采样 ffill,edge>0(真机制跨视界存活)
      M3_plateau 参数高原:窗口网格邻域最小 edge>0(无窗树缺席 NaN,不入分母)
      M4_rank    序数投影:敞口改走 ts_rank(120) rank-gate 书,edge>0(edge 由排序信息承载)
    回溯标定(2026-07-24,六池 70 已付费 val 触碰树,scripts/diag_moment_battery.py):
    ρ(通过率,val)=+0.30、gap(≥3/4 vs ≤1/4)=+0.31 双过预注册判据;单矩 ρ:lag +0.46 /
    rank +0.44 / weekly +0.29 / plateau +0.17。安装规则:**全矩通过(n_pass==n_present)
    才触碰 val**(全过组 val median +0.79 vs 队列 +0.44)。已知代价:会拦截个别高 val
    脆弱树(fee/vol-kurt 族星 2/4),精度优先与二级门同哲学。
    2026-07-25 crypto_daily 窗口网格改档(7/30/90/120/180/240/365,金融学意义档):
    M3 邻域随之变(120→90/180、240→180/365);回溯标定 ρ 数字按旧网格(3..240)计,
    方向性结论沿用,旧网格窗(3/5/10/20/40/60)的历史树跑 M3 会 grid.index 大声崩。"""
    values = np.ascontiguousarray(eval_tree(tree, panels, None), dtype=np.float64)
    ctx = _QNT_CTX
    out: Dict[str, float] = {}
    v = np.full_like(values, np.nan)
    v[1:] = values[:-1]
    out['M1_lag'] = _qnt_ts_net(v)[0]
    idx = (np.arange(values.shape[0]) // 7) * 7
    out['M2_weekly'] = _qnt_ts_net(np.ascontiguousarray(values[idx]))[0]
    vs = _window_neighbor_trees(tree)
    out['M3_plateau'] = (min(evaluate_alpha(t2, panels, cfg, None)['net_sharpe']
                             for t2 in vs) if vs else float('nan'))
    W = qnt_ts_weights(values, ctx['liquid'], 120)
    W[:ctx['row0']] = 0.0
    res = run_bt(np.ascontiguousarray(W), ctx['static'], ctx['row0'], ctx['bcfg'])
    out['M4_rank'] = float(res['sharpe']) - ctx['bench_sharpe']
    present = [k for k, x in out.items() if np.isfinite(x)]
    out['n_present'] = len(present)
    out['n_pass'] = int(sum(out[k] > 0.0 for k in present))
    return out
