"""AlphaSAGE GFN reward 桥(2026-06-24):seq → decode → evaluate_alpha → R = R_IC + λ(T)·R_SA + η·R_NOV。

忠实复刻 AlphaSAGE 三项奖励(reward = R_Q + λ·R_SA + η·R_NOV,floor 到 _R_FLOOR ≈ exp(-10)):
    R_Q   = |候选 top-K gross Sharpe|             —— gross-aware 质量(2026-06-24 由 |rank IC| 改;治 IC↔PnL
            背离:rv_* 类特征 IC 高但 top-K 极端押注 PnL 烂,|IC| 奖励它们 → OOS 崩,Sharpe 不奖励)
    R_SA  = 结构感知**稠密**奖励(论文核心 = 参考 `alpha_pool.compute_ssl_reward`,专治 reward 稀疏):
            候选的 RGCN 结构签名 e 在**池成员**里取 k 个最近结构邻居,
            w_k = softmax(−‖e−e_k‖²/τ)(结构嵌入相似度权重),
            d_behav_k = 标准化 per_t_pnl 的 MSE(= 2(1−corr);行为空间距离,尺度无关),
            consistency = Σ_k w_k·d_behav_k,R_SA = exp(−consistency) ∈ (0,1]。
            "结构相似 → 行为应相似":结构邻域行为越一致 R_SA 越高 → 把稀疏 IC 平滑成稠密结构信号。
            池 < 2 成员 → 0(同参考 size<=1 路径)。
    R_NOV = 1 − R²,R² = max_k corr(候选 per_t_pnl, 成员_k)²  —— 与池最像成员的平方 PnL 相关 = 正交新颖度
    λ(T)/η 由 trainer 写入 sched(λ 高→低:早期靠结构稠密先验导航探索,晚期让 R_IC 主导;η 常数 = 真正交门)。
每个采样 alpha 评估后 `pool.try_new` 生长正交池(自带 IC+Δ+holdout 准入门;池 admission = IC+行为新颖,
与参考一致,**不开** embedding G2 结构去重 → 不动已验证 eval gate)。非法/越成本/below-gate → R=floor,不入池。
复用 evaluator + LinearAlphaPool = 接入既有基础设施;R_SA/RGCN 是方法层增强,不改基础设施语义。
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List

import numpy as np

from evaluation.cache import EvalCache
from rl.alpha_pool import AlphaPool
from rl.evaluator import EvalConfig, evaluate_alpha

from search.alphasage.gfn_codec import cost_ok, decode, is_complete

# GFlowNet reward 必须 > 0(log_rewards = log(R));非法/below-gate 给正 floor(= AlphaSAGE exp(-10))。
# 注意:不可复用 evaluator._REWARD_FLOOR(= -2.0,是 GP/DEAP fitness 哨兵,log 会 NaN)。
_R_FLOOR = math.exp(-10)


def _standardize(v: np.ndarray) -> np.ndarray:
    """逐序列 z-score(零均值单位方差,NaN→0);标准化后 MSE = 2(1−corr),尺度无关有区分度。"""
    x = np.nan_to_num(v, nan=0.0)
    s = x.std()
    return (x - x.mean()) / s if s > 1e-12 else x * 0.0


def _r_sa(cand_emb: np.ndarray, cand_pnl_std: np.ndarray, pool: AlphaPool,
          embedder: Callable[[List], np.ndarray], k: int, tau: float) -> float:
    """R_SA = exp(−Σ w_k·d_behav_k):结构 kNN(池成员)+ 行为一致性(标准化 per_t_pnl MSE)。"""
    members = pool.members
    if len(members) < 2:
        return 0.0
    memb_embs = embedder([m.tree for m in members])              # (n,H) 新鲜结构签名(无 staleness)
    dist2 = np.sum((memb_embs - cand_emb[None, :]) ** 2, axis=1)  # 结构嵌入平方距离
    kk = min(k, len(members))
    idx = np.argpartition(dist2, kk - 1)[:kk]
    w = np.exp(-(dist2[idx] - dist2[idx].min()) / tau)          # softmax(−dist²/τ),减最小值稳数值
    w = w / w.sum()
    consistency = 0.0
    for wi, mi in zip(w, idx):
        d_behav = float(np.mean((cand_pnl_std - _standardize(members[mi].per_t_pnl)) ** 2))
        consistency += float(wi) * d_behav
    return math.exp(-consistency)


class _Stats:
    """批级 reward 诊断累加器(区分模式坍缩 vs IC 噪声墙)。trainer 每 log 窗口读 + reset。"""
    __slots__ = ('n', 'floor_incomplete', 'floor_cost', 'floor_belowgate',
                 'valid', 'hashes', 'valid_abs_ics', 'len_sum', 'sa_sum')

    def __init__(self):
        self.reset()

    def reset(self):
        self.n = 0
        self.floor_incomplete = 0
        self.floor_cost = 0
        self.floor_belowgate = 0
        self.valid = 0
        self.hashes = set()
        self.valid_abs_ics = []
        self.len_sum = 0
        self.sa_sum = 0.0


def make_reward_fn(panels: Dict[str, np.ndarray], y: np.ndarray,
                   pool: AlphaPool, nov_weight: float,
                   embedder: Callable[[List], np.ndarray], sched: Dict[str, float],
                   r_sa_k: int, r_sa_tau: float,
                   cache_bytes: int = 512 * 1024 ** 2) -> Callable[[List[int]], float]:
    """构造 GFNEnv 注入的 reward_fn(闭包持 panels/y/pool/cache/embedder/sched)。
    embedder:List[AlphaTree] → (n,H) 结构签名(= RGCN.embed_trees,R_SA 用)。
    sched:trainer 每 episode 写 {'lam':λ(T), 'eta':η}(λ 时变退火)。
    pool 在训练中被原地生长;reward_fn.stats 暴露批级诊断(trainer 读取)。"""
    cfg = EvalConfig()
    cache = EvalCache(max_bytes=cache_bytes)
    stats = _Stats()

    def reward_fn(seq: List[int]) -> float:
        stats.n += 1
        stats.len_sum += len(seq)
        if not seq or not is_complete(seq):
            stats.floor_incomplete += 1
            return _R_FLOOR
        tree = decode(seq)
        stats.hashes.add(tree.hash)                           # 唯一树计数(模式坍缩信号)
        if not cost_ok(tree):
            stats.floor_cost += 1
            return _R_FLOOR
        m = evaluate_alpha(tree, panels, y, pool, cfg, cache, diagnostics=False)
        ptp = m['per_t_pnl']
        if ptp is None:                                       # below-gate / 坏结构
            stats.floor_belowgate += 1
            return _R_FLOOR
        stats.valid += 1
        stats.valid_abs_ics.append(abs(m['rank_ic']))           # diag 仍记 |IC| 作参考(reward 已改 Sharpe)
        # 质量 = |top-K gross Sharpe|(gross-aware reward,治 |IC| 的 IC↔PnL 背离:rv_* 类 IC 高但极端 PnL 烂 → Sharpe 低)
        r_q = abs(m['topk_sharpe'])
        r_nov = 1.0 - pool._proj_r2(pool._corr_with_pool(ptp))   # 1 − max_k corr²(候选 vs 池)
        # R_SA 在 try_new 之前算(候选尚未入池 → 不会成为自身邻居,同参考 exclude_self)。
        cand_emb = embedder([tree])[0]                        # RGCN 结构签名(H,)
        r_sa = _r_sa(cand_emb, _standardize(ptp), pool, embedder, r_sa_k, r_sa_tau)
        stats.sa_sum += r_sa
        # 池按 Sharpe 质量(cand_quality)留/剔;PoolMember.rank_ic 仍存真 IC 给 bundle/参考。
        pool.try_new(tree, m['rank_ic'], ptp, cand_quality=r_q)
        return max(r_q + sched['lam'] * r_sa + sched['eta'] * r_nov, _R_FLOOR)

    reward_fn.stats = stats
    return reward_fn
