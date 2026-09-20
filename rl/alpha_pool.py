"""
LinearAlphaPool — 正交 alpha 池(2026-07-22 时序书口径;2026-07-25 起两引擎传入的
质量 q = R4 滚动折中位 edge = evaluator['cv_edge'],>0 ⇔ ≥3/5 折胜锚;此前 = 全窗
edge = evaluator['net_sharpe'];旧截面 softmax 书超额 IR / |rank IC| 口径随截面→时序
转向退役。池本身对 q 的口径不感知,只作幅度)。

admission / 留池:质量幅度 q(>0 才入池;≤0 在 try_new 拒 —— 选不出超额=无用)。
  正交贡献 **Δ = q²·(1−R²)**(2026-07-06 DPP 边际增益,质量进平方;核 L_ij=q_i q_j ρ_ij,加入 i 的
  log-det 增益 = log(q_i²(1−R²_{i|S})),排序等价 q²(1−R²)),**R² = 候选 per_t_pnl 对全体成员 pnl 张成
  子空间的 OLS R²(demean 截距)** —— clone/族内兄弟(R²→1)→ Δ→0 自动拒。
    正交贡献门:Δ = q²·(1−R²) < delta_floor → 拒(质量弱或冗余)。
  每次 admit 后 leave-one-out prune,踢 argmin Δ_loo(=q²·(1−R²_loo)),触发:① size>capacity(硬顶 12)
  ② Δ_loo<floor → 更强/更独立候选挤掉弱钉子户,连续自清洁。
  (N_eff 自适应容量 + pnl_corrs 矩阵 2026-07-23 删:用户裁定冗余 —— 池尺寸只由 capacity 硬顶
  + Δ floor 控制。)池**不定权**(eval/deploy = 成员
  gate 等权 ensemble,见 main.eval_pool_val;两级等权定论)。

per_t_pnl(时序书口径)= rr_book − rr_ew 超额流(evaluator._qnt_ts_net 的 ex;供 R² 正交度量
+ 行为多样性,质量幅度走 edge)。

2026-07-22 冗余机制清除:PoolMember.rank_ic / embedding(参考量 + 僵尸嵌入去重支路,全管线
embed_fn=None 从未生效)删;库记录同步瘦身。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

from config.config import ini
from backtest import ops
from evaluation.ast import AlphaTree
from markets.profile import ACTIVE as _PROFILE


# ============================================================================
# Pool member
# ============================================================================

@dataclass
class PoolMember:
    tree:       AlphaTree
    per_t_pnl:  np.ndarray                     # (T,) rr_book − rr_ew 超额流,R² 正交度量用


# ============================================================================
# LinearAlphaPool
# ============================================================================

@dataclass
class LinearAlphaPool:
    capacity: int = 12                                    # 硬顶(2026-07-06 24→12)
    # 最小正交贡献 Δ = net_sharpe²·(1−R²)(DPP 边际增益,质量进平方):admission/prune 门 = Δ<floor → reject/evict。
    delta_floor:       float = ini('alpha_pool', 'delta_floor',       0.003)
    # deploy 环境可供的最大因子回看深度(bar 数):required_depth 超此的树每次决策整树
    # NaN→哑火,结构性硬拒。由 active MarketProfile 单源。
    max_panel_depth:   int   = _PROFILE.deploy_depth_bars

    members:  List[PoolMember] = field(default_factory=list)
    # 平行数组(与 members 同序)。single_q = 各成员 edge(admission/prune 质量幅度,恒 >0)。
    single_q:   np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    _failure_cache: Set[int] = field(default_factory=set, repr=False)

    # ----- 基本 query -----

    @property
    def size(self) -> int:
        return len(self.members)

    def to_jsonable(self) -> List[Dict]:
        return [{'tree':   m.tree.to_dict(),
                 'pretty': m.tree.pretty(),
                 'edge':   float(q)}
                for m, q in zip(self.members, self.single_q)]

    # ----- 正交 Δ 度量(reward-无关准入)-----

    @staticmethod
    def _ols_r2(X: np.ndarray, y: np.ndarray) -> float:
        """demean OLS R²(截距吸收);y 零方差 → 1.0(无信息 pnl 视同全冗余)。"""
        yc = y - y.mean()
        vy = float(yc @ yc)
        if vy < 1e-18:
            return 1.0
        Xc = X - X.mean(axis=0, keepdims=True)
        beta, *_ = np.linalg.lstsq(Xc, yc, rcond=None)
        resid = yc - Xc @ beta
        return float(min(max(1.0 - resid @ resid / vy, 0.0), 1.0))

    def subspace_r2(self, cand_per_t_pnl: np.ndarray) -> float:
        """冗余度 R²(子空间口径,2026-07-05):候选 per_t_pnl 对**全体成员 pnl 联合张成子空间**
        的 OLS R²。clone → 1、与池整体正交 → 0、空池 → 0。gfn_reward R_NOV 同口径复用。"""
        if self.size == 0:
            return 0.0
        X = np.nan_to_num(np.stack([m.per_t_pnl for m in self.members], axis=1), nan=0.0)
        return self._ols_r2(X, np.nan_to_num(cand_per_t_pnl, nan=0.0))

    def _leave_one_out_r2(self) -> np.ndarray:
        """每个成员的 leave-one-out R²_i = 成员 i 的 pnl 对"其余 n-1 个成员联合子空间"的
        OLS R² = 把成员 i 当新候选重算一遍 admission R²。n≤1 → 0。"""
        n = self.size
        out = np.zeros(n, dtype=np.float64)
        if n <= 1:
            return out
        P = np.nan_to_num(np.stack([m.per_t_pnl for m in self.members], axis=1), nan=0.0)
        for i in range(n):
            out[i] = self._ols_r2(np.delete(P, i, axis=1), P[:, i])
        return out

    def _leave_one_out_deltas(self) -> np.ndarray:
        """每个成员的 leave-one-out Δ_i = net_sharpe_i²·(1−R²_loo_i)(DPP 边际增益):把成员 i 当
        新候选对其余成员重算 admission Δ。小 = edge 弱 / 冗余 → prune 优先踢。n≤1 → R²=0 → Δ=q²。"""
        return self.single_q ** 2 * (1.0 - self._leave_one_out_r2())

    # ----- admission(commit-then-revert)-----

    def try_new(
        self, cand_tree: AlphaTree, cand_per_t_pnl: np.ndarray, cand_quality: float,
    ) -> Tuple[bool, float]:
        """commit-then-revert admission。返回 (accepted, 0.0)。
        cand_per_t_pnl 必传(R² 正交度量 + 行为多样性);
        cand_quality = 质量幅度 = 候选时序书 edge(带符号;≤0 直接拒、>0 才入池;
          admission/prune 的 single_q 源)。池不定权(eval/deploy = 成员 gate 等权 ensemble)。"""
        q = cand_quality
        sig = cand_tree.hash
        if sig in self._failure_cache:
            return False, 0.0
        # edge≤0 不入池(long-only 下负 edge 因子无法反向利用 —— 质量幅度须 >0;
        # GFN reward 仍见带符号 r_q 学梯度,池只收正 edge)
        if not (cand_quality > 0.0):
            self._failure_cache.add(sig)
            return False, 0.0
        # 深度硬门:嵌套窗+operand warmup 超部署 panel 物理深度的树,离线 T 充裕评得动、
        # 部署时每次决策整树 NaN(深度死)→ 结构性拒,不进池。
        if cand_tree.required_depth() > self.max_panel_depth:
            self._failure_cache.add(sig)
            return False, 0.0

        # 正交贡献门(DPP 边际增益 Δ = q²·(1−R²);子空间 (1−R²) 统一涵盖克隆:R²→1 → Δ→0 自动拒)。
        n = self.size
        r2 = self.subspace_r2(cand_per_t_pnl)
        delta = q * q * (1.0 - r2)                               # DPP:质量进平方
        if delta < self.delta_floor:
            self._failure_cache.add(sig)
            return False, 0.0

        # tentative admit(先无条件 commit,prune loop 再裁谁留)
        new_member = PoolMember(
            tree=cand_tree, per_t_pnl=cand_per_t_pnl.astype(np.float64, copy=True),
        )
        cand_idx = n                               # 候选下标(prune 中随删随减)
        self.members.append(new_member)
        self.single_q = np.concatenate([self.single_q, [q]])

        # 连续 net-aware prune:不论满没满,leave-one-out 当留池判据,growth 阶段就剔净弱 /
        #   变冗余的旧成员,不永久占坑。踢 argmin Δ_loo(净最弱/最冗余先走),触发条件:
        #   ① over_cap:size > capacity(硬顶 12)
        #   ② (Δ_loo < floor).any():有成员贡献跌破门
        #   全不触发 → break。
        while self.size > 1:
            deltas = self._leave_one_out_deltas()      # 成员 Δ=net_sharpe²·(1−R²_loo) 留池贡献
            over_cap = self.size > self.capacity
            if over_cap or (deltas < self.delta_floor).any():
                worst = int(np.argmin(deltas))         # 踢 argmin Δ_loo(净最弱/最冗余)
            else:
                break
            if worst == cand_idx:
                self._evict(cand_idx)               # 候选自己最该踢(净最弱/最冗余)→ revert
                self._failure_cache.add(sig)
                return False, 0.0
            self._evict(worst)
            if worst < cand_idx:
                cand_idx -= 1

        self._failure_cache = set()
        return True, 0.0

    def _evict(self, idx: int) -> None:
        """删除 members[idx] + 同步删 single_q。"""
        n = self.size
        self.members.pop(idx)
        keep = [i for i in range(n) if i != idx]
        self.single_q = self.single_q[keep]


# 向后兼容别名(main.py / evaluator.py 用 AlphaPool 名字 import)
AlphaPool = LinearAlphaPool


# ============================================================================
# 跨 run 累积的正交 alpha 库(2026-06-22,原 alpha_library 并入)
#
# 只存因子树(tree/pretty/quality=edge + 源 run),**不存 pnl 序列**(太大)。
# - `_greedy_orthogonal`:按 edge 降序贪心,保留与已选 pnl |pearson| < corr_max 的(C 算子 pnl_corr_vec)。
# - `append_run`:run 尾用内存中 collected 的 pnl 贪心挑 within-run 正交高 edge,append 到库
#   (只 within-run 正交;跨 run 冗余 + 衰减 + 封顶留给定时清理 `scripts/clean_alpha_library.py`)。
#
# 质量门 = edge>0(alpha_pool.try_new 准入,below-gate 候选 per_t_pnl 不进 collected),不另设 knob。
# ============================================================================

def _greedy_orthogonal(items: List[Tuple], corr_max: float, cap: int = 0) -> List[Tuple]:
    """items=[(tree, …, per_t_pnl), …] 已按 edge 降序(pnl 恒在末位 it[-1])。
    贪心保留与已选 pnl |pearson| < corr_max 的;cap>0 时满 cap 即止。返回保留项(原 tuple 不变)。"""
    kept: List[Tuple] = []
    kept_pnl: List[np.ndarray] = []
    for it in items:
        ptp = np.ascontiguousarray(it[-1], dtype=np.float64)
        if kept_pnl:
            M = np.ascontiguousarray(np.stack(kept_pnl))
            if float(np.abs(ops.pnl_corr_vec(ptp, M)).max()) >= corr_max:
                continue
        kept.append(it)
        kept_pnl.append(ptp)
        if cap and len(kept) >= cap:
            break
    return kept


def append_run(collected: Dict, lib_path: Path, corr_max: float, max_size: int, run_ts: str) -> int:
    """从本轮 collected(值=(tree, q=edge, ptp))贪心挑 within-run 正交高 **edge**,
    append 到库文件(只追加,跨 run 冗余留给清理)。返回本轮入库数。"""
    items = sorted(collected.values(), key=lambda x: -x[1])    # (tree, q, ptp),pnl 在末位
    kept = _greedy_orthogonal(items, corr_max, cap=max_size)
    existing = json.loads(lib_path.read_text(encoding='utf-8')) if lib_path.exists() else []
    for tree, quality, _ in kept:
        existing.append({'tree': tree.to_dict(), 'pretty': tree.pretty(),
                         'quality': float(quality), 'run': run_ts})
    lib_path.parent.mkdir(parents=True, exist_ok=True)
    lib_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding='utf-8')
    return len(kept)
