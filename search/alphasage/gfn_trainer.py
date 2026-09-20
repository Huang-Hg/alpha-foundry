"""AlphaSAGE GFlowNet alpha-mining baseline trainer(2026-06-24)。

**独立 baseline,与 GP 平级,不碰 GP**。P(α)∝R 采样:GFN(**RGCN 结构感知编码器** + DiscretePolicyEstimator
pf/pb,在最左推导空间生成 alpha)+ EntropyTBGFlowNet TB loss(熵正则防早熟坍缩);
reward = R_IC + λ(T)·R_SA + η(T)·R_NOV(`gfn_reward`,复用 evaluator + LinearAlphaPool;R_SA 结构感知
稠密奖励 + λ/η 时变退火 = AlphaSAGE 方法层三支柱)。训练同时把高 R 正交 alpha 收进 LinearAlphaPool。
返回 (pool, logz_hist)。OOS 用 `main.eval_pool_val`(与 GP 同一 judge)→ 平行对比 GP。
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
# cupy ops_cuda 必须在 `import torch` 前编译加载:torch cu128 的 CUDA 运行时库会 shadow cupy(cuda12.9),
# 致 Blackwell sm_120(rtx-5090)上 cupy RawModule 核 CUDA_ERROR_NO_BINARY_FOR_GPU(V100 sm_70 / T1000
# sm_75 无此问题)。cupy-first 加载好自己的库后,torch 后到不再覆盖。仅 device=cuda 触发。
from config.config import ini as _ini
if _ini('evaluator', 'device', 'cpu') == 'cuda':
    import backtest.ops_cuda  # noqa: F401  (warm-up:先于 torch 加载 cupy CUDA 库)
import torch
from gfn.modules import DiscretePolicyEstimator
from gfn.samplers import Sampler
# torchgfn 2.4.1:`NeuralNet` 已删,`MLP` 强制 ≥1 hidden layer;head 用纯 nn.Linear 接 RGCN backbone。
from torch import nn
from torch.optim import Adam

from rl.alpha_pool import AlphaPool
from rl.evaluator import qnt_bench_sharpe

from search.alphasage.gfn_env import GFNEnv
from search.alphasage.gfn_loss import EntropyTBGFlowNet
from search.alphasage.gfn_reward import make_reward_fn
from search.alphasage.gfn_rgcn import RGCNEncoder

def train_alphasage(
    panels: Dict[str, np.ndarray], capacity: int, *,
    hidden_dim: int, n_layers: int, lr: float, batch_size: int,
    n_episodes: int, nov_weight: float, entropy_coef: float, entropy_temperature: float,
    r_sa_weight: float, r_sa_k: int, r_sa_tau: float, logz_lr: float,
    seed: int = 42, device: str = 'cpu', log=print,
) -> Tuple[AlphaPool, List[float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)

    pool = AlphaPool(capacity=capacity)

    # 编码器 = 方法层核心(非基础设施):RGCN 在 partial AST 上做关系感知嵌入。pf/pb 各自独立
    # backbone(共享会让 Adam 见 duplicate params)。pf_backbone 还兼任 R_SA 的结构签名来源(embed_trees)。
    pf_backbone = RGCNEncoder(hidden_dim=hidden_dim, n_layers=n_layers).to(dev)
    pb_backbone = RGCNEncoder(hidden_dim=hidden_dim, n_layers=n_layers).to(dev)

    sched: Dict[str, float] = {'lam': r_sa_weight, 'eta': nov_weight}   # trainer 每 episode 退火更新
    reward_fn = make_reward_fn(panels, pool, nov_weight,
                               embedder=pf_backbone.embed_trees, sched=sched,
                               r_sa_k=r_sa_k, r_sa_tau=r_sa_tau)
    env = GFNEnv(reward_fn=reward_fn, device=dev)

    pf_head = nn.Linear(hidden_dim, env.n_actions)
    pb_head = nn.Linear(hidden_dim, env.n_actions - 1)
    pf = DiscretePolicyEstimator(nn.Sequential(pf_backbone, pf_head), n_actions=env.n_actions,
                                 preprocessor=env.preprocessor)
    pb = DiscretePolicyEstimator(nn.Sequential(pb_backbone, pb_head), n_actions=env.n_actions,
                                 preprocessor=env.preprocessor, is_backward=True)
    gflownet = EntropyTBGFlowNet(pf=pf, pb=pb, entropy_coef=entropy_coef,
                                 entropy_temperature=entropy_temperature).to(dev)
    sampler = Sampler(estimator=pf)
    # **logZ 单独高 lr**(GFlowNet 铁律):logZ 须追 log Σ_terminals R(此空间数 nat 级),与 policy 共用
    # lr=1e-3 会让 logZ 几乎不动 → TB 残差 (logZ+ΣlogP_F−logR)² 巨大 → TB 发散 → policy 被脏梯度推向长轨迹
    # 坍缩。分组:policy 用 lr,logZ 用 logz_lr(~0.1)。
    policy_params = [p for n, p in gflownet.named_parameters() if n != 'logZ']
    opt = Adam([{'params': policy_params, 'lr': lr},
                {'params': [gflownet.logZ], 'lr': logz_lr}])
    save_outputs = entropy_coef > 0

    import time
    bench = qnt_bench_sharpe()          # 日志绝对 Sharpe 口径(内部 reward=edge,读数=edge+锚)
    logz_hist: List[float] = []
    every = max(1, n_episodes // 50)
    denom = max(1, n_episodes - 1)
    for ep in range(n_episodes):
        p = ep / denom                                  # 训练进度 ∈ [0,1]
        sched['lam'] = r_sa_weight * (1.0 - p)          # λ(T) 高→低:结构稠密先验早期导航,晚期让 R_IC 主导
        sched['eta'] = nov_weight                       # η(T):真正交门,常数(非 warmup)
        t0 = time.perf_counter()
        traj = sampler.sample_trajectories(env=env, n=batch_size,
                                           save_estimator_outputs=save_outputs)
        t1 = time.perf_counter()
        loss = gflownet.loss(env=env, trajectories=traj)
        t2 = time.perf_counter()
        opt.zero_grad()
        loss.backward()
        opt.step()
        t3 = time.perf_counter()
        logz_hist.append(float(gflownet.logZ.item()))
        ep_total = t3 - t0
        if ep % every == 0 or ep == n_episodes - 1 or ep_total > 1.5:
            best = float(pool.single_q.max()) + bench if pool.size else float('nan')
            s = reward_fn.stats
            n = max(1, s.n)
            edges = np.asarray(s.valid_edges) if s.valid_edges else np.array([0.0])
            log(f"[asage] ep{ep}/{n_episodes} loss={loss.item():+.3f} "
                f"logZ={gflownet.logZ.item():+.3f} pool={pool.size}/{capacity} "
                f"best_sharpe={best:+.4f} lam={sched['lam']:.3f} "
                f"t_sample={t1-t0:.2f}s t_loss={t2-t1:.2f}s t_opt={t3-t2:.2f}s")
            log(f"[asage-diag] win_calls={s.n} uniq_trees={len(s.hashes)} "
                f"valid={100*s.valid/n:.0f}% floor[incomp={100*s.floor_incomplete/n:.0f}%/"
                f"cost={100*s.floor_cost/n:.0f}%/belowgate={100*s.floor_belowgate/n:.0f}%] "
                f"mean_len={s.len_sum/n:.1f} "
                f"validSharpe[mean={edges.mean()+bench:+.3f} max={edges.max()+bench:+.3f}] "
                f"R_SA_mean={s.sa_sum/n:.4f}")
            s.reset()
    log(f"[asage] done: pool_size={pool.size}/{capacity}")
    return pool, logz_hist
