"""EntropyTBGFlowNet —— torchgfn 2.4.1 适配版(2026-06-23)。

TBGFlowNet + 可选熵正则:loss = (scores + logZ)² .mean() − entropy_coef·entropy.mean()。
熵项鼓励探索(promote diversity,GFlowNet anti-mode-collapse 训练侧旋钮),需采样时
`save_estimator_outputs=True` 暴露 per-step logits;entropy_coef=0 → 直接走父类 plain TB loss。

**2.4.1 API 差**:`get_trajectories_scores` 已删,改用 `get_scores`;`Trajectories.when_is_done` 仍在。
"""
from __future__ import annotations

import torch
from gfn.containers import Trajectories
from gfn.env import Env
from gfn.gflownet.trajectory_balance import TBGFlowNet
from gfn.modules import DiscretePolicyEstimator


class EntropyTBGFlowNet(TBGFlowNet):
    def __init__(self, pf: DiscretePolicyEstimator, pb: DiscretePolicyEstimator,
                 entropy_coef: float = 0.0, entropy_temperature: float = 1.0, **kwargs):
        super().__init__(pf, pb, **kwargs)
        self.entropy_coef = entropy_coef
        self.entropy_temperature = entropy_temperature

    def loss(self, env: Env, trajectories: Trajectories,
             recalculate_all_logprobs: bool = True,
             reduction: str = 'mean',
             *,
             log_rewards: torch.Tensor | None = None) -> torch.Tensor:
        # plain TB loss(父类的处理 logZ + reduction;entropy_coef=0 时零开销)
        loss = super().loss(env, trajectories,
                            recalculate_all_logprobs=recalculate_all_logprobs,
                            reduction=reduction, log_rewards=log_rewards)
        if self.entropy_coef <= 0 or trajectories.estimator_outputs is None:
            return loss

        # 熵项(每条轨迹有效步的 logits 熵之和;done/sink 步 logits = -inf → softmax NaN → where 置 0)
        logits = trajectories.estimator_outputs            # (max_len, batch, n_actions)
        temp_logits = logits / self.entropy_temperature
        probs = torch.softmax(temp_logits, dim=-1)
        entropy_per_step = -torch.sum(probs * torch.log(probs.clamp(min=1e-9)), dim=-1)
        max_len = logits.shape[0]
        step_indices = torch.arange(max_len, device=logits.device).unsqueeze(1)
        valid_mask = step_indices < trajectories.terminating_idx.unsqueeze(0)   # 2.4.1 改名
        masked_entropy = torch.where(valid_mask, entropy_per_step,
                                     torch.zeros_like(entropy_per_step))
        # **每步均值**(非 sum):sum 会让长轨迹累积更多熵 → 减 coef·熵 实际奖励变长 → 助推 mean_len 失控。
        # 改 mean(÷有效步数)= 长度无关的"每决策熵",只鼓励每步多样、不奖励轨迹拉长。
        n_steps = valid_mask.sum(dim=0).clamp(min=1)       # (batch,) 每轨迹有效步数
        entropy_term = masked_entropy.sum(dim=0) / n_steps  # (batch,) 每步平均熵
        return loss - self.entropy_coef * entropy_term.mean()
