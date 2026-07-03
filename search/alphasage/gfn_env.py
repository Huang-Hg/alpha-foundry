"""AlphaSAGE GFlowNet 环境(torchgfn 2.4.1 适配,2026-06-23)。

state = 定长 production-id 序列(padding=-1);action = 给最左 pending slot 选 production(由
`gfn_codec.legal_mask` 给合法 mask)或 EXIT(完成推导 → sink)。

**2.4.1 API 重构**:DiscreteEnv 不再吃 `device_str`/`preprocessor`,`update_masks` 已删,mask
改成 `DiscreteStates` 子类的 `_compute_forward_masks`/`_compute_backward_masks` 属性;step/
backward_step 必须返回 `States` 对象(非裸 tensor)。preprocessor 挂在 `self.preprocessor`,
供 `DiscretePolicyEstimator` 取。
"""
from __future__ import annotations

from typing import Callable, ClassVar, List, Optional

import numpy as np
import torch
from gfn.actions import Actions
from gfn.env import DiscreteEnv
from gfn.preprocessors import Preprocessor
from gfn.states import DiscreteStates

from search.alphasage.gfn_codec import (
    MAX_DERIV_LEN, exit_action as _exit_action, legal_mask, n_actions as _n_actions,
)
# 动作空间随 active 词表运行时变 → 一律在**构造时**读 _n_actions()/_exit_action()(此时 grammar
# 词表已定),mask / step 内用 state 自身 n_actions 保持自洽。消除 import 顺序静默陷阱。


class _IntPreprocessor(Preprocessor):
    """state 张量原样喂 encoder(encoder 内部自己 embed),不转 float。"""
    def __init__(self, output_dim: int):
        super().__init__(output_dim=output_dim)

    def preprocess(self, states) -> torch.Tensor:
        return states.tensor


def _seq(state_tensor: torch.Tensor) -> List[int]:
    return [int(x) for x in state_tensor.tolist() if x >= 0]


class GFNStates(DiscreteStates):
    """文法推导序列 States;mask 由 `gfn_codec.legal_mask`(typed CFG)逐 state 算。
    `_compute_forward_masks` / `_compute_backward_masks` 是 cached property(torchgfn 2.4.1)。
    """
    # 由 make_states_class 注入(类属性,不是实例)
    state_shape: ClassVar = (MAX_DERIV_LEN,)
    s0: ClassVar[torch.Tensor]
    sf: ClassVar[torch.Tensor]
    n_actions: ClassVar[int] = _n_actions()        # 占位默认;真实值由 make_states_class 按构造时 env 注入

    def _compute_forward_masks(self) -> torch.Tensor:
        """forward mask:每 state (N_ACTIONS,) bool。完成→仅 EXIT、未完成→栈顶 slot 合法 production。
        sink/done state(完整 tensor==sf)→ 全 False(sink 无前向动作)。
        **状态 tensor 还差 1 槽就写满 → 强制只 EXIT**(避免 sampler 在 incomplete-deriv 上无限循环;
        backward 自然反推,trajectory 仍是标准 EXIT→sink)。"""
        flat = self.tensor.view(-1, *self.state_shape)
        B = flat.shape[0]
        out = np.zeros((B, self.n_actions), dtype=bool)
        sink_flat = self.is_sink_state.view(-1)
        for i in range(B):
            if bool(sink_flat[i]):
                continue
            seq = _seq(flat[i])
            if len(seq) >= MAX_DERIV_LEN - 1:             # 写一格就顶满 → 只 EXIT
                out[i, self.n_actions - 1] = True         # EXIT = 末位动作(= NUM_PRODUCTIONS)
            else:
                out[i] = legal_mask(seq)
        return torch.tensor(out, dtype=torch.bool, device=self.tensor.device
                            ).view(*self.batch_shape, self.n_actions)

    def _compute_backward_masks(self) -> torch.Tensor:
        """backward mask:(N_ACTIONS-1,)(EXIT 不可逆 → pb 头少一列)。唯一最左推导 ⇒ 父 = 撤销末位
        production,故 mask 只在该 token 位置为 True。s0 / sink → 全 False。"""
        flat = self.tensor.view(-1, *self.state_shape)
        B = flat.shape[0]
        out = np.zeros((B, self.n_actions - 1), dtype=bool)
        sink_flat = self.is_sink_state.view(-1)
        for i in range(B):
            if bool(sink_flat[i]):
                continue
            seq = _seq(flat[i])
            if seq:
                out[i, seq[-1]] = True
        return torch.tensor(out, dtype=torch.bool, device=self.tensor.device
                            ).view(*self.batch_shape, self.n_actions - 1)


class GFNEnv(DiscreteEnv):
    def __init__(self, reward_fn: Callable[[List[int]], float],
                 device: torch.device = torch.device('cpu')):
        self.reward_fn = reward_fn
        s0 = torch.full((MAX_DERIV_LEN,), -1, dtype=torch.long, device=device)   # 空推导
        sf = torch.full((MAX_DERIV_LEN,), -2, dtype=torch.long, device=device)   # sink 哨兵
        super().__init__(
            n_actions=_n_actions(),                       # 构造时按 active 词表定动作空间
            s0=s0,
            sf=sf,
            state_shape=(MAX_DERIV_LEN,),
            dummy_action=torch.tensor([-1], device=device),
            exit_action=torch.tensor([_exit_action()], device=device),
        )
        self.preprocessor = _IntPreprocessor(output_dim=MAX_DERIV_LEN)

    def make_states_class(self) -> type:
        env_self = self

        class GFNStatesImpl(GFNStates):
            state_shape = (MAX_DERIV_LEN,)
            s0 = env_self.s0
            sf = env_self.sf
            n_actions = env_self.n_actions

        return GFNStatesImpl

    def step(self, states: DiscreteStates, actions: Actions) -> DiscreteStates:
        """EXIT → sink;否则把 production 写入栈顶。forward_mask 已保证 filled<MAX_DERIV_LEN-1
        时才允许非 EXIT(filled==MAX-1 只 EXIT),不会越界。"""
        nxt = states.tensor.clone()
        a_flat = actions.tensor.view(-1)
        for i in range(nxt.shape[0]):
            aid = int(a_flat[i].item())
            if aid == self.n_actions - 1:                 # EXIT = 末位动作
                nxt[i] = self.sf
            else:
                filled = int((nxt[i] >= 0).sum())
                nxt[i, filled] = aid
        return self.States(nxt)

    def backward_step(self, states: DiscreteStates, actions: Actions) -> DiscreteStates:
        """唯一最左推导 → 父 = 去掉最后一个 production(把末位有效 token 置 -1)。"""
        prev = states.tensor.clone()
        for i in range(prev.shape[0]):
            filled = int((prev[i] >= 0).sum())
            if filled > 0:
                prev[i, filled - 1] = -1
        return self.States(prev)

    def reward(self, final_states: DiscreteStates) -> torch.Tensor:
        flat = final_states.tensor.view(-1, *final_states.state_shape)
        r = [self.reward_fn(_seq(st)) for st in flat]
        return torch.tensor(r, dtype=torch.float, device=self.device
                            ).view(final_states.batch_shape)
