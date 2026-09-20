"""AlphaSAGE 结构感知编码器:RGCN over AST(2026-06-24)。

替代已删的 LSTM 序列编码 —— AlphaSAGE 方法层核心(非基础设施)。论文要点:把 alpha 的 AST 当
**带类型边的图**,关系特定权重让编码器区分 `sub(close, open) ≠ sub(open, close)`(left/right 不同
relation),又让 `add(close, open) ≡ add(open, close)`(commutative 同 relation,与 AST 交换律
规范化一致);LSTM/RPN 序列编码恰是它要解决的缺陷。

**忠实对齐参考实现 `alphasage/src/alpha_gfn/modules.py::GNNEncoder`**(torch_geometric RGCNConv):
    6 个 relation 按**算子角色**:0=unary / 1=commutative binary / 2=non-comm left / 3=non-comm right
                                  / 4=feature operand / 5=time-series(window)operand
    边方向 child→parent(自底向上),n_layers 跳 RGCN(ReLU),global mean-pool → (B, hidden)。
    RGCNConv 默认 root_weight=True(自连接 W0·x_i)→ 我们的 self-loop 线性等价。
本实现不引 torch_geometric:用 index_add scatter 手写 R-GCN(server venv 仅 torch + cupy)。
state = 最左推导 production-id 序列(padded -1);forward 内重放成 AST 图。pf/pb 各自独立实例
(不共享 → Adam 不见 duplicate params)。`embed_trees` 另给 R_SA 稠密奖励提供整树结构签名(mean-pool,no_grad)。
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from torch import nn

from evaluation import grammar as _grammar
from evaluation.ast import AlphaTree, Node
from evaluation.grammar import BinaryOp, ConstBinaryOp
from search.alphasage.gfn_codec import MAX_DERIV_LEN

# PRODUCTIONS / NUM_PRODUCTIONS 运行时从 _grammar 读(换市场 set_vocabulary 会重建)→ embedding
# 尺寸在**构造时**按 active 词表定,id 映射按 id(PRODUCTIONS) 缓存自动失效。消除 import 顺序陷阱。

# relation 数(按算子角色;参考 GNNEncoder=6)。
N_RELATIONS: int = 6

# commutative 算子集合(子序不影响结果 → 共用一个对称 relation,与 AST 的交换律 hash 规范化一致)。
_COMM_BIN = frozenset({BinaryOp.ADD, BinaryOp.MUL, BinaryOp.MAX, BinaryOp.MIN})
_COMM_CONST = frozenset({ConstBinaryOp.ADD_CONST, ConstBinaryOp.MUL_CONST})


def _edge_type(parent_kind: str, parent_op: object, slot_idx: int) -> int:
    """child→parent 边的 relation(按父算子角色 + 子槽位)。"""
    if parent_kind == 'unary':
        return 0                                          # unary operand
    if parent_kind == 'binary':
        return 1 if parent_op in _COMM_BIN else (2 if slot_idx == 0 else 3)
    if parent_kind == 'binary_const':
        return 1 if parent_op in _COMM_CONST else (2 if slot_idx == 0 else 3)
    if parent_kind == 'ts':                               # slot0=panel(feature), slot1=window(ts)
        return 4 if slot_idx == 0 else 5
    if parent_kind == 'pair':                             # slot0,1=panel(feature), slot2=window(ts)
        return 4 if slot_idx < 2 else 5
    raise ValueError(f'non-internal parent kind: {parent_kind}')


# ============================================================================
# 序列 ↔ 图拓扑(最左推导重放,纯整数,无 torch)
# ============================================================================

def _replay_graph(prod_ids: List[int]) -> List[Tuple[int, int, int]]:
    """重放最左推导 → edges = [(child_node_idx, parent_node_idx, edge_type)]。
    node_idx = production 在序列中的位置;边方向 child→parent(自底向上聚合)。"""
    edges: List[Tuple[int, int, int]] = []
    stack: List[Tuple[int, int, int]] = [(-1, -1, 0)]    # (parent_node_idx, parent_pid, slot_idx),根槽无父
    for node_idx, pid in enumerate(prod_ids):
        parent_idx, parent_pid, slot_idx = stack.pop()
        if parent_idx >= 0:
            pp = _grammar.PRODUCTIONS[parent_pid]
            edges.append((node_idx, parent_idx, _edge_type(pp.rhs_kind, pp.op, slot_idx)))
        n_child = len(_grammar.PRODUCTIONS[pid].children_nts)
        for i in range(n_child - 1, -1, -1):              # 子槽倒序压栈 → 最左子在栈顶
            stack.append((node_idx, pid, i))
    return edges


# ---- 整树 → production-id 序列(decode 的逆,供 embed_trees) ----
# (key→pid, window→pid) 映射按 id(PRODUCTIONS) 缓存:set_vocabulary 重建 PRODUCTIONS → 新 tuple
# id → 自动失效重算,故不会用旧词表的陈旧 id。
_PID_CACHE: dict = {}


def _prod_id_maps():
    P = _grammar.PRODUCTIONS
    key = id(P)
    hit = _PID_CACHE.get(key)
    if hit is None:
        by_key = {(p.rhs_kind, p.op): pid for pid, p in enumerate(P)}
        by_win = {p.op: pid for pid, p in enumerate(P) if p.rhs_kind == 'window'}
        _PID_CACHE.clear()                              # 只留当前词表一份
        _PID_CACHE[key] = (by_key, by_win)
        hit = _PID_CACHE[key]
    return hit


def _node_to_seq(node: Node) -> List[int]:
    """前序展开节点为 production-id 序列;ts/pair 的 window 在子树后重发 window-leaf(对齐 decode 顺序)。"""
    by_key, by_win = _prod_id_maps()
    seq = [by_key[(node.kind, node.op)]]
    for c in node.children:
        seq += _node_to_seq(c)
    if node.kind in ('ts', 'pair'):
        seq.append(by_win[node.window])
    return seq


# ============================================================================
# R-GCN
# ============================================================================

class _RGCNLayer(nn.Module):
    """R-GCN 单层:self-loop 线性(= RGCNConv root_weight)+ 逐 relation 线性聚合(index_add scatter)→ ReLU。"""
    def __init__(self, in_dim: int, out_dim: int, n_rel: int):
        super().__init__()
        self.self_lin = nn.Linear(in_dim, out_dim)
        self.rel_lin = nn.ModuleList([nn.Linear(in_dim, out_dim, bias=False) for _ in range(n_rel)])

    def forward(self, h: torch.Tensor, edges: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        out = self.self_lin(h)
        for r, (src, dst) in enumerate(edges):
            if src.numel() == 0:
                continue
            msg = self.rel_lin[r](h.index_select(0, src))
            out = out.index_add(0, dst, msg)
        return torch.relu(out)


class RGCNEncoder(nn.Module):
    """AST → (B, hidden) backbone(global mean-pool)。head(nn.Linear)接其后给动作 logits。"""
    def __init__(self, hidden_dim: int, n_layers: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.prod_embed = nn.Embedding(_grammar.NUM_PRODUCTIONS, hidden_dim)   # 构造时按 active 词表定尺寸
        self.layers = nn.ModuleList(
            [_RGCNLayer(hidden_dim, hidden_dim, N_RELATIONS) for _ in range(n_layers)])

    def _build_batch(self, flat: torch.Tensor):
        """(B, L) int → 批图全局张量(节点为各图节点的并集,disjoint-union)。"""
        dev = self.prod_embed.weight.device
        rows = flat.tolist()
        node_prods: List[int] = []
        batch_idx: List[int] = []
        edges_per_rel: List[List[Tuple[int, int]]] = [[] for _ in range(N_RELATIONS)]
        offset = 0
        for b, row in enumerate(rows):
            seq = [int(x) for x in row if x >= 0]
            for pid in seq:
                node_prods.append(pid)
                batch_idx.append(b)
            for child, parent, et in _replay_graph(seq):
                edges_per_rel[et].append((offset + child, offset + parent))   # src=child → dst=parent
            offset += len(seq)
        B = len(rows)

        def _ei(lst: List[Tuple[int, int]]):
            if not lst:
                empty = torch.empty(0, dtype=torch.long, device=dev)
                return empty, empty
            arr = torch.tensor(lst, dtype=torch.long, device=dev)
            return arr[:, 0].contiguous(), arr[:, 1].contiguous()

        edge_index = [_ei(lst) for lst in edges_per_rel]
        node_prods_t = torch.tensor(node_prods, dtype=torch.long, device=dev)
        batch_t = torch.tensor(batch_idx, dtype=torch.long, device=dev)
        return node_prods_t, batch_t, edge_index, B, offset

    def _encode(self, flat: torch.Tensor) -> torch.Tensor:
        """(B, L) int → (B, hidden) global mean-pool 结构嵌入。"""
        dev = self.prod_embed.weight.device
        H = self.hidden_dim
        node_prods, batch_t, edge_index, B, N = self._build_batch(flat)
        if N == 0:                                          # 全空树/sink → 零嵌入(mask 兜底)
            return torch.zeros(B, H, device=dev)
        h = self.prod_embed(node_prods)
        for layer in self.layers:
            h = layer(h, edge_index)
        counts = torch.zeros(B, device=dev).index_add(0, batch_t, torch.ones(N, device=dev))
        pooled = torch.zeros(B, H, device=dev).index_add(0, batch_t, h)
        return pooled / counts.clamp(min=1.0).unsqueeze(1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        lead = state.shape[:-1]
        flat = state.reshape(-1, state.shape[-1])
        out = self._encode(flat)
        return out.reshape(*lead, self.hidden_dim)

    # ---- R_SA 结构签名:整树 → mean-pool embedding(no_grad,给 reward bank) ----
    @torch.no_grad()
    def embed_trees(self, trees: List[AlphaTree]) -> np.ndarray:
        """List[AlphaTree] → (n, hidden) 结构签名(decode 逆 + 同图编码)。"""
        dev = self.prod_embed.weight.device
        seqs = [_node_to_seq(t.root) for t in trees]
        st = torch.full((len(seqs), MAX_DERIV_LEN), -1, dtype=torch.long, device=dev)
        for i, s in enumerate(seqs):
            st[i, :len(s)] = torch.tensor(s, dtype=torch.long, device=dev)
        return self._encode(st).detach().cpu().numpy().astype(np.float64)
