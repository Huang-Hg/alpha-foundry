"""GFlowNet ↔ 我们文法的 codec(AlphaSAGE 独立实现的地基,2026-06-23)。

GFN 在**最左推导序列空间**生成:state = 已选 production-id 序列;action = 给栈顶 slot 选一个
production(由 `grammar.is_legal` mask),栈空 = 推导完成(terminal,接 EXIT → sink)。
每个 AlphaTree 对应唯一的最左推导 → **backward policy P_B trivial**(父 = 去掉最后一个 production),
规避 `ast.py` 交换律规范化导致的多父配平坑(规范化只在 decode 末端做,用于入库去重)。

action 空间 = PRODUCTIONS(NUM_PRODUCTIONS 条)+ 末位 EXIT。
state 张量 = 定长 production-id 序列(padded -1),mask 每步由序列重放出栈顶 slot 的 ctx 算。
"""
from __future__ import annotations

from typing import List

import numpy as np

from evaluation import grammar as _grammar
from evaluation.ast import AlphaTree, make_node
from evaluation.grammar import (
    DEFAULT_MAX_COST, DEFAULT_MAX_LEN, DEFAULT_MIN_COST,
    ExpansionContext, NonTerminal, SemKind, child_context, initial_context, is_legal,
)

# PRODUCTIONS / NUM_PRODUCTIONS 一律**运行时**从 _grammar 读(grammar.set_vocabulary 换市场会重建
# 它们)→ 动作空间随 active 词表自适应,彻底消除"import 顺序固化"静默陷阱。EXIT/动作数同理函数化。

# 窗口 slot 的 expected_out_kinds(关闭它最小成本=0,window leaf cost=0);其余 panel/numeric slot 最小=1
# (operand/const leaf cost=1)。全局成本预算用此算"关闭所有 pending 槽的最小成本"。
_WINDOW_KINDS = frozenset({SemKind.WINDOW_VAL})


def _slot_min_cost(ctx: ExpansionContext) -> int:
    return 0 if ctx.expected_out_kinds == _WINDOW_KINDS else 1

def exit_action() -> int:
    """末位动作 = 退出(完成推导 → sink)= 当前 NUM_PRODUCTIONS(随词表变)。"""
    return _grammar.NUM_PRODUCTIONS


def n_actions() -> int:
    """动作空间大小 = NUM_PRODUCTIONS + 1(EXIT);随 active 词表运行时变。"""
    return _grammar.NUM_PRODUCTIONS + 1


MAX_DERIV_LEN: int = 64                            # 定长 state 张量上限(物理 padding;真长度由 MAX_LEN 卡)
MAX_COST: int = DEFAULT_MAX_COST
MIN_COST: int = DEFAULT_MIN_COST
MAX_LEN: int = DEFAULT_MAX_LEN                     # 推导 production 数上限(含 window leaf;[dsl] 经 grammar 单链)


def _pending_stack(prod_ids: List[int]) -> List[ExpansionContext]:
    """重放最左推导,返回剩余待填 slot 的 ctx 栈(栈顶 stack[-1] = 下一个要填的 slot;空 = 完成)。"""
    stack: List[ExpansionContext] = [initial_context(MAX_COST)]
    for pid in prod_ids:
        ctx = stack.pop()                          # 该 production 填的(最左)slot
        prod = _grammar.PRODUCTIONS[pid]
        n_child = len(prod.children_nts)
        # 子 slot 倒序压栈 → 最左子在栈顶(保持最左推导 = 前序)
        for i in range(n_child - 1, -1, -1):
            stack.append(child_context(prod, i, (), ctx))
    return stack


def legal_mask(prod_ids: List[int]) -> np.ndarray:
    """给定已选 production 序列,返回 [N_ACTIONS] bool:每个 production 是否合法 + EXIT。
    完成(栈空)→ 仅 EXIT;未完成 → 栈顶 slot 的合法 production(EXIT 非法)。

    **全局成本预算**(2026-06-24,治模式坍缩):grammar 的 is_legal 成本前瞻是**逐路径**的(child_context
    每个子槽各自拿 remaining_cost,兄弟子树互不知 → binary(big_a,big_b) 总成本可远超 MAX_COST,既爆 cost-floor
    又让推导无界生长撞 MAX_DERIV_LEN 不完整 → GFN 前向策略坍缩到超长垃圾)。此处叠加全局约束:
    committed(已选 production 成本和)+ prod.cost + 关闭(其余 pending 槽 + prod 新子槽)的最小成本 ≤ MAX_COST。
    admissible(每槽最小成本 = 能用最便宜 leaf 真关闭:panel/numeric=1、window=0)→ 不会把可完成的树剪掉、
    也不会画进死角(committed 逼近 25 却剩 pending 槽无合法 leaf)。物理上界死总成本 → 长度有界、不完整消失。"""
    mask = np.zeros(n_actions(), dtype=bool)
    stack = _pending_stack(prod_ids)
    if not stack:                                  # 推导完成 → 只能 EXIT
        mask[exit_action()] = True
        return mask
    committed = sum(_grammar.PRODUCTIONS[pid].cost for pid in prod_ids)
    other_min = sum(_slot_min_cost(c) for c in stack[:-1])   # 关闭其余 pending 槽的最小成本
    # 长度预算(防过拟合长树):每个 pending 槽 + prod 每个子槽都需 ≥1 production 关闭(window leaf 也算 1)。
    cur_len = len(prod_ids)
    n_other = len(stack) - 1                                  # 除当前槽外的待填槽数(各需 ≥1 production)
    ctx = stack[-1]
    for i, prod in enumerate(_grammar.PRODUCTIONS):
        if not is_legal(prod, ctx):
            continue
        n_children = len(prod.children_nts)
        # 成本预算:committed + prod 成本 + 关闭所有剩余槽的最小成本 ≤ MAX_COST
        if committed + prod.cost + other_min + sum(1 for nt in prod.children_nts if nt != NonTerminal.WINDOW) > MAX_COST:
            continue
        # 长度预算:最终 production 数下界 = 已选 + prod 自身 + 关闭(其余 pending + prod 新子槽)各 1 ≤ MAX_LEN
        if cur_len + 1 + n_other + n_children > MAX_LEN:
            continue
        mask[i] = True
    return mask


def is_complete(prod_ids: List[int]) -> bool:
    return len(_pending_stack(prod_ids)) == 0


def decode(prod_ids: List[int]) -> AlphaTree:
    """完成的 production 序列(前序)→ AlphaTree。window/const 作 leaf 经 make_node 收进父节点。"""
    idx = [0]

    def build():
        prod = _grammar.PRODUCTIONS[prod_ids[idx[0]]]
        idx[0] += 1
        children = tuple(build() for _ in range(len(prod.children_nts)))
        return make_node(prod, children)

    return AlphaTree(build())


def cost_ok(tree: AlphaTree) -> bool:
    return MIN_COST <= tree.total_cost() <= MAX_COST
