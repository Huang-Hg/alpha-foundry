"""DEAP 强类型 GP 因子挖掘 baseline(2026-06-22)。

目的:公式化 alpha 搜索 —— 评估侧全复用共享管线(同 panel/y、同 `evaluate_alpha` standalone 净
Sharpe reward + G0/G1/G2 门、同 `alpha_pool.try_new` net 感知准入、同 val/test pool_ret 口径),
搜索器 = DEAP NSGA-II 强类型 GP。

强类型(STGP):DEAP `PrimitiveSetTyped`,4 个类型 {Panel, WinT, AddC, MulC}——panel 与
窗口/两类常数分型,结构合法(杜绝 `tanh(0.5)`/`ts_min(3,w)`),与文法句法槽型一一对应
(文法语义 forbid 层 2026-07-22 A/B 证否删除 → GP/GFN 搜索空间自此完全同构)。
cs 算子族 2026-07-22 随截面口径退役(时序书转向,见 evaluator/grammar)。

搜索空间预算与 GFN 同源同刻度([dsl] 唯一来源,2026-07-03 统一):cost ∈ [min_cost, max_cost],
长度 ≤ max_len。DEAP prefix 每元素(primitive/operand/window/const terminal)≡ 文法一条 production
→ `len(ind)` = 推导长度,与 gfn_codec.MAX_LEN 同刻度;初始化超长重采、mate/mutate 超长回退
(staticLimit)。max_len=5 实测极限勿调大(长树 = 多重比较选择偏差,同 reward=flat 证否机制);
旧 [gp_baseline] init_*_height / max_height 三旋钮已删(init 高度 ramped 1..max_len−1 推导得出)。
primitive/terminal 一一镜像 DSL 算子(unary/binary/binary_const/ts/pair
+ operands/WINDOWS/ADD/MULPOW consts)。

fitness = NSGA-II 双目标(均最大化;2026-07-22 截面→时序转向):
  obj1 = R4 滚动折中位 edge(= evaluate_alpha['cv_edge'],2026-07-25 与 GFN 同刀:
         5 折 ~18mo 折 edge 中位数,时间轴尖峰不给分;带符号保留选择梯度;collect/入池
         另须 >0 ⇔ ≥3/5 折胜锚。谱系:4 折一致性**门** 2026-07-22 实证击穿删除 ——
         唯一 4/4 折成员 val 最差;R4 换的是取出统计量本身而非后置门,勿混)
  obj2 = 行为多样性 = 1 − max_{j≠i}|pnl_corr(ex_i, ex_j)|(种群内,ex = rr_book − rr_ew 超额流)
无效树(cost 越界 / 门拒 / per_t_pnl=None)→ (_REWARD_FLOOR, 0) 被支配。

评估恒 CPU(日线 panel 极小,单候选 C 内核毫秒级)。

桥:DEAP PrimitiveTree(prefix)→ Node.make → AlphaTree。ts/pair 的 window、binary_const 的 const
都是末位 terminal,按类型解析进父节点(const → const_add/const_mul 叶,window → Node.window attr)。
"""
from __future__ import annotations

import json
import random
from inspect import isclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from deap import algorithms, base, creator, gp, tools

from config.config import ini
from backtest import ops
from evaluation.cache import EvalCache
from evaluation import grammar as _grammar
from evaluation.ast import AlphaTree, Node
from evaluation.grammar import (
    ADD_CONSTANTS, MULPOW_CONSTANTS,
    BinaryOp, ConstBinaryOp, PairOp, TsOp, UnaryOp,
)
from rl.alpha_pool import AlphaPool
from rl.evaluator import EvalConfig, _REWARD_FLOOR, evaluate_alpha, qnt_bench_sharpe


# ============================================================================
# config([gp_baseline];cost / 长度预算复用 [dsl],与 GFN 同源)
# ============================================================================
_POP        = int(ini('gp_baseline', 'pop_size', 2000))
_NGEN       = int(ini('gp_baseline', 'n_gen', 40))
_CXPB       = ini('gp_baseline', 'cxpb', 0.7)
_MUTPB      = ini('gp_baseline', 'mutpb', 0.2)
# [dsl] 预算经 grammar 常量单链同源(ini → grammar → GFN/GP;勿再直读 ini 造第二读点)
_MAX_LEN    = _grammar.DEFAULT_MAX_LEN        # 推导 production 数上限(=len(prefix);= gfn_codec.MAX_LEN)
_MAX_COST   = _grammar.DEFAULT_MAX_COST
_MIN_COST   = _grammar.DEFAULT_MIN_COST


# ============================================================================
# 强类型 primitive set + DEAP→AlphaTree 桥映射
# ============================================================================
class Panel: pass      # 主因子类型((T,S) panel);所有算子输出 + operand 输入
class WinT:  pass      # 窗口整数(ts/pair 末槽)
class AddC:  pass      # add_const 常数槽(ADD_CONSTANTS)
class MulC:  pass      # mul_const / pow_const 常数槽(MULPOW_CONSTANTS)


def _cname(c: float) -> str:
    return ('n' if c < 0 else '') + str(abs(c)).replace('.', 'p')


def _build_pset() -> Tuple[gp.PrimitiveSetTyped, Dict[str, tuple], Dict[str, tuple]]:
    pset = gp.PrimitiveSetTyped("alpha", [], Panel)
    prim_map: Dict[str, tuple] = {}    # primitive .name → (kind, op)
    term_map: Dict[str, tuple] = {}    # terminal  .name → (tag, value)
    _id = lambda *a: None              # 占位 func:桥手工解析,gp.compile 不用

    for op in UnaryOp:
        nm = 'u_' + op.value
        pset.addPrimitive(_id, [Panel], Panel, name=nm); prim_map[nm] = ('unary', op)
    for op in BinaryOp:
        nm = 'b_' + op.value
        pset.addPrimitive(_id, [Panel, Panel], Panel, name=nm); prim_map[nm] = ('binary', op)
    for op in ConstBinaryOp:
        ct = AddC if op == ConstBinaryOp.ADD_CONST else MulC
        nm = 'bc_' + op.value
        pset.addPrimitive(_id, [Panel, ct], Panel, name=nm); prim_map[nm] = ('binary_const', op)
    for op in TsOp:
        nm = 't_' + op.value
        pset.addPrimitive(_id, [Panel, WinT], Panel, name=nm); prim_map[nm] = ('ts', op)
    for op in PairOp:
        nm = 'p_' + op.value
        pset.addPrimitive(_id, [Panel, Panel, WinT], Panel, name=nm); prim_map[nm] = ('pair', op)

    for _op in _grammar.ACTIVE_VOCAB.operands:
        nm = 'op_' + _op.name
        pset.addTerminal(_op.name, Panel, name=nm); term_map[nm] = ('operand', _op.name)
    for w in _grammar.ACTIVE_WINDOWS:                  # 运行时读 → set_windows(日线集)经 rebuild_pset 生效
        nm = 'w%d' % w
        pset.addTerminal(w, WinT, name=nm); term_map[nm] = ('window', int(w))
    for c in ADD_CONSTANTS:
        nm = 'ac_' + _cname(c)
        pset.addTerminal(c, AddC, name=nm); term_map[nm] = ('const_add', float(c))
    for c in MULPOW_CONSTANTS:
        nm = 'mc_' + _cname(c)
        pset.addTerminal(c, MulC, name=nm); term_map[nm] = ('const_mul', float(c))
    return pset, prim_map, term_map


_PSET, _PRIM_MAP, _TERM_MAP = _build_pset()


def rebuild_pset() -> None:
    """按当前 `_grammar.ACTIVE_VOCAB` + `ACTIVE_WINDOWS` 重建模块级 _PSET/_PRIM_MAP/_TERM_MAP
    (operand 叶 + window 叶集)。operator(unary/binary/ts/pair)+ const 不随市场变。
    run_gp 入口调用 → 消除『import gp_baseline(冻结 crypto 词表/1h 窗)早于 set_vocabulary/set_windows
    (equity 词表/日线窗)』的 import 顺序陷阱(_generate/gen_expr/deap_to_alphatree 均运行时读 _PSET)。"""
    global _PSET, _PRIM_MAP, _TERM_MAP
    _PSET, _PRIM_MAP, _TERM_MAP = _build_pset()


# ---- typed 树生成(DEAP genFull/HalfAndHalf 对 terminal-only 类型 WinT/AddC/MulC 会崩:
#      非叶深度强求 primitive 而该类型无 primitive。这里复刻 DEAP generate + "无 primitive → 回退
#      terminal" 的标准修法,half-and-half(一半 full 一半 grow))----
def _generate(min_: int, max_: int, condition, type_=None) -> list:
    if type_ is None:
        type_ = _PSET.ret
    expr, height = [], random.randint(min_, max_)
    stack = [(0, type_)]
    while stack:
        depth, t = stack.pop()
        if condition(height, depth) or not _PSET.primitives[t]:    # 叶 / terminal-only 类型 → terminal
            term = random.choice(_PSET.terminals[t])
            if isclass(term):
                term = term()
            expr.append(term)
        else:
            prim = random.choice(_PSET.primitives[t])
            expr.append(prim)
            for arg in reversed(prim.args):
                stack.append((depth + 1, arg))
    return expr


def gen_expr(min_: int, max_: int, pset=None, type_=None) -> list:
    """half-and-half typed 生成;pset/type_ 由 mutUniform 以 kwargs 传入(此处 pset 固定 _PSET)。"""
    if random.random() < 0.5:
        cond = lambda h, d: d == h                                       # full
    else:
        cond = lambda h, d: d == h or (d >= min_ and random.random() < 0.35)   # grow
    return _generate(min_, max_, cond, type_)


def gen_expr_capped() -> list:
    """初始化生成:ramped 高度 1..(max_len−1)(len≤5 时最深形态=一元链,高度 = max_len−1),
    超长(len>max_len)重采 —— 初始种群整体落在与 GFN 相同的推导长度空间。"""
    while True:
        e = gen_expr(1, _MAX_LEN - 1)
        if len(e) <= _MAX_LEN:
            return e


def deap_to_alphatree(ind) -> AlphaTree:
    """DEAP PrimitiveTree(prefix 序列)→ AlphaTree。递归消费 prefix:panel 子树递归建 Node,
    ts/pair 末位 WinT terminal 收进 Node.window,binary_const 末位 const terminal 建 const_add/mul 叶。"""
    idx = [0]

    def build() -> Node:
        el = ind[idx[0]]; idx[0] += 1
        if isinstance(el, gp.Primitive):
            kind, op = _PRIM_MAP[el.name]
            if kind == 'unary':
                return Node.make(kind, op, None, (build(),))
            if kind == 'binary':
                l = build(); r = build()
                return Node.make('binary', op, None, (l, r))
            if kind == 'binary_const':
                l = build()
                cel = ind[idx[0]]; idx[0] += 1            # AddC/MulC terminal
                ckind, cval = _TERM_MAP[cel.name]         # 'const_add' | 'const_mul'
                const_leaf = Node.make(ckind, cval, None, ())
                return Node.make('binary_const', op, None, (l, const_leaf))
            if kind == 'ts':
                c = build()
                wel = ind[idx[0]]; idx[0] += 1            # WinT terminal
                return Node.make('ts', op, _TERM_MAP[wel.name][1], (c,))
            if kind == 'pair':
                l = build(); r = build()
                wel = ind[idx[0]]; idx[0] += 1
                return Node.make('pair', op, _TERM_MAP[wel.name][1], (l, r))
            raise ValueError(f'unknown primitive kind {kind}')
        # Terminal:此路径只可能是 operand(window/const 由父 primitive 直接消费)
        tag, val = _TERM_MAP[el.name]
        return Node.make('operand', val, None, ())

    return AlphaTree(build())


# ============================================================================
# DEAP creator(进程内一次)
# ============================================================================
creator.create("FitnessGP", base.Fitness, weights=(1.0, 1.0))   # 双目标均最大化
creator.create("IndividualGP", gp.PrimitiveTree, fitness=creator.FitnessGP)


# ============================================================================
# 评估:bridge → evaluate_alpha → (净 Sharpe, per_t_pnl);种群内算多样性 obj2
# ============================================================================
def _eval_and_assign(individuals, panels, cfg, cache, collected: Dict) -> int:
    """逐个体评估并写 fitness(obj1 时序书 edge, obj2 种群内行为多样性)。edge 为正的候选
    累进 collected(按 tree.hash 取最大 edge,供逐代建池)。返回有效候选数。"""
    results: List[Optional[tuple]] = []
    for ind in individuals:
        tree = deap_to_alphatree(ind)
        cost = tree.total_cost()
        if not (_MIN_COST <= cost <= _MAX_COST):          # DSL cost 预算
            results.append(None); continue
        m = evaluate_alpha(tree, panels, cfg, cache)
        ptp, q = m['per_t_pnl'], m['cv_edge']             # R4 滚动折中位 edge(07-25 与 GFN 同刀,见 evaluator)
        if ptp is None or not np.isfinite(q):             # 门拒 / below-gate(per_t_pnl 仍供 obj2 + 准入 R²)
            results.append(None); continue
        rec = (tree, float(q), ptp)
        results.append(rec)
        h = tree.hash
        # collect 门:edge>0(折一致性门 2026-07-22 实证击穿删除)
        if q > 0.0 and (h not in collected or q > collected[h][1]):
            collected[h] = rec

    valid = [i for i, r in enumerate(results) if r is not None]
    M = np.ascontiguousarray(np.stack([results[i][2] for i in valid])) if valid else None
    for li, i in enumerate(valid):
        corr = np.abs(ops.pnl_corr_vec(np.ascontiguousarray(results[i][2]), M))
        corr[li] = 0.0                                    # 排除自相关(=1)
        nov = 1.0 - float(corr.max()) if M.shape[0] > 1 else 1.0
        individuals[i].fitness.values = (results[i][1], nov)
    for i, r in enumerate(results):
        if r is None:
            individuals[i].fitness.values = (_REWARD_FLOOR, 0.0)
    return len(valid)


def _pool_offer_fresh(pool, collected: Dict, offered: set) -> None:
    """把本代新增 collected 候选按超额 IR 降序过准入(try_new;近克隆门 + Δ=q·(1−R²) 正交门)
    增量入池。2026-07-03 起逐代增量建池(与 alphasage 边搜边建同型,中断即有池;旧终局全排序重建已删)。
    同 hash 只 offer 一次(hash 定树 → q 恒同);后代更强候选靠 AlphaPool prune/evict 汰换。"""
    fresh = [(h, collected[h]) for h in collected if h not in offered]
    for h, (tree, q, ptp) in sorted(fresh, key=lambda x: -x[1][1]):
        pool.try_new(tree, ptp, q)
        offered.add(h)


# ============================================================================
# NSGA-II 主循环
# ============================================================================
def run_gp(panels, capacity: int, seed: int = 42,
           pop_size: Optional[int] = None, n_gen: Optional[int] = None, log=print,
           snap_path: Optional[str] = None):
    """跑 DEAP NSGA-II 强类型 GP,返回 (pool, collected)。pop_size 须被 4 整除(selTournamentDCD)。
    池逐代增量构建(_pool_offer_fresh),snap_path 给定时每代覆盖写 {'gen','pool'} 快照 → 中断可恢复。"""
    pop_size = pop_size or _POP
    n_gen = n_gen if n_gen is not None else _NGEN
    if pop_size % 4 != 0:
        pop_size += 4 - pop_size % 4
    random.seed(seed); np.random.seed(seed)

    # 运行时按当前 ACTIVE_VOCAB 重建 _PSET(operand 叶集),消除 import 顺序陷阱(gp-equity 必需:
    # set_vocabulary(equity) 在 import gp_baseline 之后才执行)。再前置断言词表 ⊆ panels,vocab/panels
    # 不一致即大声崩(否则深处 eval 才 KeyError,难定位)。
    rebuild_pset()
    _vocab_ops = {v for (t, v) in _TERM_MAP.values() if t == 'operand'}
    assert _vocab_ops <= set(panels.keys()), \
        f'GP operand 词表 {sorted(_vocab_ops - set(panels.keys()))} 不在 panels(keys={sorted(panels)});' \
        f'set_vocabulary 与 panels 不一致(import 顺序或市场切换出错)'

    tb = base.Toolbox()
    tb.register("expr", gen_expr_capped)
    tb.register("individual", tools.initIterate, creator.IndividualGP, tb.expr)
    tb.register("population", tools.initRepeat, list, tb.individual)
    tb.register("mate", gp.cxOnePoint)
    tb.register("expr_mut", gen_expr, 0, 2)                   # mutUniform 以 (pset=, type_=) kwargs 调
    tb.register("mutate", gp.mutUniform, expr=tb.expr_mut, pset=_PSET)
    # 长度预算与 GFN 同刻度:len(prefix)=production 数;超长子代回退父本(staticLimit)
    tb.decorate("mate", gp.staticLimit(len, _MAX_LEN))
    tb.decorate("mutate", gp.staticLimit(len, _MAX_LEN))
    tb.register("select", tools.selNSGA2)

    eval_pool = AlphaPool(capacity=capacity)
    cfg = EvalConfig()
    cache = EvalCache(max_bytes=512 * 1024 ** 2)
    collected: Dict[int, tuple] = {}

    offered: set = set()

    def _pool_gen(gen: int) -> None:
        _pool_offer_fresh(eval_pool, collected, offered)
        if snap_path:
            with open(snap_path, 'w', encoding='utf-8') as f:
                f.write(json.dumps({'gen': gen, 'pool': eval_pool.to_jsonable()},
                                   ensure_ascii=False) + '\n')

    bench = qnt_bench_sharpe()          # 日志绝对 Sharpe 口径(内部 reward=edge,读数=edge+锚)
    pop = tb.population(n=pop_size)
    nv = _eval_and_assign(pop, panels, cfg, cache, collected)
    pop = tb.select(pop, pop_size)                            # 赋 NSGA-II crowding
    _pool_gen(0)
    log(f"[gp] gen00 valid={nv}/{pop_size} collected={len(collected)} pool={eval_pool.size} "
        f"(EW 锚 train Sharpe {bench:+.4f})")

    for gen in range(1, n_gen + 1):
        offspring = tools.selTournamentDCD(pop, pop_size)
        offspring = algorithms.varAnd(offspring, tb, _CXPB, _MUTPB)
        nv = _eval_and_assign(offspring, panels, cfg, cache, collected)
        pop = tb.select(pop + offspring, pop_size)
        _pool_gen(gen)
        best = max((c[1] for c in collected.values()), default=float('nan'))
        log(f"[gp] gen{gen:02d} valid={nv} collected={len(collected)} pool={eval_pool.size} "
            f"best_sharpe={best + bench:+.4f}")

    log(f"[gp] done: collected={len(collected)} pool_size={eval_pool.size}/{capacity}")
    return eval_pool, collected
