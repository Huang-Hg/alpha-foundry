"""FastExpr —— alpha 表达式字符串 ↔ 可执行 AlphaTree(对标 BRAIN 平台的 FastExpr + 本地校验器)。

子 agent 只写字符串,本模块负责:词法/语法解析、算子名与元数数校验、operand 是否在词表、
窗口是否在白名单、常数是否在常数池、cost 预算。**报错必须精确到位置与原因** ——
agent 拿到的错误信息要足以自行改对,不需要读源码。

语法(= `AlphaTree.pretty()` 的逆):
    operand              btc_fees
    unary(1)             sign(x)  abs neg log square sqrt tanh inv s_log_1p
    binary(2)            add(a,b) sub mul div max min gt lt
    binary_const(x,k)    add_const(x, 0.5)  mul_const(x, 2)  pow_const(x, -0.5)
    ts(x,w)              ts_mean(x, 120)  …共 20 个
    pair(a,b,w)          ts_corr(a, b, 120)  ts_cov(a, b, 120)

用法:
    from fastexpr import parse, validate
    tree = parse('sign(ts_slope(btc_feepertx, 365))')     # 失败抛 ExprError(含位置)
    v = validate('...', panels)                            # → dict(ok, error, cost, fields, windows)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.ast import AlphaTree, Node
from evaluation.grammar import (
    ADD_CONSTANTS, MULPOW_CONSTANTS, BinaryOp, ConstBinaryOp, LogicalBinaryOp,
    LogicalUnaryOp, PairOp, PairParamOp, ParamOp, TernaryOp, TsInjOp, TsOp, TsParamOp,
    UnaryOp, ACTIVE_WINDOWS,
)

MAX_COST = 25          # [quant] config.ini [dsl].max_cost

# ---------------------------------------------------------------- 可变参数策略
# 窗口白名单 ACTIVE_WINDOWS 与常数池 ADD/MULPOW_CONSTANTS 是**为 GP/GFN 搜索空间**设的:
# 离散化才能枚举产生式、才能做 cost 前瞻剪枝。**LLM 写 alpha 不按那种方式搜** —— 它一次
# 提出少量有经济学动机的表达式,离散化不换来任何搜索收益,只白扔表达力(用户 2026-08-13 裁定)。
# ⇒ 本车道放开为连续参数,只保留物理边界。原白名单降级为**推荐网格**,仅供生成端参考
# (它此前的唯一消费者是 L3 电池的窗口邻域构造,该层已于 2026-08-13 整层删除)。
FREE_PARAMS = True
MIN_WINDOW, MAX_WINDOW = 2, 1500     # 上界 ≈ 面板行数量级;下界 2(w=1 使多数 ts 退化为恒等)

_UNARY = {o.value: o for o in UnaryOp}
_LOGUNARY = {o.value: o for o in LogicalUnaryOp}
_BINARY = {o.value: o for o in BinaryOp}
_LOGBINARY = {o.value: o for o in LogicalBinaryOp}
_BINCONST = {o.value: o for o in ConstBinaryOp}
_PARAM = {o.value: o for o in ParamOp}          # panel ⊕ 连续标量,不受常数池约束
_TERNARY = {o.value: o for o in TernaryOp}
_TS = {o.value: o for o in TsOp}
_TSINJ = {o.value: o for o in TsInjOp}          # 同 ts 槽型,注入层补齐
_TSPARAM = {o.value: o for o in TsParamOp}      # panel + window + 连续标量
_PAIR = {o.value: o for o in PairOp}
_PAIRPARAM = {o.value: o for o in PairParamOp}  # panel + panel + window + 连续标量
ALL_OPS = {**_UNARY, **_LOGUNARY, **_BINARY, **_LOGBINARY, **_BINCONST, **_PARAM,
           **_TERNARY, **_TS, **_TSINJ, **_TSPARAM, **_PAIR, **_PAIRPARAM}

ARITY = {**{k: 1 for k in _UNARY}, **{k: 1 for k in _LOGUNARY},
         **{k: 2 for k in _BINARY}, **{k: 2 for k in _LOGBINARY},
         **{k: 2 for k in _BINCONST}, **{k: 2 for k in _PARAM},
         **{k: 3 for k in _TERNARY}, **{k: 2 for k in _TS}, **{k: 2 for k in _TSINJ},
         **{k: 3 for k in _TSPARAM}, **{k: 3 for k in _PAIR},
         **{k: 4 for k in _PAIRPARAM}}


class ExprError(ValueError):
    """解析/校验失败。message 含出错位置与可执行的修复建议。"""


_TOK = re.compile(r'\s*(?:(?P<num>-?\d+\.?\d*)|(?P<id>[A-Za-z_][A-Za-z_0-9]*)|(?P<p>[(),]))')


def _tokenize(s: str):
    out, i = [], 0
    while i < len(s):
        m = _TOK.match(s, i)
        if not m:
            if s[i].isspace():
                i += 1
                continue
            raise ExprError(f'位置 {i}:无法识别的字符 {s[i]!r}\n  {s}\n  {" " * i}^')
        kind = m.lastgroup
        out.append((kind, m.group(kind), m.start(kind)))
        i = m.end()
    return out


class _P:
    def __init__(self, s, operands):
        self.s = s
        self.t = _tokenize(s)
        self.i = 0
        self.operands = operands

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else (None, None, len(self.s))

    def eat(self, val=None):
        k, v, p = self.peek()
        if k is None:
            raise ExprError(f'表达式在位置 {p} 意外结束(缺 {val or "内容"})\n  {self.s}')
        if val is not None and v != val:
            raise ExprError(f'位置 {p}:期望 {val!r},实际 {v!r}\n  {self.s}\n  {" " * p}^')
        self.i += 1
        return v, p

    def parse(self):
        n = self.expr()
        if self.i != len(self.t):
            _, v, p = self.peek()
            raise ExprError(f'位置 {p}:表达式结束后有多余内容 {v!r}\n  {self.s}\n  {" " * p}^')
        return n

    def expr(self) -> Node:
        k, v, p = self.peek()
        if k == 'num':
            raise ExprError(f'位置 {p}:{v} —— 裸常数不能作节点。常数只能作 add_const/'
                            f'mul_const/pow_const 的第二个参数,或 ts_*/pair 的窗口\n  {self.s}')
        if k != 'id':
            raise ExprError(f'位置 {p}:期望算子名或字段名,实际 {v!r}\n  {self.s}\n  {" " * p}^')
        self.eat()
        nk, nv, np_ = self.peek()
        if nv != '(':                                     # 叶子 = operand
            if v not in self.operands:
                near = [o for o in self.operands if o.startswith(v[:3])][:5]
                raise ExprError(f'位置 {p}:未知字段 {v!r}(不在当前词表 {len(self.operands)} 个 '
                                f'operand 中)。相近: {near or "无"}\n  {self.s}')
            return Node.make('operand', v, None, ())
        if v not in ALL_OPS:
            near = [o for o in ALL_OPS if o.startswith(v[:3])][:5]
            raise ExprError(f'位置 {p}:未知算子 {v!r}。相近: {near or "无"}\n  {self.s}')
        self.eat('(')
        args, wins = [], []
        while True:
            k2, v2, p2 = self.peek()
            if k2 == 'num':
                self.eat()
                wins.append((float(v2), p2))
                args.append(None)
            else:
                args.append(self.expr())
                wins.append(None)
            k3, v3, _ = self.peek()
            if v3 == ',':
                self.eat(',')
                continue
            self.eat(')')
            break
        return self._build(v, args, wins, p)

    def _build(self, op, args, wins, p) -> Node:
        want = ARITY[op]
        if len(args) != want:
            raise ExprError(f'位置 {p}:{op} 需要 {want} 个参数,给了 {len(args)}\n  {self.s}')
        if op in _UNARY or op in _LOGUNARY:
            self._no_num(op, wins, 0, p)          # sign(0.5) 之类退化为常数,结构性禁止
            return Node.make('unary', (_UNARY.get(op) or _LOGUNARY[op]), None, (args[0],))
        if op in _BINARY or op in _LOGBINARY:
            return Node.make('binary', (_BINARY.get(op) or _LOGBINARY[op]), None,
                             (self._panel(args, wins, 0), self._panel(args, wins, 1)))
        if op in _PARAM:                       # panel ⊕ 连续标量:不查常数池
            self._no_num(op, wins, 0, p)
            if wins[1] is None:
                raise ExprError(f'位置 {p}:{op} 第 2 个参数必须是标量'
                                f'(阈值/幅度,连续取值)\n  {self.s}')
            return Node.make('binary_const', _PARAM[op], None,
                             (args[0], Node.make('const_add', wins[1][0], None, ())))
        if op in _BINCONST:
            self._no_num(op, wins, 0, p)
            if wins[1] is None:
                raise ExprError(f'位置 {p}:{op} 第 2 个参数必须是常数\n  {self.s}')
            k = wins[1][0]
            pool = ADD_CONSTANTS if op == 'add_const' else MULPOW_CONSTANTS
            if not FREE_PARAMS and k not in pool:
                raise ExprError(f'位置 {wins[1][1]}:{op} 常数 {k:g} 不在常数池 '
                                f'{sorted(pool)}\n  {self.s}')
            if k == 0.0 and op != 'add_const':
                raise ExprError(f'位置 {wins[1][1]}:{op} 常数 0 使整棵子树退化'
                                f'(mul_const→恒 0,pow_const→恒 1)\n  {self.s}')
            leaf = 'const_add' if op == 'add_const' else 'const_mul'
            return Node.make('binary_const', _BINCONST[op], None,
                             (args[0], Node.make(leaf, k, None, ())))
        if op in _TERNARY:
            return Node.make('ternary', _TERNARY[op], None,
                             tuple(self._panel(args, wins, i) for i in range(3)))
        if op in _TS or op in _TSINJ:
            self._no_num(op, wins, 0, p)
            return Node.make('ts', (_TS.get(op) or _TSINJ[op]),
                             self._win(op, wins, 1, p), (args[0],))
        if op in _TSPARAM:                     # f(x, w, k):标量在窗口之后,同 BRAIN 参数序
            self._no_num(op, wins, 0, p)
            if wins[2] is None:
                raise ExprError(f'位置 {p}:{op} 第 3 个参数必须是标量'
                                f'(BRAIN 里的 factor= / constant= / driver=)\n  {self.s}')
            return Node.make('ts_param', _TSPARAM[op], self._win(op, wins, 1, p),
                             (args[0], Node.make('const_add', wins[2][0], None, ())))
        if op in _PAIRPARAM:                   # f(y, x, w, k)
            self._no_num(op, wins, 0, p)
            self._no_num(op, wins, 1, p)
            if wins[3] is None:
                raise ExprError(f'位置 {p}:{op} 第 4 个参数必须是标量'
                                f'(rettype:0=残差 1=截距 2=斜率 3=拟合 4=R²)\n  {self.s}')
            return Node.make('pair_param', _PAIRPARAM[op], self._win(op, wins, 2, p),
                             (args[0], args[1],
                              Node.make('const_add', wins[3][0], None, ())))
        self._no_num(op, wins, 0, p)
        self._no_num(op, wins, 1, p)
        return Node.make('pair', _PAIR[op], self._win(op, wins, 2, p), (args[0], args[1]))

    def _panel(self, args, wins, idx) -> Node:
        """panel 槽:子表达式原样返回;**裸标量自动包成常数叶**(整面板广播)。

        `gt(rangepos, 0.5)` / `if_else(gt(x,0), a, 1)` 是 LLM 写门控最自然的写法,
        而 GP 侧禁止裸常数是为了不浪费搜索空间(常数叶只能挂在 binary_const 上)——
        注入层不搜索,禁它只是逼人写 `gt(x, add_const(mul_const(x,0), 0.5))` 这种噪音。
        求值器两侧本就支持常数叶(`expression._eval_block` 填块 / `gpu.eval_node` cp.full),
        零新增语义。纯常数树仍由 `validate()` 的「至少一个 operand」拦下。"""
        if wins[idx] is None:
            return args[idx]
        return Node.make('const_add', wins[idx][0], None, ())

    def _no_num(self, op, wins, idx, p):
        if wins[idx] is not None:
            raise ExprError(f'位置 {wins[idx][1]}:{op} 第 {idx + 1} 个参数必须是子表达式,'
                            f'不能是常数 {wins[idx][0]:g}\n  {self.s}')

    def _win(self, op, wins, idx, p) -> int:
        if wins[idx] is None:
            raise ExprError(f'位置 {p}:{op} 最后一个参数必须是窗口整数,'
                            f'合法值 {list(ACTIVE_WINDOWS)}\n  {self.s}')
        w, wp = wins[idx]
        if w != int(w):
            raise ExprError(f'位置 {wp}:窗口 {w:g} 必须是整数\n  {self.s}')
        w = int(w)
        if not FREE_PARAMS:
            if w not in ACTIVE_WINDOWS:
                raise ExprError(f'位置 {wp}:窗口 {w} 不在白名单 {list(ACTIVE_WINDOWS)}\n  {self.s}')
        elif not (MIN_WINDOW <= w <= MAX_WINDOW):
            raise ExprError(f'位置 {wp}:窗口 {w} 越界 [{MIN_WINDOW}, {MAX_WINDOW}]'
                            f'(w=1 使多数 ts 算子退化为恒等;上界约面板行数量级)\n  {self.s}')
        return w


def parse(s: str, operands=None) -> AlphaTree:
    """表达式字符串 → AlphaTree。operands 缺省用当前面板词表(需先 engine.load())。"""
    if operands is None:
        import engine
        operands = set(engine.operands())
    return AlphaTree(_P(s.strip(), set(operands)).parse())


def validate(s: str, operands=None) -> Dict:
    """→ dict(ok, error, cost, n_ops, fields, windows, pretty, tree)。不抛异常,供批量预筛。
    `tree` = 已解析的 AlphaTree:解析 0.13 ms/条,批量路径上重复 parse 与 A800 的 GPU 段同量级。"""
    try:
        t = parse(s, operands)
    except ExprError as e:
        return dict(ok=False, error=str(e))
    fields, wins, ops = set(), set(), 0

    def walk(n):
        nonlocal ops
        if n.kind == 'operand':
            fields.add(n.op)
        elif n.kind not in ('const_add', 'const_mul'):
            ops += 1
        if n.window is not None:
            wins.add(n.window)
        for c in n.children:
            walk(c)

    walk(t.root)
    cost = t.total_cost()
    if cost > MAX_COST:
        return dict(ok=False, error=f'cost {cost} > 上限 {MAX_COST}(config.ini [dsl].max_cost)。'
                                    f'简化表达式或减少嵌套层数', cost=cost)
    if not fields:
        return dict(ok=False, error='表达式不含任何 operand(纯常数树),evaluator G0 会结构拒')
    return dict(ok=True, error=None, cost=cost, n_ops=ops, fields=sorted(fields),
                windows=sorted(wins), pretty=t.pretty(), tree=t)
