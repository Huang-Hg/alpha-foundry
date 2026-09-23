"""FastExpr: formula strings <-> trees, with a validator whose messages are exact enough to fix a formula without
reading source code (they are what an LLM author reads back).

Grammar = the inverse of `Tree.pretty()`:
    field                         btc_fees
    op(args...)                   every operator of foundry.ops.registry; windows are integer literals, the scalar
                                  of a binary_const / ts_param / pair_param operator is a numeric literal
A bare number in a panel slot becomes a constant panel (`gt(x, 0.5)`); a formula must reference at least one field.
Parameters are continuous: windows only need MIN_WINDOW..MAX_WINDOW, scalars are free (the discrete ladders belong
to the search grammar, not to hand-written formulas).
"""
from __future__ import annotations

import re
from typing import Dict, Iterable

from foundry.config import ExprConfig
from foundry.expr.ast import Node, Tree
from foundry.ops.registry import ARITY, MAX_WINDOW, MIN_WINDOW, OPS

MAX_COST = ExprConfig().max_cost


class ExprError(ValueError):
    """Parse or validation failure; the message carries the position and the fix."""


TOKEN = re.compile(r'\s*(?:(?P<num>-?\d+\.?\d*(?:[eE][-+]?\d+)?)|(?P<id>[A-Za-z_][A-Za-z_0-9]*)|(?P<p>[(),]))')


def tokens(s: str):
    out, i = [], 0
    while i < len(s):
        m = TOKEN.match(s, i)
        if not m:
            if s[i].isspace():
                i += 1
                continue
            raise ExprError(f'位置 {i}:无法识别的字符 {s[i]!r}\n  {s}\n  {" " * i}^')
        out.append((m.lastgroup, m.group(m.lastgroup), m.start(m.lastgroup)))
        i = m.end()
    return out


class _Parser:
    def __init__(self, s: str, fields):
        self.s, self.t, self.i, self.fields = s, tokens(s), 0, fields

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else (None, None, len(self.s))

    def eat(self, want=None):
        k, v, p = self.peek()
        if k is None:
            raise ExprError(f'表达式在位置 {p} 意外结束(缺 {want or "内容"})\n  {self.s}')
        if want is not None and v != want:
            raise ExprError(f'位置 {p}:期望 {want!r},实际 {v!r}\n  {self.s}\n  {" " * p}^')
        self.i += 1
        return v, p

    def parse(self) -> Node:
        n = self.expr()
        if self.i != len(self.t):
            _, v, p = self.peek()
            raise ExprError(f'位置 {p}:表达式结束后有多余内容 {v!r}\n  {self.s}\n  {" " * p}^')
        return n

    def expr(self) -> Node:
        k, v, p = self.peek()
        if k == 'num':
            raise ExprError(f'位置 {p}:{v} —— 裸常数不能作节点。常数只能作带标量参数算子的标量、'
                            f'比较等二元算子的一侧,或时序算子的窗口\n  {self.s}')
        if k != 'id':
            raise ExprError(f'位置 {p}:期望算子名或字段名,实际 {v!r}\n  {self.s}\n  {" " * p}^')
        self.eat()
        if self.peek()[1] != '(':
            if v not in self.fields:
                near = [f for f in self.fields if f.startswith(v[:3])][:5]
                raise ExprError(f'位置 {p}:未知字段 {v!r}(不在当前词表 {len(self.fields)} 个字段中)。'
                                f'相近: {near or "无"}\n  {self.s}')
            return Node.make('operand', v)
        if v not in OPS:
            near = [o for o in OPS if o.startswith(v[:3])][:5]
            raise ExprError(f'位置 {p}:未知算子 {v!r}。相近: {near or "无"}\n  {self.s}')
        self.eat('(')
        args, nums = [], []
        while True:
            k2, v2, p2 = self.peek()
            if k2 == 'num':
                self.eat()
                nums.append((float(v2), p2))
                args.append(None)
            else:
                args.append(self.expr())
                nums.append(None)
            if self.peek()[1] == ',':
                self.eat(',')
                continue
            self.eat(')')
            break
        return self.build(v, args, nums, p)

    def build(self, name, args, nums, p) -> Node:
        kind = OPS[name].kind
        if len(args) != ARITY[kind]:
            raise ExprError(f'位置 {p}:{name} 需要 {ARITY[kind]} 个参数,给了 {len(args)}\n  {self.s}')
        if kind == 'unary':
            return Node.make(kind, name, None, (self.sub(name, args, nums, 0),))
        if kind in ('binary', 'ternary'):
            return Node.make(kind, name, None, tuple(self.panel(args, nums, i) for i in range(len(args))))
        if kind == 'binary_const':
            k = self.scalar(name, nums, 1, p, '标量')
            if name in ('mul_const', 'pow_const') and k == 0.0:
                raise ExprError(f'位置 {nums[1][1]}:{name} 常数 0 使整棵子树退化(mul_const→恒 0,pow_const→恒 1)\n  {self.s}')
            leaf = 'const_mul' if name in ('mul_const', 'pow_const') else 'const_add'
            return Node.make(kind, name, None, (self.sub(name, args, nums, 0), Node.make(leaf, k)))
        if kind == 'ts':
            return Node.make(kind, name, self.window(name, nums, 1, p), (self.sub(name, args, nums, 0),))
        if kind == 'ts_param':
            k = self.scalar(name, nums, 2, p, '标量(衰减因子 / 平移量 / 分布码)')
            return Node.make(kind, name, self.window(name, nums, 1, p),
                             (self.sub(name, args, nums, 0), Node.make('const_add', k)))
        if kind == 'pair':
            return Node.make(kind, name, self.window(name, nums, 2, p),
                             (self.sub(name, args, nums, 0), self.sub(name, args, nums, 1)))
        k = self.scalar(name, nums, 3, p, '标量(rettype:0=残差 1=截距 2=斜率 3=拟合 4=R²)')
        return Node.make(kind, name, self.window(name, nums, 2, p),
                         (self.sub(name, args, nums, 0), self.sub(name, args, nums, 1), Node.make('const_add', k)))

    def panel(self, args, nums, i) -> Node:
        """A panel slot: a subexpression, or a bare number broadcast as a constant panel."""
        return args[i] if nums[i] is None else Node.make('const_add', nums[i][0])

    def sub(self, name, args, nums, i) -> Node:
        if nums[i] is not None:
            raise ExprError(f'位置 {nums[i][1]}:{name} 第 {i + 1} 个参数必须是子表达式,不能是常数 {nums[i][0]:g}\n  {self.s}')
        return args[i]

    def scalar(self, name, nums, i, p, what) -> float:
        if nums[i] is None:
            raise ExprError(f'位置 {p}:{name} 第 {i + 1} 个参数必须是{what}\n  {self.s}')
        return nums[i][0]

    def window(self, name, nums, i, p) -> int:
        if nums[i] is None:
            raise ExprError(f'位置 {p}:{name} 第 {i + 1} 个参数必须是窗口整数 [{MIN_WINDOW}, {MAX_WINDOW}]\n  {self.s}')
        w, wp = nums[i]
        if w != int(w):
            raise ExprError(f'位置 {wp}:窗口 {w:g} 必须是整数\n  {self.s}')
        if not MIN_WINDOW <= w <= MAX_WINDOW:
            raise ExprError(f'位置 {wp}:窗口 {int(w)} 越界 [{MIN_WINDOW}, {MAX_WINDOW}]'
                            f'(w=1 使多数时序算子退化为恒等;上界约面板行数量级)\n  {self.s}')
        return int(w)


def parse(s: str, fields: Iterable[str]) -> Tree:
    """Formula string -> Tree; raises ExprError with the position of the first problem."""
    return Tree(_Parser(s.strip(), set(fields)).parse())


def causality(tree: Tree):
    """Why the tree is not causal on every backend, or None."""
    bad = [f'{n.op} 窗口 {n.window} > {OPS[n.op].causal_window}' for n in tree.root.walk()
           if not n.leaf and OPS[n.op].causal_window is not None and n.window > OPS[n.op].causal_window]
    return f'非因果算子:{"; ".join(bad)}(CPU 实现在该窗口以上用全样本桶边界)' if bad else None


def validate(s: str, fields: Iterable[str]) -> Dict:
    """-> dict(ok, error, cost, n_ops, fields, windows, pretty, tree); never raises (batch screening)."""
    try:
        t = parse(s, fields)
    except ExprError as e:
        return dict(ok=False, error=str(e))
    nodes = list(t.root.walk())
    cost = t.cost()
    if cost > MAX_COST:
        return dict(ok=False, error=f'cost {cost} > 上限 {MAX_COST}([expr] max_cost)。简化表达式或减少嵌套层数', cost=cost)
    if not any(n.kind == 'operand' for n in nodes):
        return dict(ok=False, error='表达式不含任何字段(纯常数树)')
    why = causality(t)
    if why:
        return dict(ok=False, error=why, cost=cost)
    return dict(ok=True, error=None, cost=cost, n_ops=sum(not n.leaf for n in nodes), fields=t.fields(),
                windows=sorted({n.window for n in nodes if n.window is not None}), pretty=t.pretty(), tree=t)
