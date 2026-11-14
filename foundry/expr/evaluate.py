"""Tree evaluation on any operator backend.

`evaluate(node, fields, backend, cache, universe)` walks the tree once, calls the backend implementation of every operator
with the signature of its kind, and keeps each internal subtree's result in `cache` under the subtree hash, so the
common subtrees of a batch of formulas are computed once. Field panels are returned as they are (never copied into
the cache). Backend implementations are prepared by the runtime and injected.
"""
from __future__ import annotations

from typing import Dict, Mapping

from foundry.expr.ast import Node
from foundry.contracts import Backend


def evaluate(n: Node, fields: Mapping, backend: Backend, cache: Dict[int, object], universe):
    if n.kind == 'operand':
        return fields[n.op]
    if n.kind in ('const_add', 'const_mul'):
        return backend.xp.full(next(iter(fields.values())).shape, float(n.op), dtype=backend.xp.float64)
    if n.hash in cache:
        return cache[n.hash]
    f = backend.impl[n.op]
    kind, ch = n.kind, n.children

    def sub(i):
        return evaluate(ch[i], fields, backend, cache, universe)

    if kind == 'unary':
        x=sub(0)
        v = f(backend.xp.where(universe,x,backend.xp.nan)) if n.op.startswith('cs_') else f(x)
    elif kind == 'binary':
        v = f(sub(0), sub(1))
    elif kind == 'binary_const':
        v = f(sub(0), float(ch[1].op))
    elif kind == 'ternary':
        v = f(sub(0), sub(1), sub(2))
    elif kind == 'ts':
        v = f(sub(0), n.window)
    elif kind == 'ts_param':
        v = f(sub(0), n.window, float(ch[1].op))
    elif kind == 'pair':
        v = f(sub(0), sub(1), n.window)
    else:
        v = f(sub(0), sub(1), n.window, float(ch[2].op))
    cache[n.hash] = v
    return v
