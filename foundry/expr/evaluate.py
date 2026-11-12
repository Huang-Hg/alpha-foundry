"""Tree evaluation on any operator backend.

`evaluate(node, fields, backend, cache)` walks the tree once, calls the backend implementation of every operator
with the signature of its kind, and keeps each internal subtree's result in `cache` under the subtree hash, so the
common subtrees of a batch of formulas are computed once. Field panels are returned as they are (never copied into
the cache). A backend is (array module, implementation table); `backend(name)` loads foundry/ops/<name>.py:
'cpu' (the C++ reference), 'cuda' (cupy kernels) or 'portable' (numpy only).
"""
from __future__ import annotations

import importlib
from typing import Callable, Dict, Mapping, NamedTuple

from foundry.expr.ast import Node


class Backend(NamedTuple):
    xp: object
    impl: Dict[str, Callable]


def backend(name: str) -> Backend:
    xp = importlib.import_module('cupy' if name == 'cuda' else 'numpy')
    return Backend(xp, importlib.import_module(f'foundry.ops.{name}').IMPL)


def evaluate(n: Node, fields: Mapping, backend: Backend, cache: Dict[int, object]):
    if n.kind == 'operand':
        return fields[n.op]
    if n.kind in ('const_add', 'const_mul'):
        return backend.xp.full(next(iter(fields.values())).shape, float(n.op), dtype=backend.xp.float64)
    if n.hash in cache:
        return cache[n.hash]
    f = backend.impl[n.op]
    kind, ch = n.kind, n.children

    def sub(i):
        return evaluate(ch[i], fields, backend, cache)

    if kind == 'unary':
        v = f(sub(0))
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
