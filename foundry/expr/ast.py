"""Formula trees with a Merkle hash.

A node is (kind, op, window, children). Leaves: `operand` (op = field name) and the scalar leaves `const_add` /
`const_mul` (op = float). Every other kind is an operator of the registry; `window` is set on windowed kinds.
The hash of a node mixes its own (kind, op, window) with its children's hashes, children of a commutative operator
in hash order, so equivalent writings share one hash (the evaluators key their subtree caches on it).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from foundry.ops.registry import COST, OPS

_FNV_OFFSET, _FNV_PRIME, _MASK64 = 0xcbf29ce484222325, 0x100000001b3, 0xffffffffffffffff


def _fnv(data: bytes, h: int = _FNV_OFFSET) -> int:
    for b in data:
        h = ((h ^ b) * _FNV_PRIME) & _MASK64
    return h


def _combine(parts) -> int:
    return _fnv(b''.join(p.to_bytes(8, 'little') for p in parts))


@dataclass(frozen=True, eq=False)
class Node:
    kind: str
    op: object
    window: Optional[int]
    children: Tuple['Node', ...]
    hash: int

    @staticmethod
    def make(kind: str, op, window: Optional[int] = None, children: Tuple['Node', ...] = ()) -> 'Node':
        if kind in ('binary', 'pair') and OPS[op].commutative and children[0].hash > children[1].hash:
            children = (children[1], children[0])
        seed = _fnv(f'{kind}|{op}|{window}'.encode('utf-8'))
        return Node(kind, op, window, children, _combine((seed,) + tuple(c.hash for c in children)) if children else seed)

    @property
    def leaf(self) -> bool:
        return self.kind in ('operand', 'const_add', 'const_mul')

    def walk(self):
        yield self
        for c in self.children:
            yield from c.walk()


@dataclass(frozen=True, eq=False)
class Tree:
    root: Node

    @property
    def hash(self) -> int:
        return self.root.hash

    def cost(self) -> int:
        return sum(COST[n.kind] for n in self.root.walk())

    def fields(self):
        return sorted({n.op for n in self.root.walk() if n.kind == 'operand'})

    def pretty(self) -> str:
        return _pretty(self.root)


def _pretty(n: Node) -> str:
    if n.kind == 'operand':
        return n.op
    if n.kind in ('const_add', 'const_mul'):
        return f'{n.op:g}'
    args = [_pretty(c) for c in n.children]
    if n.kind in ('ts', 'pair'):
        args.append(str(n.window))
    elif n.kind in ('ts_param', 'pair_param'):         # the scalar comes after the window
        args.insert(len(args) - 1, str(n.window))
    return f'{n.op}({", ".join(args)})'
