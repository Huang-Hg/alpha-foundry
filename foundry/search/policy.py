"""Search grammar and the native typed-prefix GFlowNet policy (foundry.search._gfn).

The grammar is a text protocol, one production per line: token, output slot, AST cost, child count, child slots.
Slots: PANEL a sub-expression, CONST a constant, WINDOW a causal time-series window, NONZERO a constant
other than 0, PAIR_WINDOW a pair window. Sampling and training run in the extension.
"""
from __future__ import annotations

from foundry.ops.registry import COST, OPS, SEARCH_CONSTANTS, SEARCH_PRODUCTIONS, WINDOW_LADDER, Slot
from foundry.search import _gfn


def grammar(fields: list[str], window_caps: tuple[int, int]) -> str:
    """window_caps = (largest time-series window, largest pair window) the panel depth allows."""
    rows = [f'{name} {int(Slot.PANEL)} {COST["operand"]} 0' for name in sorted(fields)]
    rows += [f'{p.operator} {int(Slot.PANEL)} {COST[OPS[p.operator].kind]} {len(p.slots)} '
             + ' '.join(str(int(slot)) for slot in p.slots) for p in SEARCH_PRODUCTIONS]
    # A window is an attribute of its operator's node, so it adds no cost of its own.
    rows += [f'{w} {int(Slot.WINDOW)} 0 0' for w in WINDOW_LADDER if w <= window_caps[0]]
    rows += [f'{w} {int(Slot.PAIR_WINDOW)} 0 0' for w in WINDOW_LADDER if w <= window_caps[1]]
    rows += [f'{k} {int(Slot.CONST)} {COST["const_add"]} 0' for k in SEARCH_CONSTANTS]
    return '\n'.join(rows)


class Policy:
    """The native policy over the grammar of `fields`: one sample() then one train() per epoch."""

    def __init__(self, fields, seed, max_tokens, max_cost, max_ts_depth, window_caps):
        self.spec = grammar(fields, window_caps)
        self.native = _gfn.Policy(self.spec, max_tokens, max_cost, max_ts_depth, seed)

    def sample(self, count, probes, context, random=False):
        """-> dict(epoch, ordinary, probe_count, expressions): `ordinary` policy samples, then 12 per probe.
        random=True draws uniformly over legal actions (no probes)."""
        return self.native.sample(count, probes, context, random)

    def train(self, rewards, epoch, steps, lr, opd):
        """One log-reward per expression of batch `epoch`. Releases the GIL, so it can run in a worker thread."""
        return self.native.train(rewards, epoch, steps, lr, opd)

    def gradient_check(self, rewards):
        return self.native.gradient_check(rewards)

    def checkpoint(self, path, load=False):
        (self.native.load if load else self.native.save)(path)
