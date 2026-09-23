"""Search grammar and the native typed-prefix GFlowNet policy (foundry.search._gfn).

The grammar is a text protocol, one production per line: token, output slot, AST cost, child count, child slots.
Slots: PANEL a sub-expression, CONST a constant, WINDOW a time-series window, CAUSAL_WINDOW a WINDOW token no larger
than the causal limit of the operator (registry OPS[..].causal_window, handed to the policy), NONZERO a constant
other than 0, PAIR_WINDOW a pair window. Sampling, training and CEM numerics all run in the extension.
"""
from __future__ import annotations

import numpy as np

from foundry.ops.registry import COST, OPS, WINDOW_LADDER
from foundry.search import _gfn

PANEL, CONST, WINDOW, CAUSAL_WINDOW, NONZERO, PAIR_WINDOW = range(6)

# The searchable part of the operator table; the order fixes the action ids and so the sampled sequences.
UNARY = ('sign', 'abs', 'neg', 'log', 'sqrt', 'tanh', 's_log_1p')
BINARY = ('add', 'sub', 'mul', 'div', 'max', 'min', 'gt', 'lt')
COMPARE_CONST = ('gt', 'lt')                              # comparisons are also offered against a constant
BINARY_CONST = ('add_const', 'mul_const', 'pow_const')
NONZERO_CONST = ('mul_const', 'pow_const')                # a zero constant collapses the subtree
TS = ('ts_mean', 'ts_std', 'ts_min', 'ts_max', 'ts_delta', 'ts_ref', 'ts_slope', 'ts_zscore', 'ts_chanpos',
      'ts_arg_min', 'ts_arg_max', 'ts_ddown', 'ts_rank')
PAIR = ('ts_corr', 'ts_cov')
CONSTANTS = ('-2', '-1', '-0.5', '-0.2', '-0.1', '-0.05', '0', '0.05', '0.1', '0.2', '0.5', '0.8', '1', '2')
(CAUSAL,) = {OPS[name].causal_window for name in TS if OPS[name].causal_window}     # one CAUSAL_WINDOW slot type


def _op(name, *slots):
    return f'{name} {PANEL} {COST[OPS[name].kind]} {len(slots)} ' + ' '.join(map(str, slots))


def grammar(fields, window_caps):
    """window_caps = (largest time-series window, largest pair window) the panel depth allows."""
    rows = [f'{name} {PANEL} {COST["operand"]} 0' for name in sorted(fields)]
    rows += [_op(name, PANEL) for name in UNARY]
    rows += [_op(name, PANEL, PANEL) for name in BINARY]
    rows += [_op(name, PANEL, CONST) for name in COMPARE_CONST]
    rows += [_op(name, PANEL, NONZERO if name in NONZERO_CONST else CONST) for name in BINARY_CONST]
    rows += [_op(name, PANEL, CAUSAL_WINDOW if OPS[name].causal_window else WINDOW) for name in TS]
    rows += [_op(name, PANEL, PANEL, PAIR_WINDOW) for name in PAIR]
    # A window is an attribute of its operator's node, so it adds no cost of its own.
    rows += [f'{w} {WINDOW} 0 0' for w in WINDOW_LADDER if w <= window_caps[0]]
    rows += [f'{w} {PAIR_WINDOW} 0 0' for w in WINDOW_LADDER if w <= window_caps[1]]
    rows += [f'{k} {CONST} {COST["const_add"]} 0' for k in CONSTANTS]
    return '\n'.join(rows)


class Policy:
    """The native policy over the grammar of `fields`: one sample() then one train() per epoch."""

    def __init__(self, fields, seed, max_tokens, max_cost, max_ts_depth, window_caps):
        self.spec = grammar(fields, window_caps)
        self.native = _gfn.Policy(self.spec, max_tokens, max_cost, max_ts_depth, CAUSAL, seed)

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


class CEM:
    """Cross-entropy search over additive member weights in (0,1), parametrised by their logits."""

    def __init__(self, initial_logits):
        self.mean = np.array(initial_logits, dtype=np.float64)
        self.sd = np.full(len(self.mean), .5)

    def sample(self, count, seed):
        """-> (count, n) float64 weights; the next update() scores these rows."""
        self.weights = _gfn.cem_sample(self.mean, self.sd, count, seed)
        return self.weights

    def update(self, scores, elite):
        _gfn.cem_update(self.weights, scores, elite, self.mean, self.sd)
