"""CPU reference backend of the formula operators: the C++ extension foundry.ops._ops on float64 (T,S) arrays.

Kernels take C-contiguous panels and return new arrays; `IMPL` maps every registry name to a callable with the
signature of its kind (see foundry.ops.registry).
"""
from __future__ import annotations

from foundry.ops import _ops as c
from foundry.ops.registry import OPS

IMPL = dict(
    abs=c.abs_, neg=c.neg, sign=c.sign, log=c.log_, square=c.square, sqrt=c.sqrt_, tanh=c.tanh_, inv=c.inv,
    s_log_1p=c.s_log_1p, **{'not': c.not_}, is_nan=c.is_nan,
    add=c.add, sub=c.sub, mul=c.mul, div=c.div_, max=c.max_b, min=c.min_b, gt=c.gt, lt=c.lt,
    **{'and': c.and_, 'or': c.or_}, equal=c.equal, not_equal=c.not_equal,
    greater_equal=c.greater_equal, less_equal=c.less_equal,
    add_const=c.add_const, mul_const=c.mul_const, pow_const=c.pow_const,
    left_tail=c.left_tail, right_tail=c.right_tail, hump=c.hump,
    if_else=c.if_else, trade_when=c.trade_when,
    cs_rank=c.cs_rank, cs_zscore=c.cs_zscore, cs_demean=c.cs_demean, cs_scale=c.cs_scale,
    **{name: getattr(c, name) for name, op in OPS.items() if op.kind in ('ts', 'ts_param', 'pair', 'pair_param')},
)
assert set(IMPL) == set(OPS), set(OPS) ^ set(IMPL)
