"""The operator table: every DSL operator exactly once.

An operator is a name plus its node kind; the kind fixes the call signature, the arity of its written form and its
cost class. The parser, the validator, the search grammar and the tree hash read this table; each compute backend
(ops/cpu.py, ops/cuda.py, ops/portable.py) maps the same names to implementations with the signature of the kind:

    unary          f(x)                 binary       f(a, b)             binary_const f(x, k)
    ternary        f(a, b, c)           ts           f(x, w)             ts_param     f(x, w, k)
    pair           f(a, b, w)           pair_param   f(a, b, w, k)

This module imports nothing numerical, so hosts without the native extensions (a hosted notebook) can use it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

# Written arguments of each kind (panels, windows and scalars alike).
ARITY = {'unary': 1, 'binary': 2, 'binary_const': 2, 'ternary': 3, 'ts': 2, 'ts_param': 3, 'pair': 3, 'pair_param': 4}
# Tree cost per node kind (leaves included); a formula's cost is the sum over its nodes.
COST = {'operand': 1, 'const_add': 1, 'const_mul': 1, 'unary': 2, 'binary': 3, 'binary_const': 2, 'ternary': 4,
        'ts': 3, 'pair': 5, 'ts_param': 3, 'pair_param': 5}
WINDOWED = frozenset({'ts', 'ts_param', 'pair', 'pair_param'})
# Window ladder shared by the search grammar and the neighbourhood of a hypothesis family.
WINDOW_LADDER = (2, 3, 5, 7, 10, 14, 20, 30, 45, 60, 90, 120, 180, 240, 365, 730)
MIN_WINDOW, MAX_WINDOW = 2, 1500      # w=1 makes most time-series operators the identity; the top is ~ the panel depth


@dataclass(frozen=True)
class Op:
    name: str
    kind: str
    commutative: bool = False         # children are ordered by hash, so equivalent trees hash alike
    doc: str = ''


_TABLE = (
    # element-wise
    Op('abs', 'unary'), Op('neg', 'unary'), Op('sign', 'unary'), Op('log', 'unary', doc='x<=0 -> NaN'),
    Op('square', 'unary'), Op('sqrt', 'unary', doc='x<0 -> NaN'), Op('tanh', 'unary'),
    Op('inv', 'unary', doc='x==0 -> NaN'), Op('s_log_1p', 'unary', doc='sign(x)*log1p(|x|)'),
    Op('not', 'unary', doc='x>0 -> 0 else 1'), Op('is_nan', 'unary', doc='the one operator that does not propagate NaN'),
    Op('add', 'binary', True), Op('sub', 'binary'), Op('mul', 'binary', True), Op('div', 'binary', doc='b==0 -> NaN'),
    Op('max', 'binary', True), Op('min', 'binary', True), Op('gt', 'binary'), Op('lt', 'binary'),
    Op('and', 'binary'), Op('or', 'binary'), Op('equal', 'binary'), Op('not_equal', 'binary'),
    Op('greater_equal', 'binary'), Op('less_equal', 'binary'),
    Op('add_const', 'binary_const'), Op('mul_const', 'binary_const'),
    Op('pow_const', 'binary_const', doc='sign(x)*|x|^k'),
    Op('left_tail', 'binary_const', doc='x>k -> NaN'), Op('right_tail', 'binary_const', doc='x<k -> NaN'),
    Op('hump', 'binary_const', doc='keep the previous output unless |x - previous| > k'),
    Op('if_else', 'ternary', doc='c>0 ? a : b'),
    Op('trade_when', 'ternary', doc='(entry, alpha, exit): exit>0 -> 0; entry>0 -> alpha; else hold'),
    # cross-sectional (per day across assets)
    Op('cs_rank', 'unary', doc='average rank in [0,1]'), Op('cs_zscore', 'unary'), Op('cs_demean', 'unary'),
    Op('cs_scale', 'unary', doc='x / sum|x|'),
    # time series
    Op('ts_mean', 'ts'), Op('ts_std', 'ts'), Op('ts_max', 'ts'), Op('ts_min', 'ts'), Op('ts_sum', 'ts'),
    Op('ts_rank', 'ts', doc='exact trailing-window average rank; ties share rank'),
    Op('ts_arg_max', 'ts'), Op('ts_arg_min', 'ts'), Op('ts_ema', 'ts'), Op('ts_wma', 'ts'), Op('ts_ref', 'ts'),
    Op('ts_delta', 'ts'), Op('ts_skew', 'ts'), Op('ts_kurt', 'ts'), Op('ts_mad', 'ts'), Op('ts_slope', 'ts'),
    Op('ts_chanpos', 'ts', doc='(x-min)/(max-min); a flat window is 0.5'), Op('ts_ddown', 'ts', doc='x/max-1; max<=0 -> NaN'),
    Op('ts_zscore', 'ts', doc='population std'), Op('ts_snr', 'ts', doc='mean / population std'),
    Op('ts_backfill', 'ts', doc='last valid value within w rows'), Op('ts_median', 'ts'),
    Op('ts_av_diff', 'ts', doc='x - mean ignoring NaN'),
    Op('ts_decay_exp_window', 'ts_param', doc='(x, w, factor)'), Op('ts_scale', 'ts_param', doc='(x, w, k): chanpos + k'),
    Op('ts_quantile', 'ts_param', doc='(x, w, driver): 0 gaussian, 1 uniform, 2 cauchy'),
    Op('ts_corr', 'pair', True), Op('ts_cov', 'pair', True),
    Op('ts_regression', 'pair_param', doc='(y, x, w, rettype): 0 residual, 1 intercept, 2 slope, 3 fit, 4 R^2'),
)
OPS: Dict[str, Op] = {op.name: op for op in _TABLE}


def by_kind(kind: str):
    return tuple(name for name, op in OPS.items() if op.kind == kind)
