"""Q25 CUDA float64 expression operators and device-resident calendar metadata."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cupy as cp
import numpy as np

from engine import PPY  # before ops_cuda: engine sets Q25_CUDA_FP64, which ops_cuda reads once at import
from backtest import ops_cuda as oc
from evaluation.grammar import (BinaryOp, ConstBinaryOp, LogicalBinaryOp, LogicalUnaryOp,
                                PairOp, PairParamOp, ParamOp, TernaryOp, TsInjOp, TsOp,
                                TsParamOp, UnaryOp)

D = oc.DTYPE


# ---------------------------------------------------------------- 补齐算子
# 谓词族(gt/lt/logic/tail/if_else)全部走**融合 ElementwiseKernel**,不用 cp.where 组合。
# 2026-08-13 实测(T×S=3103×74):组合写法每个算子 0.26–0.55 ms,而同形状的纯算术
# ufunc 只要 0.05 ms —— 差的是 `cp.where(mask, nan, f(a,b).astype(D)).astype(D)` 里
# 3~4 个中间张量的分配与来回读写(全是带宽,不是算力)。融合后与算术同级。
# 这一族恰好是 LLM 写条件型 alpha 的主力,所以值得单独修。
_NAN_PRE = r'''
template<typename T> __device__ __forceinline__ T q25nan();
template<> __device__ __forceinline__ float  q25nan<float>()  { return __int_as_float((int)0x7fffffff); }
template<> __device__ __forceinline__ double q25nan<double>() { return __longlong_as_double((long long)0x7ff8000000000000ULL); }
'''


def _ew(name: str, in_params: str, body: str):
    return cp.ElementwiseKernel(in_params, 'T o', body, f'q25_{name}', preamble=_NAN_PRE)


def _pred2(name: str, cond: str):
    """二元谓词:任一侧 NaN → NaN,否则 1/0(逐字对齐 ops_xs.cpp 的 FM_BIN_NAN)。"""
    return _ew(name, 'T a, T b',
               f'o = (isnan(a) || isnan(b)) ? q25nan<T>() : (T)(({cond}) ? 1 : 0);')


gt = _pred2('gt', 'a > b')
lt = _pred2('lt', 'a < b')


# 0 分母 → NaN(对齐 C 内核:零方差窗不产生 ±inf 而是无定义)。
# 融合版:原先 `cp.where(|d|<eps, nan, n/cp.where(|d|<eps,1,d))` 要 4 趟显存;
# ts_zscore/ts_snr/ts_chanpos/ts_ddown/ts_scale/ts_regression 六处都走它。
_safe_div = _ew('safe_div', 'T n, T d',
                'o = (fabs((double)d) < 1e-12) ? q25nan<T>() : (T)(n / d);')


# C 的 fm_ts_zscore / fm_ts_snr 用**总体**标准差(ss/w,ops_ts.cpp:390/404),而 ts_std 算子本身是样本
# (w−1)。此前这里直接用 oc.ts_std ⇒ 差 sqrt(w/(w−1)):w=120 时 4.2e-3、w=30 时 1.7e-2 —— 2026-09-19
# 在 f64 卡上复现同样的 4.18e-3 才确认是口径不是精度(原注释归因 f32 相消是错的)。
def _pop_std(x, w):
    return oc.ts_std(x, w) * D(((w - 1) / w) ** 0.5)


def ts_zscore(x, w):
    return _safe_div(x - oc.ts_mean(x, w), _pop_std(x, w))


def ts_snr(x, w):
    return _safe_div(oc.ts_mean(x, w), _pop_std(x, w))


# 平窗(hi == lo)时 C 内核返回 **0.5**(`ops_ts.cpp:353`:`(mx > mn) ? … : 0.5`),
# 语义是「通道中位」;而 `_safe_div` 给 NaN,下游 `_book` 把 NaN 当**空仓**。
# 两者差的不是精度是仓位:实测 dvol w7 上 **2.15%** 的格子 GPU 凭空空仓。
# ts_chanpos 是池里出现最多的算子(1,297 条里 1,283 条用到),所以这 2% 值得修。
# 与 ts_ddown 同一类缺陷(见下方长注),都是照抄除法却没照抄定义域。
# ⚠️ NaN 传播必须**显式**写:`(hi > lo)` 在 hi/lo 为 NaN 时为 false,会把整个 warmup 段
# (ma_dist_60 w120 上占 54%)误判成「平窗」而填 0.5。第一版就是这么写错的,gpu.py 的
# 对拍立刻从 NaN 一致 1.0000 掉到 0.9617 —— 这也说明那套对拍**是能抓 NaN 的**,
# ts_ddown 当初漏网是因为它的分歧点恰好在「C 给 NaN 而 GPU 给数」的方向上被
# `isfinite` 交集掩掉了,不是对拍本身没有 NaN 项。
_chanpos = _ew('chanpos', 'T x, T lo, T hi',
               'o = (isnan(x) || isnan(lo) || isnan(hi)) ? q25nan<T>() '
               ': ((hi > lo) ? (T)((x - lo) / (hi - lo)) : (T)0.5);')


def ts_chanpos(x, w):
    lo, hi = oc.ts_min(x, w), oc.ts_max(x, w)
    return _chanpos(x, lo, hi)


# ts_ddown 的根因是 **NaN 定义域不匹配,不是数值精度**(2026-08-14 修;第一版诊断猜的是
# f32 灾难性相消,被 `tools/ddown_root_cause.py` 证伪 —— 两种代数写法在共同定义域上误差相同)。
#
# C 内核 `ops_ts.cpp:370` 是 `(mx > 0.0) ? v/mx - 1.0 : NAN` —— **滚动最大值 ≤0 时无定义**。
# 旧 GPU 版走 `_safe_div`,只在 `|mx| < 1e-12` 时给 NaN ⇒ **mx 为负时 C 给 NaN、GPU 给数值**。
# 而 ts_ddown 的常见入参(`ma_dist_*` / `ret_*`)是有符号的,整窗低于均线时滚动最大值就是负的。
#
# 实测「gold 判无定义而 GPU 给了数」的格子占比:
#     ma_dist_60 w20 **16.2%** · ret_20 w20 8.2% · ma_dist_20 w20 6.2%
#     btc_rv / rangepos(滚动最大值恒正)**0.00%** —— 与机制预测完全吻合
# 这些格子上 GPU 的书是**凭空建仓**的。端到端后果(tools/gpu_error.py,46,210 条全量配对):
# 含 ts_ddown 的 119 条 |ΔSharpe| 中位 0.0468 / 最大 0.3925,而不含它的 3,164 条中位恰好
# 0.000000;越过 G2 那条 1.58 线的 32 次判定翻转里 **22 次**出在这一个算子上。修复后该族
# 中位降到 0.000000、最大 0.0018、翻转 0 次,G1-G3 池 1,238 → **1,297**(+59,+4.8%)。
#
# ⚠️ 教训:`__main__` 的算子级对拍**一直是过的**。它按相对误差比数值,而分歧点上 gold 是
# NaN —— 用 `isfinite` 掩码一取交集,恰好把唯一的分歧点全掩掉了。**算子对拍必须单独报
# NaN 一致率,只比数值等于没比。**
#
# 写成 (x−mx)/mx 而非 x/mx−1 只是顺带的条件数改善(共同域上实测无差别),不是修复本身。
_ddown = _ew('ddown', 'T x, T mx',
             'o = (mx > (T)0) ? (T)((x - mx) / mx) : q25nan<T>();')


def ts_ddown(x, w):
    return _ddown(x, oc.ts_max(x, w))


# 成书:liquid 且有限 → clip[0,1] / n_liq,否则 0(现金)。与 engine._book 逐字同语义,单 launch。
_book = _ew('book', 'T v, bool liq, T nliq',
            'o = (liq && isfinite(v)) ? (((v < (T)0) ? (T)0 : ((v > (T)1) ? (T)1 : v)) / nliq) : (T)0;')


_UNARY = {UnaryOp.ABS: oc.abs_, UnaryOp.NEG: oc.neg, UnaryOp.SIGN: oc.sign,
          UnaryOp.LOG: oc.log, UnaryOp.SQUARE: oc.square, UnaryOp.SQRT: oc.sqrt_,
          UnaryOp.TANH: oc.tanh_, UnaryOp.INV: oc.inv, UnaryOp.S_LOG_1P: oc.s_log_1p}
_BINARY = {BinaryOp.ADD: oc.add, BinaryOp.SUB: oc.sub, BinaryOp.MUL: oc.mul,
           BinaryOp.DIV: oc.div, BinaryOp.MAX: oc.max_b, BinaryOp.MIN: oc.min_b,
           BinaryOp.GT: gt, BinaryOp.LT: lt}
_BINCONST = {ConstBinaryOp.ADD_CONST: oc.add_const, ConstBinaryOp.MUL_CONST: oc.mul_const,
             ConstBinaryOp.POW_CONST: oc.pow_const}
_TS = {TsOp.TS_MEAN: oc.ts_mean, TsOp.TS_STD: oc.ts_std, TsOp.TS_MAX: oc.ts_max,
       TsOp.TS_MIN: oc.ts_min, TsOp.TS_SUM: oc.ts_sum, TsOp.TS_RANK: oc.ts_rank,
       TsOp.TS_ARG_MAX: oc.ts_arg_max, TsOp.TS_ARG_MIN: oc.ts_arg_min,
       TsOp.TS_EMA: oc.ts_ema, TsOp.TS_WMA: oc.ts_wma, TsOp.TS_REF: oc.ts_ref,
       TsOp.TS_DELTA: oc.ts_delta, TsOp.TS_SKEW: oc.ts_skew, TsOp.TS_KURT: oc.ts_kurt,
       TsOp.TS_MAD: oc.ts_mad, TsOp.TS_SLOPE: oc.ts_slope,
       TsOp.TS_CHANPOS: ts_chanpos, TsOp.TS_DDOWN: ts_ddown,
       TsOp.TS_ZS: ts_zscore, TsOp.TS_SNR: ts_snr}
_PAIR = {PairOp.TS_CORR: oc.ts_corr, PairOp.TS_COV: oc.ts_cov}


# ---------------------------------------------------------------- 注入层算子(GPU 侧)
# 语义逐条对齐 C 内核(backtest/ops/ops_xs.cpp);is_nan 刻意不传播 NaN。
_not = _ew('not', 'T a', 'o = isnan(a) ? q25nan<T>() : (T)((a > 0) ? 0 : 1);')
_is_nan = _ew('is_nan', 'T a', 'o = (T)(isnan(a) ? 1 : 0);')     # 唯一不传播 NaN 的

_LOGUNARY = {LogicalUnaryOp.NOT: _not, LogicalUnaryOp.IS_NAN: _is_nan}
_LOGBINARY = {
    LogicalBinaryOp.AND:       _pred2('and', '(a > 0) && (b > 0)'),
    LogicalBinaryOp.OR:        _pred2('or', '(a > 0) || (b > 0)'),
    LogicalBinaryOp.EQUAL:     _pred2('equal', 'a == b'),
    LogicalBinaryOp.NOT_EQUAL: _pred2('not_equal', 'a != b'),
    LogicalBinaryOp.GE:        _pred2('ge', 'a >= b'),
    LogicalBinaryOp.LE:        _pred2('le', 'a <= b'),
}
# 阈值 k 声明为 float64 ⇒ 比较在双精度里做,与 C 内核同侧;此前 `cp.where(x > k, ...)`
# 会先把 k 降到 f32,制造出一整类**只在阈值 ±1ulp 上**的分歧(verify_ops 里那条边界判据
# 就是为它写的)。现在只剩输入本身的 f32 舍入这一个来源。
_PARAM = {
    ParamOp.LEFT_TAIL:  _ew('left_tail', 'T a, float64 k',
                            'o = (a > (T)k) ? q25nan<T>() : a;'),
    ParamOp.RIGHT_TAIL: _ew('right_tail', 'T a, float64 k',
                            'o = (a < (T)k) ? q25nan<T>() : a;'),
}
_TERNARY = {
    TernaryOp.IF_ELSE: _ew('if_else', 'T c, T a, T b',
                           'o = isnan(c) ? q25nan<T>() : ((c > 0) ? a : b);'),
}


# ---------------------------------------------------------------- 注入层 rolling(GPU 侧)
_scale_fin = _ew('scale_fin', 'T x, T lo, T hi, float64 k',
                 'T d = hi - lo;'
                 'o = (fabs((double)d) < 1e-12) ? q25nan<T>() : (T)((x - lo) / d + k);')
_avd_pre = cp.ElementwiseKernel('T x', 'T xz, T fm',
                                'bool f = isfinite(x); xz = f ? x : (T)0; fm = (T)(f ? 1 : 0);',
                                'q25_avd_pre')
_avd_post = _ew('avd_post', 'T x, T s, T n',
                'o = (isfinite(x) && n > 0) ? (T)(x - s / n) : q25nan<T>();')


def _ts_scale(x, w, k):
    return _scale_fin(x, oc.ts_min(x, w), oc.ts_max(x, w), float(k))


def _ts_av_diff(x, w):
    """x − mean(x,d),**均值忽略 NaN**(BRAIN 定义)。
    oc.ts_sum 会传播 NaN ⇒ 先把 NaN 换 0 再求和,另用有限位掩码求个数。
    前后两步各融成一个 kernel(pre 一次出两张图,post 一次收口),把 7 趟显存压到 4 趟。"""
    xz, fm = _avd_pre(x)
    return _avd_post(x, oc.ts_sum(xz, w), oc.ts_sum(fm, w))


def _ts_quantile(x, w, driver):
    """窗内序数位置 → 逆 CDF。端点约定与 C 内核一致:绘图位置 p=(r+0.5)/w,
    r 由 oc.ts_rank 的 [0,1] 读数还原(r = rank01·(w−1))⇒ 两端有限、对称。"""
    from cupyx.scipy.special import ndtri
    p = (oc.ts_rank(x, w) * (w - 1.0) + 0.5) / float(w)
    dr = int(round(float(driver)))
    if dr == 1:
        return p.astype(D)
    if dr == 2:
        return cp.tan(cp.pi * (p - 0.5)).astype(D)
    return ndtri(p.astype(cp.float64)).astype(D)


def _ts_regression(y, x, w, rettype):
    """窗内 y ~ α+β·x 滚动 OLS。由 oc.ts_sum 组合(min_periods 与 NaN 传播由它保证);
    但 x/y 任一侧 NaN 必须同时废掉两侧的和 ⇒ 先做联合掩码。"""
    m = cp.isfinite(x) & cp.isfinite(y)
    xm = cp.where(m, x, cp.nan).astype(D)
    ym = cp.where(m, y, cp.nan).astype(D)
    n = float(w)
    Sx, Sy = oc.ts_sum(xm, w), oc.ts_sum(ym, w)
    Sxx, Sxy = oc.ts_sum(xm * xm, w), oc.ts_sum(xm * ym, w)
    dxx = n * Sxx - Sx * Sx
    beta = _safe_div(n * Sxy - Sx * Sy, dxx)
    alpha = (Sy - beta * Sx) / n
    rt = int(round(float(rettype)))
    if rt == 1:
        return alpha.astype(D)
    if rt == 2:
        return beta.astype(D)
    fit = alpha + beta * x
    if rt == 3:
        return fit.astype(D)
    if rt == 4:
        Syy = oc.ts_sum(ym * ym, w)
        cov = n * Sxy - Sx * Sy
        return _safe_div(cov * cov, dxx * (n * Syy - Sy * Sy)).astype(D)
    return (y - fit).astype(D)


_TSINJ = {TsInjOp.TS_AV_DIFF: _ts_av_diff}
_TSPARAM = {TsParamOp.TS_SCALE: _ts_scale, TsParamOp.TS_QUANTILE: _ts_quantile}
_PAIRPARAM = {PairParamOp.TS_REGRESSION: _ts_regression}

# **列内时序递归算子**:out[t] 依赖 out[t−1],沿 T 不可并行。批并行 GPU 实现需要
# segmented scan,收益不抵实现与验证风险(且这两个算子多用于终选阶段的少量候选)。
# ⇒ 含这些算子的候选整棵树走 CPU f64(见 `plan()`),GPU 批只跑纯可并行的树。
# 另有三个注入层 rolling 无 GPU 实现,原因各不相同:
#   ts_backfill / ts_median  —— 逐点回看 / 窗内排序,cupy 里没有能复用的 primitive,
#                               写自定义 kernel 的收益不抵验证风险(它们是深加工算子,不在海选热路径)
#   ts_decay_exp_window      —— 需按 w 个滞后加权,GPU 上要 w 次 kernel 启动(w 最大 1500),
#                               比 CPU 单 pass 还慢
from backtest.ops_cuda import q25 as native
_TERNARY[TernaryOp.TRADE_WHEN]=native.trade_when
_PARAM[ParamOp.HUMP]=native.hump
_TSINJ.update({TsInjOp.TS_BACKFILL:native.backfill,TsInjOp.TS_MEDIAN:native.median})
_TSPARAM[TsParamOp.TS_DECAY_EXP_WINDOW]=native.decay

_GPU_OPS = frozenset(
    set(_UNARY) | set(_LOGUNARY) | set(_BINARY) | set(_LOGBINARY) |
    set(_BINCONST) | set(_PARAM) | set(_TERNARY) | set(_TS) | set(_PAIR) |
    set(_TSINJ) | set(_TSPARAM) | set(_PAIRPARAM))


# rolling 核的 shared-memory tile 上限(逐卡不同;T1000 实测 2032,pair 核 1008)。
# `fastexpr.MAX_WINDOW` = 1500 ⇒ **完全合法的表达式可以要求 GPU 装不下的窗口**,
# 必须按能力路由,不能指望"没人会这么写"。2026-08-13 前 `ts_mean(x, 730)` 会让
# `ops_cuda._moment` 抛裸 IndexError,把同批其他候选一起打死。
# mult=2 只对 kind=='pair'(ts_corr/ts_cov 两个数组同时进 shared);
# ts_regression / ts_quantile / ts_scale / ts_av_diff 的 GPU 实现由单数组核组合 ⇒ mult=1。
MAX_W_1 = oc.max_window(1)
MAX_W_2 = oc.max_window(2)


def _win_cap(n) -> int:
    return MAX_W_2 if n.kind == 'pair' else MAX_W_1


def _over_window(n) -> bool:
    w = getattr(n, 'window', None)
    return w is not None and int(w) > _win_cap(n)


def can_gpu(tree) -> bool:
    """整棵树是否可走 GPU 批。任一节点算子不在 GPU 实现表内、**或窗口超出本卡 tile 上限**
    ⇒ 明确拒绝，不存在 CPU 回退。"""
    def walk(n):
        if n.kind not in ('operand', 'const_add', 'const_mul') and n.op not in _GPU_OPS:
            return False
        if _over_window(n):
            return False
        return all(walk(c) for c in n.children)
    return walk(tree.root if hasattr(tree, 'root') else tree)


def unsupported_ops(tree) -> list:
    """树里尚不支持 CUDA 的原因(诊断用):算子名,或 `算子@w=窗口>上限`。"""
    out = []

    def walk(n):
        nm = getattr(n.op, 'value', str(n.op))
        if n.kind not in ('operand', 'const_add', 'const_mul') and n.op not in _GPU_OPS:
            out.append(nm)
        elif _over_window(n):
            out.append(f'{nm}@w={int(n.window)}>{_win_cap(n)}')
        for c in n.children:
            walk(c)

    walk(tree.root if hasattr(tree, 'root') else tree)
    return out


def eval_node(n, dp: Dict[str, cp.ndarray], cache: Dict[int, cp.ndarray]) -> cp.ndarray:
    """AST 节点 → 设备端 (T,S)。subtree-hash 缓存(同 CPU 侧策略,批内公共子树只算一次)。"""
    h = n.hash
    if h in cache:
        return cache[h]
    k = n.kind
    if k == 'operand':
        return dp[n.op]                                   # panels 视图,不入 cache
    if k in ('const_add', 'const_mul'):
        return cp.full(next(iter(dp.values())).shape, float(n.op), dtype=D)
    c = [eval_node(ch, dp, cache) for ch in n.children]
    if k == 'unary':
        v = (_UNARY.get(n.op) or _LOGUNARY[n.op])(c[0])
    elif k == 'binary':
        v = (_BINARY.get(n.op) or _LOGBINARY[n.op])(c[0], c[1])
    elif k == 'binary_const':
        f = _BINCONST.get(n.op) or _PARAM[n.op]
        v = f(c[0], float(n.children[1].op))
    elif k == 'ternary':
        v = _TERNARY[n.op](c[0], c[1], c[2])
    elif k == 'ts':
        v = (_TS.get(n.op) or _TSINJ[n.op])(c[0], int(n.window))
    elif k == 'ts_param':
        v = _TSPARAM[n.op](c[0], int(n.window), float(n.children[1].op))
    elif k == 'pair':
        v = _PAIR[n.op](c[0], c[1], int(n.window))
    elif k == 'pair_param':
        v = _PAIRPARAM[n.op](c[0], c[1], int(n.window), float(n.children[2].op))
    else:
        raise KeyError(f'未知节点 kind={k}')
    cache[h] = v
    return v


N_STREAMS = 8       # 树求值并发 stream 数(2026-09-19):滚动核沿 T 串行扫,单次 launch 有 ~60 µs 的
                    # 延迟地板、只用到一小部分 SM(A800 上树求值占批耗时 73%,134 µs/节点);不同树
                    # 互不依赖 ⇒ 轮转到 N 条 stream 上让 kernel 重叠。上限 ≈ Python 侧发核开销
                    # (~30-40 µs/节点)/ GPU 侧 134 µs ≈ 3-4×,再多 stream 无益。


class GpuBatch:
    """面板/静态量常驻显存;设备数组与日历索引；CUDA 批量年度指标。"""

    def __init__(self, panels: Dict[str, np.ndarray], liquid: np.ndarray,
                 static: Dict[str, np.ndarray], row0: int, dates: np.ndarray, batch: int = 256):
        self.dp = {k: cp.ascontiguousarray(cp.asarray(v,dtype=D)) for k, v in panels.items()}
        self.dliq = cp.asarray(liquid.astype(np.bool_))
        self.row0 = int(row0)
        self.batch = batch
        self.nliq = cp.maximum(self.dliq.sum(1, keepdims=True), 1).astype(D)
        self.T, self.S = liquid.shape
        # 日历年指示矩阵 (Y, T-row0):逐年 Sharpe / 在场比 = 一次矩阵乘,口径同 engine._metrics
        yrs = dates[self.row0:].astype('datetime64[Y]').astype(int) + 1970
        self.years = np.unique(yrs)
        self.days = np.array([int((yrs == y).sum()) for y in self.years])
        self.dM = cp.asarray((yrs[None, :] == self.years[:, None]).astype(np.float64))
        self.ddays = cp.asarray(self.days.astype(np.float64))
        self.k_top = max(1, int((self.T - self.row0) * 0.01))     # 集中度:top 1% 交易日
        self._streams = [cp.cuda.Stream(non_blocking=True) for _ in range(N_STREAMS)]




    def yearly(self, rt: cp.ndarray, attend: cp.ndarray, lg: cp.ndarray):
        """CUDA calendar reductions; reuse caller's log returns, avoid duplicate sorts."""
        s1 = rt @ self.dM.T
        s2 = (rt * rt) @ self.dM.T
        mean = s1 / self.ddays
        sd = cp.sqrt(cp.maximum(s2 / self.ddays - mean * mean, 0.0))
        pos = sd > 0
        ysh = cp.where(pos, mean / cp.where(pos, sd, 1.0) * PPY ** 0.5, 0.0)
        yatt = (attend.astype(cp.float64) @ self.dM.T) / self.ddays
        ylg = lg @ self.dM.T
        return ysh, yatt, ylg
