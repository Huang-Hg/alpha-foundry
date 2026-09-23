"""CUDA backend of the formula operators: cupy arrays in, cupy arrays out, always float64.

Rolling statistics run the shared-memory tile kernels of kernels/ops.cu, path-dependent operators the column
walkers of kernels/paths.cu; the rest are fused element-wise kernels. `IMPL` maps every registry name to its
implementation (signatures per kind, see foundry.ops.registry). Launch geometry is derived from the device
properties, so the same code adapts to a 4 GB consumer card and a data-centre card; a window whose tile does not fit
the card's shared memory is reported by `unsupported` before anything is launched.
"""
from __future__ import annotations

from pathlib import Path

import cupy as cp
import numpy as np

from foundry.ops.registry import OPS

_KERNELS = Path(__file__).with_name('kernels')
_OPS = cp.RawModule(code=(_KERNELS / 'ops.cu').read_text(encoding='utf-8'), options=('--std=c++14',))
_PATHS = cp.RawModule(code=(_KERNELS / 'paths.cu').read_text(encoding='utf-8'), options=('--std=c++14', '--fmad=false'))

D = cp.float64
_ELEM = 8
_PROPS = cp.cuda.runtime.getDeviceProperties(0)
_WARP = _PROPS['warpSize']
_OPTIN = _PROPS['sharedMemPerBlockOptin']
_MAXTHR = _PROPS['maxThreadsPerBlock']
_SHBUDGET = min(_OPTIN, _PROPS['sharedMemPerMultiprocessor'] // 3)   # keeps occupancy; raised only when a window needs it


def _optin_max(fn) -> int:
    """Raise a kernel's dynamic shared-memory limit as far as the driver accepts: dynamic + static must stay within
    sharedMemPerBlockOptin, and Volta+ reserves 1 KB per block, so step down from optin - static by 1 KB."""
    for cap in range(_OPTIN - int(fn.shared_size_bytes), 0, -1024):
        try:
            fn.max_dynamic_shared_size_bytes = cap
            return cap
        except cp.cuda.driver.CUDADriverError:
            continue
    return _SHBUDGET


_k_moment, _k_ext, _k_pair, _k_ema, _k_cs, _k_cs_rank = (
    _OPS.get_function(n) for n in ('k_moment', 'k_ext', 'k_pair', 'k_ema', 'k_cs', 'k_cs_rank'))
_DYNMAX = min(_optin_max(f) for f in (_k_moment, _k_ext, _k_pair, _k_cs_rank))
_DUMMY = cp.zeros(1, dtype=D)                 # centering anchor placeholder for kmode 0 (never dereferenced)


def _candidate_tiles(w, budget, mult=1):
    """(assets, days) tiles whose (days + w - 1) x assets halo fits `budget`; larger tiles (more reuse) first."""
    out = [(ss, tt) for ss in (8, 16, 32) for tt in (256, 192, 128, 96, 64, 48, 32, 16)
           if ss * tt <= _MAXTHR and ss * tt % _WARP == 0 and (tt + w - 1) * ss * _ELEM * mult <= budget]
    return sorted(out, key=lambda st: -st[0] * st[1])


def _tile(w, mult=1):
    return (_candidate_tiles(w, _SHBUDGET, mult) or _candidate_tiles(w, _DYNMAX, mult))[0]


def _max_window(mult) -> int:
    lo, hi = 2, 1 << 16
    while lo < hi:                            # monotone: a larger window is harder to fit
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if _candidate_tiles(mid, _DYNMAX, mult) else (lo, mid - 1)
    return lo


MAX_WINDOW = _max_window(1)                   # single-array rolling kernels
MAX_PAIR_WINDOW = _max_window(2)              # ts_corr / ts_cov hold two arrays in shared memory


def window_cap(op: str) -> int:
    return MAX_PAIR_WINDOW if OPS[op].kind == 'pair' else MAX_WINDOW


def unsupported(tree):
    """Reasons this tree cannot run on this card: unknown operators or windows above the card's tile limit."""
    out = []
    for n in tree.root.walk():
        if n.leaf:
            continue
        if n.op not in IMPL:
            out.append(n.op)
        elif n.window is not None and n.window > window_cap(n.op):
            out.append(f'{n.op}@w={n.window}>{window_cap(n.op)}')
    return out


# ---------------------------------------------------------------- rolling kernels
def _moment(x, w, op, kmode, K):
    T, S = x.shape
    out = cp.empty_like(x)
    ss, tt = _tile(w)
    _k_moment(((S + ss - 1) // ss, (T + tt - 1) // tt), (ss, tt),
              (x, out, np.int32(T), np.int32(S), np.int32(w), np.int32(op), K, np.int32(kmode)),
              shared_mem=(tt + w - 1) * ss * _ELEM)
    return out


def _first_valid(x):
    """First valid value of every column (skew/kurt centering anchor); an all-NaN column -> 0."""
    fin = ~cp.isnan(x)
    return cp.where(fin.any(axis=0), x[fin.argmax(axis=0), cp.arange(x.shape[1])], 0).astype(D)


def _ext(x, w, op):
    T, S = x.shape
    out = cp.empty_like(x)
    ss, tt = _tile(w)
    _k_ext(((S + ss - 1) // ss, (T + tt - 1) // tt), (ss, tt), (x, out, np.int32(T), np.int32(S), np.int32(w), np.int32(op)),
           shared_mem=(tt + w - 1) * ss * _ELEM)
    return out


def _pair(a, b, w, op):
    T, S = a.shape
    out = cp.empty_like(a)
    ss, tt = _tile(w, 2)
    _k_pair(((S + ss - 1) // ss, (T + tt - 1) // tt), (ss, tt),
            (a, b, out, np.int32(T), np.int32(S), np.int32(w), np.int32(op)), shared_mem=2 * (tt + w - 1) * ss * _ELEM)
    return out


def ts_sum(x, w): return _moment(x, w, 0, 0, _DUMMY)
def ts_mean(x, w): return _moment(x, w, 1, 0, _DUMMY)
def ts_std(x, w): return _moment(x, w, 2, 0, _DUMMY)
def ts_mad(x, w): return _moment(x, w, 5, 0, _DUMMY)
def ts_skew(x, w): return _moment(x, w, 3, 1, _first_valid(x))
def ts_kurt(x, w): return _moment(x, w, 4, 1, _first_valid(x))
def ts_max(x, w): return _ext(x, w, 0)
def ts_min(x, w): return _ext(x, w, 1)
def ts_wma(x, w): return _ext(x, w, 2)
def ts_slope(x, w): return _ext(x, w, 3)
def ts_rank(x, w): return _ext(x, w, 4)
def ts_arg_max(x, w): return _ext(x, w, 5)
def ts_arg_min(x, w): return _ext(x, w, 6)
def ts_corr(a, b, w): return _pair(a, b, w, 0)
def ts_cov(a, b, w): return _pair(a, b, w, 1)


def ts_ema(x, span):
    T, S = x.shape
    out = cp.empty_like(x)
    _k_ema(((S + 127) // 128,), (128,), (x, out, np.int32(T), np.int32(S), np.int32(span)))
    return out


def ts_ref(x, w):
    out = cp.full_like(x, cp.nan)
    if w < x.shape[0]:
        out[w:] = x[:-w]
    return out


def ts_delta(x, w):
    return x - ts_ref(x, w)


def _cs(x, op):
    T, S = x.shape
    out = cp.empty_like(x)
    _k_cs((T,), (256,), (x, out, np.int32(S), np.int32(op)))
    return out


def cs_rank(x):
    """Per-day average rank in [0,1], O(S^2) per day with the row in shared memory; a day with < 2 values is NaN.
    (A bitonic sort was 8-12x slower on every real universe, 299-666 assets: its barriers cost more than the count.)"""
    T, S = x.shape
    out = cp.empty_like(x)
    _k_cs_rank((T,), (256,), (x, out, np.int32(S)), shared_mem=S * _ELEM)
    return out


# ---------------------------------------------------------------- element-wise (fused, one pass each)
_NAN = r'''
template<typename T> __device__ __forceinline__ T qnan();
template<> __device__ __forceinline__ double qnan<double>() { return __longlong_as_double((long long)0x7ff8000000000000ULL); }
'''


def _ew(name, params, body):
    return cp.ElementwiseKernel(params, 'T o', body, f'fx_{name}', preamble=_NAN)


def _pred(name, cond):
    """Binary predicate: NaN on either side -> NaN, else 1/0 (the CPU FM_BIN_NAN semantics)."""
    return _ew(name, 'T a, T b', f'o = (isnan(a) || isnan(b)) ? qnan<T>() : (T)(({cond}) ? 1 : 0);')


_sign = cp.ElementwiseKernel('T x', 'T o', 'o = isnan(x) ? x : (x > (T)0 ? (T)1 : (x < (T)0 ? (T)-1 : (T)0));', 'fm_sign')
_log = cp.ElementwiseKernel('T x', 'T o', 'o = (isnan(x) || x <= (T)0) ? (T)nan("") : log(x);', 'fm_log')
_sqrt = cp.ElementwiseKernel('T x', 'T o', 'o = (isnan(x) || x < (T)0) ? (T)nan("") : sqrt(x);', 'fm_sqrt')
_inv = cp.ElementwiseKernel('T x', 'T o', 'o = (isnan(x) || x == (T)0) ? (T)nan("") : (T)1 / x;', 'fm_inv')
_s_log_1p = cp.ElementwiseKernel('T x', 'T o', 'o = isnan(x) ? x : (x >= (T)0 ? log1p(x) : -log1p(-x));', 'fm_s_log_1p')
_div = cp.ElementwiseKernel('T a, T b', 'T o', 'o = (b == (T)0) ? (T)nan("") : a / b;', 'fm_div')
_pow = cp.ElementwiseKernel('T x, T k', 'T o',
                            'T av = fabs(x); if (av == (T)0) { o = (T)0; } else { T p = pow(av, k); o = (x < (T)0) ? -p : p; }',
                            'fm_pow_const')
_not = _ew('not', 'T a', 'o = isnan(a) ? qnan<T>() : (T)((a > 0) ? 0 : 1);')
_is_nan = _ew('is_nan', 'T a', 'o = (T)(isnan(a) ? 1 : 0);')
# Thresholds are float64 parameters, so the comparison happens on the same side of the rounding as on the CPU.
_left_tail = _ew('left_tail', 'T a, float64 k', 'o = (a > (T)k) ? qnan<T>() : a;')
_right_tail = _ew('right_tail', 'T a, float64 k', 'o = (a < (T)k) ? qnan<T>() : a;')
_if_else = _ew('if_else', 'T c, T a, T b', 'o = isnan(c) ? qnan<T>() : ((c > 0) ? a : b);')
# A zero denominator is undefined (NaN), never +-inf, like the CPU kernels.
_safe_div = _ew('safe_div', 'T n, T d', 'o = (fabs((double)d) < 1e-12) ? qnan<T>() : (T)(n / d);')
# A flat window is the middle of the channel (0.5) on the CPU; NaN must be propagated explicitly, because
# (hi > lo) is false for NaN bounds and would turn the whole warm-up into flat windows.
_chanpos = _ew('chanpos', 'T x, T lo, T hi',
               'o = (isnan(x) || isnan(lo) || isnan(hi)) ? qnan<T>() : ((hi > lo) ? (T)((x - lo) / (hi - lo)) : (T)0.5);')
# Undefined when the rolling maximum is <= 0 (CPU: (mx > 0) ? v/mx - 1 : NaN); signed inputs such as ma_dist_*
# have negative maxima whenever a whole window sits below its mean, so this is a domain, not a precision, matter.
_ddown = _ew('ddown', 'T x, T mx', 'o = (mx > (T)0) ? (T)((x - mx) / mx) : qnan<T>();')
_scale = _ew('scale_fin', 'T x, T lo, T hi, float64 k',
             'T d = hi - lo; o = (fabs((double)d) < 1e-12) ? qnan<T>() : (T)((x - lo) / d + k);')
_avd_pre = cp.ElementwiseKernel('T x', 'T xz, T fm', 'bool f = isfinite(x); xz = f ? x : (T)0; fm = (T)(f ? 1 : 0);', 'fx_avd_pre')
_avd_post = _ew('avd_post', 'T x, T s, T n', 'o = (isfinite(x) && n > 0) ? (T)(x - s / n) : qnan<T>();')


def _pop_std(x, w):
    """Population standard deviation (ts_zscore / ts_snr use ss/w on the CPU; ts_std is the sample one)."""
    return ts_std(x, w) * D(((w - 1) / w) ** 0.5)


def ts_zscore(x, w): return _safe_div(x - ts_mean(x, w), _pop_std(x, w))
def ts_snr(x, w): return _safe_div(ts_mean(x, w), _pop_std(x, w))
def ts_chanpos(x, w): return _chanpos(x, ts_min(x, w), ts_max(x, w))
def ts_ddown(x, w): return _ddown(x, ts_max(x, w))
def ts_scale(x, w, k): return _scale(x, ts_min(x, w), ts_max(x, w), float(k))


def ts_av_diff(x, w):
    """x - mean(x, w) with the mean ignoring NaN: sums of zero-filled values and of finiteness flags."""
    xz, fm = _avd_pre(x)
    return _avd_post(x, ts_sum(xz, w), ts_sum(fm, w))


def ts_quantile(x, w, driver):
    """Plotting position p = (r + 0.5)/w of the rank within the window, through the inverse CDF of the driver."""
    from cupyx.scipy.special import ndtri
    p = (ts_rank(x, w) * (w - 1.0) + 0.5) / float(w)
    code = int(round(float(driver)))
    if code == 1:
        return p.astype(D)
    if code == 2:
        return cp.tan(cp.pi * (p - 0.5)).astype(D)
    return ndtri(p.astype(cp.float64)).astype(D)


def ts_regression(y, x, w, rettype):
    """Rolling OLS y ~ a + b x from rolling sums; a NaN on either side voids both sums."""
    m = cp.isfinite(x) & cp.isfinite(y)
    xm = cp.where(m, x, cp.nan).astype(D)
    ym = cp.where(m, y, cp.nan).astype(D)
    n = float(w)
    Sx, Sy = ts_sum(xm, w), ts_sum(ym, w)
    Sxx, Sxy = ts_sum(xm * xm, w), ts_sum(xm * ym, w)
    dxx = n * Sxx - Sx * Sx
    beta = _safe_div(n * Sxy - Sx * Sy, dxx)
    alpha = (Sy - beta * Sx) / n
    code = int(round(float(rettype)))
    if code == 1:
        return alpha.astype(D)
    if code == 2:
        return beta.astype(D)
    fit = alpha + beta * x
    if code == 3:
        return fit.astype(D)
    if code == 4:
        Syy = ts_sum(ym * ym, w)
        cov = n * Sxy - Sx * Sy
        return _safe_div(cov * cov, dxx * (n * Syy - Sy * Sy)).astype(D)
    return (y - fit).astype(D)


# ---------------------------------------------------------------- path-dependent (one thread per column)
def _recursive(x, y, z, mode, window=0, k=0.):
    T, S = x.shape
    out = cp.empty_like(x)
    _PATHS.get_function('k_recursive')(((S + 127) // 128,), (128,),
                                       (x, y, z, out, np.int32(T), np.int32(S), np.int32(mode), np.int32(window), np.float64(k)))
    return out


def trade_when(entry, alpha, exit_): return _recursive(entry, alpha, exit_, 0)
def hump(x, k): return _recursive(x, x, x, 1, k=k)
def ts_backfill(x, w): return _recursive(x, x, x, 2, window=w)


def ts_decay_exp_window(x, w, factor):
    T, S = x.shape
    out = cp.empty_like(x)
    _PATHS.get_function('k_decay')(((x.size + 255) // 256,), (256,), (x, out, np.int32(T), np.int32(S), np.int32(w), np.float64(factor)))
    return out


def ts_median(x, w):
    """Window sort in bounded chunks of days; a window with a NaN is NaN."""
    T, S = x.shape
    out = cp.full_like(x, cp.nan)
    if T < w:
        return out
    view = cp.lib.stride_tricks.sliding_window_view(x, w, axis=0)
    for lo in range(0, T - w + 1, 128):
        v = view[lo:lo + 128]
        s = cp.sort(v, axis=2)
        mid = s[:, :, w // 2] if w % 2 else (s[:, :, w // 2] + s[:, :, w // 2 - 1]) * 0.5
        out[lo + w - 1:lo + w - 1 + len(v)] = cp.where(cp.isnan(v).any(2), cp.nan, mid)
    return out


IMPL = dict(
    abs=cp.abs, neg=lambda x: -x, sign=_sign, log=_log, square=lambda x: x * x, sqrt=_sqrt, tanh=cp.tanh, inv=_inv,
    s_log_1p=_s_log_1p, **{'not': _not}, is_nan=_is_nan,
    add=lambda a, b: a + b, sub=lambda a, b: a - b, mul=lambda a, b: a * b, div=_div,
    max=cp.maximum, min=cp.minimum, gt=_pred('gt', 'a > b'), lt=_pred('lt', 'a < b'),
    **{'and': _pred('and', '(a > 0) && (b > 0)'), 'or': _pred('or', '(a > 0) || (b > 0)')},
    equal=_pred('equal', 'a == b'), not_equal=_pred('not_equal', 'a != b'),
    greater_equal=_pred('ge', 'a >= b'), less_equal=_pred('le', 'a <= b'),
    add_const=lambda x, k: x + D(k), mul_const=lambda x, k: x * D(k), pow_const=lambda x, k: _pow(x, D(k)),
    left_tail=lambda x, k: _left_tail(x, float(k)), right_tail=lambda x, k: _right_tail(x, float(k)), hump=hump,
    if_else=_if_else, trade_when=trade_when,
    cs_rank=cs_rank, cs_zscore=lambda x: _cs(x, 0), cs_demean=lambda x: _cs(x, 1), cs_scale=lambda x: _cs(x, 2),
    ts_mean=ts_mean, ts_std=ts_std, ts_max=ts_max, ts_min=ts_min, ts_sum=ts_sum, ts_rank=ts_rank,
    ts_arg_max=ts_arg_max, ts_arg_min=ts_arg_min, ts_ema=ts_ema, ts_wma=ts_wma, ts_ref=ts_ref, ts_delta=ts_delta,
    ts_skew=ts_skew, ts_kurt=ts_kurt, ts_mad=ts_mad, ts_slope=ts_slope, ts_chanpos=ts_chanpos, ts_ddown=ts_ddown,
    ts_zscore=ts_zscore, ts_snr=ts_snr, ts_backfill=ts_backfill, ts_median=ts_median, ts_av_diff=ts_av_diff,
    ts_decay_exp_window=ts_decay_exp_window, ts_scale=ts_scale, ts_quantile=ts_quantile,
    ts_corr=ts_corr, ts_cov=ts_cov, ts_regression=ts_regression,
)
assert set(IMPL) == set(OPS), set(OPS) ^ set(IMPL)
