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
D = cp.float64
_ELEM = 8
_NAN = '\ntemplate<typename T> __device__ __forceinline__ T qnan();\ntemplate<> __device__ __forceinline__ double qnan<double>() { return __longlong_as_double((long long)0x7ff8000000000000ULL); }\n'

class Ops:

    def __init__(self):
        self._OPS = cp.RawModule(code=(_KERNELS / 'ops.cu').read_text(encoding='utf-8'), options=('--std=c++14',))
        self._PATHS = cp.RawModule(code=(_KERNELS / 'paths.cu').read_text(encoding='utf-8'), options=('--std=c++14', '--fmad=false'))
        self._PROPS = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
        self._WARP = self._PROPS['warpSize']
        self._OPTIN = self._PROPS['sharedMemPerBlockOptin']
        self._MAXTHR = self._PROPS['maxThreadsPerBlock']
        self._SHBUDGET = min(self._OPTIN, self._PROPS['sharedMemPerMultiprocessor'] // 3)
        self._k_moment, self._k_ext, self._k_pair, self._k_ema, self._k_cs, self._k_cs_rank = (self._OPS.get_function(n) for n in ('k_moment', 'k_ext', 'k_pair', 'k_ema', 'k_cs', 'k_cs_rank'))
        self._DYNMAX = min((self._optin_max(f) for f in (self._k_moment, self._k_ext, self._k_pair, self._k_cs_rank)))
        self._DUMMY = cp.zeros(1, dtype=D)
        self.max_window = self._max_window(1)
        self.max_pair_window = self._max_window(2)
        self._sign = cp.ElementwiseKernel('T x', 'T o', 'o = isnan(x) ? x : (x > (T)0 ? (T)1 : (x < (T)0 ? (T)-1 : (T)0));', 'fm_sign')
        self._log = cp.ElementwiseKernel('T x', 'T o', 'o = (isnan(x) || x <= (T)0) ? (T)nan("") : log(x);', 'fm_log')
        self._sqrt = cp.ElementwiseKernel('T x', 'T o', 'o = (isnan(x) || x < (T)0) ? (T)nan("") : sqrt(x);', 'fm_sqrt')
        self._inv = cp.ElementwiseKernel('T x', 'T o', 'o = (isnan(x) || x == (T)0) ? (T)nan("") : (T)1 / x;', 'fm_inv')
        self._s_log_1p = cp.ElementwiseKernel('T x', 'T o', 'o = isnan(x) ? x : (x >= (T)0 ? log1p(x) : -log1p(-x));', 'fm_s_log_1p')
        self._div = cp.ElementwiseKernel('T a, T b', 'T o', 'o = (b == (T)0) ? (T)nan("") : a / b;', 'fm_div')
        self._pow = cp.ElementwiseKernel('T x, T k', 'T o', 'T av = fabs(x); if (av == (T)0) { o = (T)0; } else { T p = pow(av, k); o = (x < (T)0) ? -p : p; }', 'fm_pow_const')
        self._not = self._ew('not', 'T a', 'o = isnan(a) ? qnan<T>() : (T)((a > 0) ? 0 : 1);')
        self._is_nan = self._ew('is_nan', 'T a', 'o = (T)(isnan(a) ? 1 : 0);')
        self._left_tail = self._ew('left_tail', 'T a, float64 k', 'o = (a > (T)k) ? qnan<T>() : a;')
        self._right_tail = self._ew('right_tail', 'T a, float64 k', 'o = (a < (T)k) ? qnan<T>() : a;')
        self._if_else = self._ew('if_else', 'T c, T a, T b', 'o = isnan(c) ? qnan<T>() : ((c > 0) ? a : b);')
        self._safe_div = self._ew('safe_div', 'T n, T d', 'o = (fabs((double)d) < 1e-12) ? qnan<T>() : (T)(n / d);')
        self._chanpos = self._ew('chanpos', 'T x, T lo, T hi', 'o = (isnan(x) || isnan(lo) || isnan(hi)) ? qnan<T>() : ((hi > lo) ? (T)((x - lo) / (hi - lo)) : (T)0.5);')
        self._ddown = self._ew('ddown', 'T x, T mx', 'o = (mx > (T)0) ? (T)((x - mx) / mx) : qnan<T>();')
        self._scale = self._ew('scale_fin', 'T x, T lo, T hi, float64 k', 'T d = hi - lo; o = (fabs((double)d) < 1e-12) ? qnan<T>() : (T)((x - lo) / d + k);')
        self._avd_pre = cp.ElementwiseKernel('T x', 'T xz, T fm', 'bool f = isfinite(x); xz = f ? x : (T)0; fm = (T)(f ? 1 : 0);', 'fx_avd_pre')
        self._avd_post = self._ew('avd_post', 'T x, T s, T n', 'o = (isfinite(x) && n > 0) ? (T)(x - s / n) : qnan<T>();')
        self.IMPL = dict(abs=cp.abs, neg=lambda x: -x, sign=self._sign, log=self._log, square=lambda x: x * x, sqrt=self._sqrt, tanh=cp.tanh, inv=self._inv, s_log_1p=self._s_log_1p, **{'not': self._not}, is_nan=self._is_nan, add=lambda a, b: a + b, sub=lambda a, b: a - b, mul=lambda a, b: a * b, div=self._div, max=cp.maximum, min=cp.minimum, gt=self._pred('gt', 'a > b'), lt=self._pred('lt', 'a < b'), **{'and': self._pred('and', '(a > 0) && (b > 0)'), 'or': self._pred('or', '(a > 0) || (b > 0)')}, equal=self._pred('equal', 'a == b'), not_equal=self._pred('not_equal', 'a != b'), greater_equal=self._pred('ge', 'a >= b'), less_equal=self._pred('le', 'a <= b'), add_const=lambda x, k: x + D(k), mul_const=lambda x, k: x * D(k), pow_const=lambda x, k: self._pow(x, D(k)), left_tail=lambda x, k: self._left_tail(x, float(k)), right_tail=lambda x, k: self._right_tail(x, float(k)), hump=self.hump, if_else=self._if_else, trade_when=self.trade_when, cs_rank=self.cs_rank, cs_zscore=lambda x: self._cs(x, 0), cs_demean=lambda x: self._cs(x, 1), cs_scale=lambda x: self._cs(x, 2), ts_mean=self.ts_mean, ts_std=self.ts_std, ts_max=self.ts_max, ts_min=self.ts_min, ts_sum=self.ts_sum, ts_rank=self.ts_rank, ts_arg_max=self.ts_arg_max, ts_arg_min=self.ts_arg_min, ts_ema=self.ts_ema, ts_wma=self.ts_wma, ts_ref=self.ts_ref, ts_delta=self.ts_delta, ts_skew=self.ts_skew, ts_kurt=self.ts_kurt, ts_mad=self.ts_mad, ts_slope=self.ts_slope, ts_chanpos=self.ts_chanpos, ts_ddown=self.ts_ddown, ts_zscore=self.ts_zscore, ts_snr=self.ts_snr, ts_backfill=self.ts_backfill, ts_median=self.ts_median, ts_av_diff=self.ts_av_diff, ts_decay_exp_window=self.ts_decay_exp_window, ts_scale=self.ts_scale, ts_quantile=self.ts_quantile, ts_corr=self.ts_corr, ts_cov=self.ts_cov, ts_regression=self.ts_regression)
        assert set(self.IMPL) == set(OPS), set(OPS) ^ set(self.IMPL)

    def _optin_max(self, fn):
        cap = self._OPTIN - int(fn.shared_size_bytes) - 1024
        fn.max_dynamic_shared_size_bytes = cap
        return cap

    def _candidate_tiles(self, w, budget, mult=1):
        """(assets, days) tiles whose (days + w - 1) x assets halo fits `budget`; larger tiles (more reuse) first."""
        out = [(ss, tt) for ss in (8, 16, 32) for tt in (256, 192, 128, 96, 64, 48, 32, 16) if ss * tt <= self._MAXTHR and ss * tt % self._WARP == 0 and ((tt + w - 1) * ss * _ELEM * mult <= budget)]
        return sorted(out, key=lambda st: -st[0] * st[1])

    def _tile(self, w, mult=1):
        return (self._candidate_tiles(w, self._SHBUDGET, mult) or self._candidate_tiles(w, self._DYNMAX, mult))[0]

    def _max_window(self, mult) -> int:
        lo, hi = (2, 1 << 16)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            lo, hi = (mid, hi) if self._candidate_tiles(mid, self._DYNMAX, mult) else (lo, mid - 1)
        return lo

    def window_cap(self, op: str) -> int:
        return self.max_pair_window if OPS[op].kind == 'pair' else self.max_window

    def unsupported(self, tree):
        """Reasons this tree cannot run on this card: unknown operators or windows above the card's tile limit."""
        out = []
        for n in tree.root.walk():
            if n.leaf:
                continue
            if n.op not in self.IMPL:
                out.append(n.op)
            elif n.window is not None and n.window > self.window_cap(n.op):
                out.append(f'{n.op}@w={n.window}>{self.window_cap(n.op)}')
        return out

    def _moment(self, x, w, op, kmode, K):
        T, S = x.shape
        out = cp.empty_like(x)
        ss, tt = self._tile(w)
        self._k_moment(((S + ss - 1) // ss, (T + tt - 1) // tt), (ss, tt), (x, out, np.int32(T), np.int32(S), np.int32(w), np.int32(op), K, np.int32(kmode)), shared_mem=(tt + w - 1) * ss * _ELEM)
        return out

    def _first_valid(self, x):
        """First valid value of every column (skew/kurt centering anchor); an all-NaN column -> 0."""
        fin = ~cp.isnan(x)
        return cp.where(fin.any(axis=0), x[fin.argmax(axis=0), cp.arange(x.shape[1])], 0).astype(D)

    def _ext(self, x, w, op):
        T, S = x.shape
        out = cp.empty_like(x)
        ss, tt = self._tile(w)
        self._k_ext(((S + ss - 1) // ss, (T + tt - 1) // tt), (ss, tt), (x, out, np.int32(T), np.int32(S), np.int32(w), np.int32(op)), shared_mem=(tt + w - 1) * ss * _ELEM)
        return out

    def _pair(self, a, b, w, op):
        T, S = a.shape
        out = cp.empty_like(a)
        ss, tt = self._tile(w, 2)
        self._k_pair(((S + ss - 1) // ss, (T + tt - 1) // tt), (ss, tt), (a, b, out, np.int32(T), np.int32(S), np.int32(w), np.int32(op)), shared_mem=2 * (tt + w - 1) * ss * _ELEM)
        return out

    def ts_sum(self, x, w):
        return self._moment(x, w, 0, 0, self._DUMMY)

    def ts_mean(self, x, w):
        return self._moment(x, w, 1, 0, self._DUMMY)

    def ts_std(self, x, w):
        return self._moment(x, w, 2, 0, self._DUMMY)

    def ts_mad(self, x, w):
        return self._moment(x, w, 5, 0, self._DUMMY)

    def ts_skew(self, x, w):
        return self._moment(x, w, 3, 1, self._first_valid(x))

    def ts_kurt(self, x, w):
        return self._moment(x, w, 4, 1, self._first_valid(x))

    def ts_max(self, x, w):
        return self._ext(x, w, 0)

    def ts_min(self, x, w):
        return self._ext(x, w, 1)

    def ts_wma(self, x, w):
        return self._ext(x, w, 2)

    def ts_slope(self, x, w):
        return self._ext(x, w, 3)

    def ts_rank(self, x, w):
        return self._ext(x, w, 4)

    def ts_arg_max(self, x, w):
        return self._ext(x, w, 5)

    def ts_arg_min(self, x, w):
        return self._ext(x, w, 6)

    def ts_corr(self, a, b, w):
        return self._pair(a, b, w, 0)

    def ts_cov(self, a, b, w):
        return self._pair(a, b, w, 1)

    def ts_ema(self, x, span):
        T, S = x.shape
        out = cp.empty_like(x)
        self._k_ema(((S + 127) // 128,), (128,), (x, out, np.int32(T), np.int32(S), np.int32(span)))
        return out

    def ts_ref(self, x, w):
        out = cp.full_like(x, cp.nan)
        if w < x.shape[0]:
            out[w:] = x[:-w]
        return out

    def ts_delta(self, x, w):
        return x - self.ts_ref(x, w)

    def _cs(self, x, op):
        T, S = x.shape
        out = cp.empty_like(x)
        self._k_cs((T,), (256,), (x, out, np.int32(S), np.int32(op)))
        return out

    def cs_rank(self, x):
        """Per-day average rank in [0,1], O(S^2) per day with the row in shared memory; a day with < 2 values is NaN.
    (A bitonic sort was 8-12x slower on every real universe, 299-666 assets: its barriers cost more than the count.)"""
        T, S = x.shape
        out = cp.empty_like(x)
        self._k_cs_rank((T,), (256,), (x, out, np.int32(S)), shared_mem=S * _ELEM)
        return out

    def _ew(self, name, params, body):
        return cp.ElementwiseKernel(params, 'T o', body, f'fx_{name}', preamble=_NAN)

    def _pred(self, name, cond):
        """Binary predicate: NaN on either side -> NaN, else 1/0 (the CPU FM_BIN_NAN semantics)."""
        return self._ew(name, 'T a, T b', f'o = (isnan(a) || isnan(b)) ? qnan<T>() : (T)(({cond}) ? 1 : 0);')

    def _pop_std(self, x, w):
        """Population standard deviation (ts_zscore / ts_snr use ss/w on the CPU; ts_std is the sample one)."""
        return self.ts_std(x, w) * D(((w - 1) / w) ** 0.5)

    def ts_zscore(self, x, w):
        return self._safe_div(x - self.ts_mean(x, w), self._pop_std(x, w))

    def ts_snr(self, x, w):
        return self._safe_div(self.ts_mean(x, w), self._pop_std(x, w))

    def ts_chanpos(self, x, w):
        return self._chanpos(x, self.ts_min(x, w), self.ts_max(x, w))

    def ts_ddown(self, x, w):
        return self._ddown(x, self.ts_max(x, w))

    def ts_scale(self, x, w, k):
        return self._scale(x, self.ts_min(x, w), self.ts_max(x, w), float(k))

    def ts_av_diff(self, x, w):
        """x - mean(x, w) with the mean ignoring NaN: sums of zero-filled values and of finiteness flags."""
        xz, fm = self._avd_pre(x)
        return self._avd_post(x, self.ts_sum(xz, w), self.ts_sum(fm, w))

    def ts_quantile(self, x, w, driver):
        """Plotting position p = (r + 0.5)/w of the rank within the window, through the inverse CDF of the driver."""
        from cupyx.scipy.special import ndtri
        p = (self.ts_rank(x, w) * (w - 1.0) + 0.5) / float(w)
        code = int(round(float(driver)))
        if code == 1:
            return p.astype(D)
        if code == 2:
            return cp.tan(cp.pi * (p - 0.5)).astype(D)
        return ndtri(p.astype(cp.float64)).astype(D)

    def ts_regression(self, y, x, w, rettype):
        """Rolling OLS y ~ a + b x from rolling sums; a NaN on either side voids both sums."""
        m = cp.isfinite(x) & cp.isfinite(y)
        xm = cp.where(m, x, cp.nan).astype(D)
        ym = cp.where(m, y, cp.nan).astype(D)
        n = float(w)
        Sx, Sy = (self.ts_sum(xm, w), self.ts_sum(ym, w))
        Sxx, Sxy = (self.ts_sum(xm * xm, w), self.ts_sum(xm * ym, w))
        dxx = n * Sxx - Sx * Sx
        beta = self._safe_div(n * Sxy - Sx * Sy, dxx)
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
            Syy = self.ts_sum(ym * ym, w)
            cov = n * Sxy - Sx * Sy
            return self._safe_div(cov * cov, dxx * (n * Syy - Sy * Sy)).astype(D)
        return (y - fit).astype(D)

    def _recursive(self, x, y, z, mode, window=0, k=0.0):
        T, S = x.shape
        out = cp.empty_like(x)
        self._PATHS.get_function('k_recursive')(((S + 127) // 128,), (128,), (x, y, z, out, np.int32(T), np.int32(S), np.int32(mode), np.int32(window), np.float64(k)))
        return out

    def trade_when(self, entry, alpha, exit_):
        return self._recursive(entry, alpha, exit_, 0)

    def hump(self, x, k):
        return self._recursive(x, x, x, 1, k=k)

    def ts_backfill(self, x, w):
        return self._recursive(x, x, x, 2, window=w)

    def ts_decay_exp_window(self, x, w, factor):
        T, S = x.shape
        out = cp.empty_like(x)
        self._PATHS.get_function('k_decay')(((x.size + 255) // 256,), (256,), (x, out, np.int32(T), np.int32(S), np.int32(w), np.float64(factor)))
        return out

    def ts_median(self, x, w):
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

    def close(self):
        self.IMPL.clear()
        self.__dict__.clear()

def prepare():
    return Ops()
