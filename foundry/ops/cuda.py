"""CUDA backend of the formula operators: cupy arrays in, cupy arrays out, always float64.

Window-local statistics use shared-memory tiles. Rolling std/slope/correlation retain the native reference's
chronological recurrences, and population standardization uses its direct two-pass variance. `IMPL` maps every registry name to its
implementation (signatures per kind, see foundry.ops.registry). Launch geometry is derived from the device
properties, so the same code adapts to a 4 GB consumer card and a data-centre card; a window whose tile does not fit
the card's shared memory is reported by `unsupported` before anything is launched.
"""
from __future__ import annotations
from pathlib import Path
import cupy as cp
import numpy as np
from foundry.ops.registry import OPS, MAX_WINDOW
_KERNELS = Path(__file__).with_name('kernels')
D = cp.float64
_ELEM = 8
_NAN = '\ntemplate<typename T> __device__ __forceinline__ T qnan();\ntemplate<> __device__ __forceinline__ double qnan<double>() { return __longlong_as_double((long long)0x7ff8000000000000ULL); }\n'

class Ops:

    def __init__(self):
        contract = _KERNELS.parent.joinpath('two_point.hpp').read_text(encoding='utf-8')
        self._OPS = cp.RawModule(code=contract+'\n'+'\n'.join((_KERNELS / name).read_text(encoding='utf-8') for name in ('ops.cu','select.cu')), options=('--std=c++14', '--fmad=false'))
        self._PATHS = cp.RawModule(code=contract+'\n'+'\n'.join((_KERNELS / n).read_text(encoding='utf-8') for n in ('paths.cu', 'rolling.cu')), options=('--std=c++14', '--fmad=false'))
        self._PROPS = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
        self._WARP = self._PROPS['warpSize']
        self._OPTIN = self._PROPS['sharedMemPerBlockOptin']
        self._MAXTHR = self._PROPS['maxThreadsPerBlock']
        self._SHBUDGET = min(self._OPTIN, self._PROPS['sharedMemPerMultiprocessor'] // 3)
        self._k_moment, self._k_ext, self._k_ema, self._k_cs, self._k_cs_rank = (self._OPS.get_function(n) for n in ('k_moment', 'k_ext', 'k_ema', 'k_cs', 'k_cs_rank'))
        self._k_pair, self._k_standardize, self._k_std = (self._PATHS.get_function(n) for n in ('k_pair', 'k_standardize', 'k_std'))
        self._staged = {name: self._PATHS.get_function(f'k_{name}_staged') for name in ('slope', 'av_diff')}
        self._history_capacity = min(self._SHBUDGET, *(self._optin_max(fn) for fn in self._staged.values()))
        self._pack_history = self._PATHS.get_function('k_history_pack')
        self._finish_history = self._PATHS.get_function('k_history_finish')
        self._median = {size:self._OPS.get_function(f'k_median{size}') for size in (32,128,512,2048)}
        self._DYNMAX = min((self._optin_max(f) for f in (self._k_moment, self._k_ext, self._k_cs_rank)))
        self.max_window = self._max_window()
        self.max_pair_window = MAX_WINDOW
        self._sign = cp.ElementwiseKernel('T x', 'T o', 'o = isnan(x) ? x : (x > (T)0 ? (T)1 : (x < (T)0 ? (T)-1 : (T)0));', 'fm_sign')
        self._log = cp.ElementwiseKernel('T x', 'T o', 'o = (isnan(x) || x <= (T)0) ? (T)nan("") : log(x);', 'fm_log')
        self._sqrt = cp.ElementwiseKernel('T x', 'T o', 'o = (isnan(x) || x < (T)0) ? (T)nan("") : sqrt(x);', 'fm_sqrt')
        # Below 2^-28 the cubic correction is smaller than half an ulp for
        # binary64. The correctly rounded result is x, including subnormals.
        self._tanh = cp.ElementwiseKernel('T x', 'T o',
            'o = fabs(x) < (T)3.7252902984619140625e-9 ? x : tanh(x);', 'fm_tanh')
        self._inv = cp.ElementwiseKernel('T x', 'T o', 'o = (isnan(x) || x == (T)0) ? (T)nan("") : (T)1 / x;', 'fm_inv')
        self._s_log_1p = cp.ElementwiseKernel('T x', 'T o', 'o = isnan(x) ? x : (x >= (T)0 ? log1p(x) : -log1p(-x));', 'fm_s_log_1p')
        self._div = cp.ElementwiseKernel('T a, T b', 'T o', 'o = (b == (T)0) ? (T)nan("") : a / b;', 'fm_div')
        self._pow = cp.ElementwiseKernel('T x, T k', 'T o', 'T av = fabs(x); if (av == (T)0) { o = (T)0; } else { T p = pow(av, k); o = (x < (T)0) ? -p : p; }', 'fm_pow_const')
        self._not = self._ew('not', 'T a', 'o = isnan(a) ? qnan<T>() : (T)((a > 0) ? 0 : 1);')
        self._is_nan = self._ew('is_nan', 'T a', 'o = (T)(isnan(a) ? 1 : 0);')
        self._left_tail = self._ew('left_tail', 'T a, float64 k', 'o = (a > (T)k) ? qnan<T>() : a;')
        self._right_tail = self._ew('right_tail', 'T a, float64 k', 'o = (a < (T)k) ? qnan<T>() : a;')
        self._if_else = self._ew('if_else', 'T c, T a, T b', 'o = isnan(c) ? qnan<T>() : ((c > 0) ? a : b);')
        self._shift = cp.ElementwiseKernel('raw T x, int64 assets, int64 lag, bool difference', 'T o',
            'o = i < lag * assets ? (T)nan("") : (difference ? x[i] - x[i-lag*assets] : x[i-lag*assets]);',
            'fm_shift')
        self._backfill = cp.ElementwiseKernel('raw T x, int64 assets, int64 window', 'T o',
            'o = (T)nan(""); for (long long j=i, stop=max((long long)(i % assets), '
            '(long long)i-(window-1)*assets); j>=stop; j-=assets) { '
            'T v=x[j]; if (!isnan(v)) { o=v; break; } }', 'fm_backfill')
        self.IMPL = dict(abs=cp.abs, neg=lambda x: -x, sign=self._sign, log=self._log, square=lambda x: x * x, sqrt=self._sqrt, tanh=self._tanh, inv=self._inv, s_log_1p=self._s_log_1p, **{'not': self._not}, is_nan=self._is_nan, add=lambda a, b: a + b, sub=lambda a, b: a - b, mul=lambda a, b: a * b, div=self._div, max=cp.maximum, min=cp.minimum, gt=self._pred('gt', 'a > b'), lt=self._pred('lt', 'a < b'), **{'and': self._pred('and', '(a > 0) && (b > 0)'), 'or': self._pred('or', '(a > 0) || (b > 0)')}, equal=self._pred('equal', 'a == b'), not_equal=self._pred('not_equal', 'a != b'), greater_equal=self._pred('ge', 'a >= b'), less_equal=self._pred('le', 'a <= b'), add_const=lambda x, k: x + D(k), mul_const=lambda x, k: x * D(k), pow_const=lambda x, k: self._pow(x, D(k)), left_tail=lambda x, k: self._left_tail(x, float(k)), right_tail=lambda x, k: self._right_tail(x, float(k)), hump=self.hump, if_else=self._if_else, trade_when=self.trade_when, cs_rank=self.cs_rank, cs_zscore=lambda x: self._cs(x, 0), cs_demean=lambda x: self._cs(x, 1), cs_scale=lambda x: self._cs(x, 2), ts_mean=self.ts_mean, ts_std=self.ts_std, ts_max=self.ts_max, ts_min=self.ts_min, ts_sum=self.ts_sum, ts_rank=self.ts_rank, ts_arg_max=self.ts_arg_max, ts_arg_min=self.ts_arg_min, ts_ema=self.ts_ema, ts_wma=self.ts_wma, ts_ref=self.ts_ref, ts_delta=self.ts_delta, ts_skew=self.ts_skew, ts_kurt=self.ts_kurt, ts_mad=self.ts_mad, ts_slope=self.ts_slope, ts_chanpos=self.ts_chanpos, ts_ddown=self.ts_ddown, ts_zscore=self.ts_zscore, ts_snr=self.ts_snr, ts_backfill=self.ts_backfill, ts_median=self.ts_median, ts_av_diff=self.ts_av_diff, ts_decay_exp_window=self.ts_decay_exp_window, ts_scale=self.ts_scale, ts_quantile=self.ts_quantile, ts_corr=self.ts_corr, ts_cov=self.ts_cov, ts_regression=self.ts_regression)
        assert set(self.IMPL) == set(OPS), set(OPS) ^ set(self.IMPL)

    def _optin_max(self, fn):
        cap = self._OPTIN - int(fn.shared_size_bytes) - 1024
        fn.max_dynamic_shared_size_bytes = cap
        return cap

    def _candidate_tiles(self, w, budget):
        """(assets, days) tiles whose (days + w - 1) x assets halo fits `budget`; larger tiles (more reuse) first."""
        out = [(ss, tt) for ss in (8, 16, 32) for tt in (256, 192, 128, 96, 64, 48, 32, 16) if ss * tt <= self._MAXTHR and ss * tt % self._WARP == 0 and ((tt + w - 1) * ss * _ELEM <= budget)]
        return sorted(out, key=lambda st: -st[0] * st[1])

    def _tile(self, w):
        return (self._candidate_tiles(w, self._SHBUDGET) or self._candidate_tiles(w, self._DYNMAX))[0]

    def _max_window(self) -> int:
        lo, hi = (2, 1 << 16)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            lo, hi = (mid, hi) if self._candidate_tiles(mid, self._DYNMAX) else (lo, mid - 1)
        return lo

    def window_cap(self, op: str) -> int:
        if op=='ts_median':return max(self._median)
        return self.max_pair_window if OPS[op].kind == 'pair' else self.max_window

    def unsupported(self, tree):
        """Unknown operators or windows beyond the implementation's explicit capabilities."""
        out = []
        for n in tree.root.walk():
            if n.leaf:
                continue
            if n.op not in self.IMPL:
                out.append(n.op)
            elif n.window is not None and n.window > self.window_cap(n.op):
                out.append(f'{n.op}@w={n.window}>{self.window_cap(n.op)}')
        return out

    def _moment(self, x, w, op):
        T, S = x.shape
        out = cp.empty_like(x)
        ss, tt = self._tile(w)
        self._k_moment(((S + ss - 1) // ss, (T + tt - 1) // tt), (ss, tt), (x, out, np.int32(T), np.int32(S), np.int32(w), np.int32(op)), shared_mem=(tt + w - 1) * ss * _ELEM)
        return out

    def _ext(self, x, w, op):
        T, S = x.shape
        out = cp.empty_like(x)
        ss, tt = self._tile(w)
        self._k_ext(((S + ss - 1) // ss, (T + tt - 1) // tt), (ss, tt), (x, out, np.int32(T), np.int32(S), np.int32(w), np.int32(op)), shared_mem=(tt + w - 1) * ss * _ELEM)
        return out

    def _pair(self, a, b, w, op):
        T, S = a.shape
        out = cp.empty_like(a)
        if w==2:
            self._PATHS.get_function('k_pair2')(((T*S+127)//128,),(128,),(a,b,out,np.int32(T),np.int32(S),np.int32(op)))
            return out
        grid, tile = self._path_tiles(T, S)
        self._k_pair(grid, (self._WARP,), (a, b, out, np.int32(T), np.int32(S), np.int32(w), np.int32(op), tile))
        return out

    def _path_tiles(self, days, assets):
        """Bound prefix replay while exposing independent output tiles to the device."""
        columns = (assets + self._WARP - 1) // self._WARP
        tiles = min(16, max(1, (days + 31) // 32),
                    max(1, self._PROPS['multiProcessorCount'] * 4 // columns))
        tile = (days + tiles - 1) // tiles
        return (columns, tiles), np.int32(tile)

    def ts_sum(self, x, w):
        return self._moment(x, w, 0)

    def ts_mean(self, x, w):
        return self._moment(x, w, 1)

    def ts_std(self, x, w):
        T, S = x.shape
        out = cp.empty_like(x)
        grid, tile = self._path_tiles(T, S)
        self._k_std(grid, (self._WARP,), (x, out, np.int32(T), np.int32(S), np.int32(w), tile))
        return out

    def ts_mad(self, x, w):
        return self._moment(x, w, 5)

    def ts_skew(self, x, w):
        return self._moment(x, w, 3)

    def ts_kurt(self, x, w):
        return self._moment(x, w, 4)

    def ts_max(self, x, w):
        return self._ext(x, w, 0)

    def ts_min(self, x, w):
        return self._ext(x, w, 1)

    def ts_wma(self, x, w):
        return self._ext(x, w, 2)

    def _history(self, x, w, name):
        """Choose the storage plan before launch; arithmetic is shared by both plans."""
        T, S = x.shape
        out = cp.empty_like(x)
        counts = cp.empty(x.shape if name == 'av_diff' else (0,), dtype=cp.int32)
        pitch = (T + 1) // 2 * 2
        if pitch * 4 * _ELEM <= self._history_capacity:
            packed = cp.empty(((S + 3) // 4, pitch, 4), dtype=D)
            self._pack_history(((S + 31) // 32, (T + 31) // 32), (32, 8),
                (x, packed, np.int32(T), np.int32(S), np.int32(pitch)))
            tile = (T + 7) // 8
            self._staged[name](((S + 3) // 4,), (self._WARP,),
                (packed, out, counts, np.int32(T), np.int32(S), np.int32(w), np.int32(pitch), np.int32(tile)),
                shared_mem=pitch * 4 * _ELEM)
        else:
            grid, tile = self._path_tiles(T, S)
            self._PATHS.get_function(f'k_{name}')(grid, (self._WARP,),
                (x, out, counts, np.int32(T), np.int32(S), np.int32(w), tile))
        self._finish_history(((x.size + 255) // 256,), (256,),
            (x, out, counts, np.int32(x.size), np.int32(w), np.int32(name == 'slope')))
        return out

    def ts_slope(self, x, w):
        return self._history(x, w, 'slope')

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
        self._k_ema(((S + self._WARP - 1) // self._WARP,), (self._WARP,), (x, out, np.int32(T), np.int32(S), np.int32(span)))
        return out

    def ts_ref(self, x, w):
        return self._shift(x, x.shape[1], w, False, cp.empty_like(x))

    def ts_delta(self, x, w):
        return self._shift(x, x.shape[1], w, True, cp.empty_like(x))

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
        self._k_cs_rank((T,), (256,), (x, out, np.int32(S)), shared_mem=S * 8)
        return out

    def _ew(self, name, params, body):
        return cp.ElementwiseKernel(params, 'T o', body, f'fx_{name}', preamble=_NAN)

    def _pred(self, name, cond):
        """Binary predicate: NaN on either side -> NaN, else 1/0 (the CPU FM_BIN_NAN semantics)."""
        return self._ew(name, 'T a, T b', f'o = (isnan(a) || isnan(b)) ? qnan<T>() : (T)(({cond}) ? 1 : 0);')

    def _standardize(self, x, w, mode):
        T, S = x.shape
        out = cp.empty_like(x)
        self._k_standardize(((T * S + 127) // 128,), (128,),
                           (x, out, np.int32(T), np.int32(S), np.int32(w), np.int32(mode)))
        return out

    def ts_zscore(self, x, w):
        return self._standardize(x, w, 0)

    def ts_snr(self, x, w):
        return self._standardize(x, w, 1)

    def _range(self, x, w, mode, shift=0.):
        T, S = x.shape
        out = cp.empty_like(x)
        self._PATHS.get_function('k_range')(((T*S+127)//128,), (128,),
            (x, out, np.int32(T), np.int32(S), np.int32(w), np.int32(mode), np.float64(shift)))
        return out

    def ts_chanpos(self, x, w):
        return self._range(x, w, 0)

    def ts_ddown(self, x, w):
        return self._range(x, w, 1)

    def ts_scale(self, x, w, k):
        return self._range(x, w, 2, k)

    def ts_av_diff(self, x, w):
        return self._history(x, w, 'av_diff')

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
        T, S = x.shape
        out = cp.empty_like(x)
        self._PATHS.get_function('k_regression')(((T*S+127)//128,), (128,),
            (y, x, out, np.int32(T), np.int32(S), np.int32(w), np.int32(round(float(rettype)))))
        return out

    def _recursive(self, x, y, z, mode, k=0.0):
        T, S = x.shape
        out = cp.empty_like(x)
        self._PATHS.get_function('k_recursive')(((S + self._WARP - 1) // self._WARP,), (self._WARP,), (x, y, z, out, np.int32(T), np.int32(S), np.int32(mode), np.float64(k)))
        return out

    def trade_when(self, entry, alpha, exit_):
        return self._recursive(entry, alpha, exit_, 0)

    def hump(self, x, k):
        return self._recursive(x, x, x, 1, k=k)

    def ts_backfill(self, x, w):
        return self._backfill(x, x.shape[1], w, cp.empty_like(x))

    def ts_decay_exp_window(self, x, w, factor):
        T, S = x.shape
        out = cp.empty_like(x)
        self._PATHS.get_function('k_decay')(((x.size + 255) // 256,), (256,), (x, out, np.int32(T), np.int32(S), np.int32(w), np.float64(factor)))
        return out

    def ts_median(self, x, w):
        """Integer selection with one bounded scratch window per lane."""
        T, S = x.shape
        out = cp.empty_like(x)
        capacity=next(size for size in self._median if size>=w)
        self._median[capacity](((T*S+127)//128,),(128,),(x,out,np.int32(T),np.int32(S),np.int32(w)))
        return out

    def close(self):
        self.IMPL.clear()
        self.__dict__.clear()

def prepare():
    return Ops()
