"""Portable operator backend: numpy only, for hosts without the native extensions (a hosted notebook).

`IMPL` maps registry names to callables with the signature of their kind (foundry/ops/registry.py). The window rule
requires rows t-w+1..t all non-NaN, so the first w-1 rows are NaN. Sum/mean share CUDA's oldest-anchored accumulation
and FMA restoration so near-zero comparisons preserve their decisions. Other rolling kernels follow the native
CPU implementation. Rolling kernels walk rows in Python and vectorise across assets; ranks count trailing values.

Usable without a native extension or a GPU.
"""
import math
from decimal import Decimal, localcontext

import numpy as np

NAN = np.nan


# ---------------------------------------------------------------- shared window machinery
def _nan_free(bad, w):
    """Rows t >= w-1 whose window [t-w+1, t] holds no `bad` cell."""
    c = np.cumsum(bad, axis=0, dtype=np.int64)
    n = c.copy()
    n[w:] -= c[:-w]
    ok = n == 0
    ok[:w - 1] = False
    return ok


def _running(cols, w):
    """Running window sums of (T,S) arrays whose NaN cells are already 0: add row t, then drop row t-w - the
    reference's order, so the rounding matches."""
    T, S = cols[0].shape
    out = [np.empty((T, S)) for _ in cols]
    acc = [np.zeros(S) for _ in cols]
    for t in range(T):
        for i, c in enumerate(cols):
            acc[i] = acc[i] + c[t]
            if t >= w:
                acc[i] = acc[i] - c[t - w]
            out[i][t] = acc[i]
    return out


def _rescan(arrays, w, fn):
    """Per-row window rescan: out[t] = fn(*windows) for t >= w-1, where each window is the list of w row blocks of one
    input, oldest first (block k holds rows k..k+n-1). Rows whose window holds NaN in any input are NaN."""
    T = len(arrays[0])
    n = T - w + 1
    out = np.full(arrays[0].shape, NAN)
    if n > 0:
        out[w - 1:] = fn(*[[a[k:k + n] for k in range(w)] for a in arrays])
    bad = np.isnan(arrays[0])
    for a in arrays[1:]:
        bad = bad | np.isnan(a)
    return np.where(_nan_free(bad, w), out, NAN)


def _seq(terms):
    """Sum in the given order (the reference loops over a window oldest row first)."""
    acc = 0.0
    for v in terms:
        acc = acc + v
    return acc


def _windows(x, w):
    """(T-w+1, S, w) view of the windows ending at rows w-1..T-1."""
    return np.lib.stride_tricks.sliding_window_view(x, w, axis=0)


def _extrema(x, w):
    """(min, max, window NaN-free) per row; min and max are exact selections, so every algorithm agrees bitwise."""
    lo, hi = np.full(x.shape, NAN), np.full(x.shape, NAN)
    if len(x) >= w:
        v = _windows(x, w)
        lo[w - 1:], hi[w - 1:] = v.min(-1), v.max(-1)
    return lo, hi, _nan_free(np.isnan(x), w)


# ---------------------------------------------------------------- element-wise
def _both(a, b):
    return np.isnan(a) | np.isnan(b)


def _flag(a, b, cond):
    """0/1 indicator with NaN propagation (gt, lt and the logical family)."""
    return np.where(_both(a, b), NAN, np.where(cond, 1.0, 0.0))


def sign(x):
    return np.where(np.isnan(x), NAN, np.where(x > 0, 1.0, np.where(x < 0, -1.0, 0.0)))


def log(x):
    return np.log(np.where(x > 0, x, NAN))


def sqrt(x):
    return np.sqrt(np.where(x >= 0, x, NAN))


def inv(x):
    return 1.0 / np.where(x == 0, NAN, x)


def div(a, b):
    return a / np.where(b == 0, NAN, b)


def pow_const(x, k):
    """sign(x)*|x|^k with 0 -> 0; the reference special-cases k = +-2, +-0.5."""
    av = np.abs(x)
    zero = av == 0
    a = np.where(zero, 1.0, av)
    if k == 2.0:
        p = a * a
    elif k == -2.0:
        p = 1.0 / (a * a)
    elif k == 0.5:
        p = np.sqrt(a)
    elif k == -0.5:
        p = 1.0 / np.sqrt(a)
    else:
        p = np.power(a, k)
    return np.where(zero, 0.0, np.where(x < 0, -p, p))


def hump(x, k):
    """Take x[t] only when it moved more than k from the held value; NaN never moves it."""
    out = np.empty(x.shape)
    prev = np.full(x.shape[1], NAN)
    for t in range(len(x)):
        v = x[t]
        prev = np.where(~np.isnan(v) & (np.isnan(prev) | (np.abs(v - prev) > k)), v, prev)
        out[t] = prev
    return out


def trade_when(entry, alpha, exit_):
    out = np.empty(alpha.shape)
    prev = np.zeros(alpha.shape[1])
    for t in range(len(alpha)):
        prev = np.where(exit_[t] > 0, 0.0, np.where(entry[t] > 0, alpha[t], prev))
        out[t] = prev
    return out


# ---------------------------------------------------------------- cross-sectional (one row = one day)
def _row_sums(cols):
    """Row sums over the assets in column order (the reference loops over s)."""
    return _seq(cols[:, j] for j in range(cols.shape[1]))


def cs_rank(x):
    """Average rank over the day's non-NaN values, scaled to [0,1]; fewer than 2 values -> NaN row."""
    valid = ~np.isnan(x)
    n = valid.sum(1, keepdims=True)
    less = (x[:, None, :] < x[:, :, None]).sum(-1)
    eq = (x[:, None, :] == x[:, :, None]).sum(-1)
    return np.where(valid & (n >= 2), (2 * less + eq - 1) / 2.0 / np.maximum(n - 1, 1), NAN)


def cs_zscore(x):
    """(x - mean) / sample std over the day; fewer than 2 values or zero variance -> NaN row."""
    valid = ~np.isnan(x)
    x0 = np.where(valid, x, 0.0)
    n = valid.sum(1)
    s, q = _row_sums(x0), _row_sums(x0 * x0)
    mean = s / np.maximum(n, 1)
    var = (q - mean * s) / np.maximum(n - 1, 1)
    ok = (n >= 2) & (var > 0)
    return np.where(ok[:, None] & valid, (x - mean[:, None]) / np.sqrt(np.where(ok, var, 1.0))[:, None], NAN)


def cs_demean(x):
    valid = ~np.isnan(x)
    n = valid.sum(1)
    mean = _row_sums(np.where(valid, x, 0.0)) / np.maximum(n, 1)
    return np.where((n > 0)[:, None] & valid, x - mean[:, None], NAN)


def cs_scale(x):
    """x / sum|x| over the day; a sum <= 1e-12 -> NaN row."""
    valid = ~np.isnan(x)
    total = _row_sums(np.where(valid, np.abs(x), 0.0))
    ok = total > 1e-12
    return np.where(ok[:, None] & valid, x / np.where(ok, total, 1.0)[:, None], NAN)


# ---------------------------------------------------------------- time series: canonical centered sums
def ts_sum(x, w):
    """CUDA's oldest-anchored window order and one-rounding anchor restoration.

    Decimal implements the one FMA per output without a platform-dependent
    long-double width or a native-extension requirement. This portable backend
    prioritizes identical threshold decisions over native backend throughput.
    """
    T, S = x.shape
    out = np.full(x.shape, NAN)
    if T < w:
        return out
    anchor = x[:T - w + 1]
    centered = np.zeros(anchor.shape)
    for k in range(w):
        centered += x[k:k + len(anchor)] - anchor
    valid = ~np.isnan(centered)
    with localcontext() as context:
        context.prec = 1100
        restored = np.fromiter((float(Decimal.from_float(float(a)) * w + Decimal.from_float(float(s)))
                                for a, s in zip(anchor[valid], centered[valid])), dtype=np.float64,
                               count=int(valid.sum()))
    result = out[w - 1:]
    result[valid] = restored
    return out


def ts_mean(x, w):
    return ts_sum(x, w) / w


def ts_std(x, w):
    """Sample std from running sum and sum of squares; a slightly negative variance (rounding) is 0."""
    nan = np.isnan(x)
    x0 = np.where(nan, 0.0, x)
    s, q = _running([x0, x0 * x0], w)
    mean = s / w
    return np.where(_nan_free(nan, w), np.sqrt(np.maximum((q - mean * s) / (w - 1), 0.0)), NAN)


def ts_av_diff(x, w):
    """x - the mean of the window's non-NaN values (NaN ignored, not propagated)."""
    nan = np.isnan(x)
    s, c = _running([np.where(nan, 0.0, x), np.where(nan, 0.0, 1.0)], w)
    return np.where((c > 0) & ~nan, x - s / np.where(c > 0, c, 1.0), NAN)


def ts_wma(x, w):
    """Linear weights 1..w (newest heaviest), bootstrapped over rows 0..w-1, then updated in O(1)."""
    T, S = x.shape
    out = np.full(x.shape, NAN)
    if T < w:
        return out
    nan = np.isnan(x)
    x0 = np.where(nan, 0.0, x)
    wsum = w * (w + 1.0) / 2.0
    s, ws = np.zeros(S), np.zeros(S)
    for i in range(w):
        s = s + x0[i]
        ws = ws + (i + 1.0) * x0[i]
    out[w - 1] = ws / wsum
    for t in range(w, T):
        ws = ws - s + w * x0[t]
        s = s - x0[t - w] + x0[t]
        out[t] = ws / wsum
    return np.where(_nan_free(nan, w), out, NAN)


def _slope_sums(x, rows, cols, w):
    """S_x, S_kx (k = 0..w-1 from rows[j]) and the NaN count of column cols[j], accumulated oldest first."""
    sx, skx, cnt = np.zeros(len(cols)), np.zeros(len(cols)), np.zeros(len(cols), np.int64)
    for k in range(w):
        v = x[rows + k, cols]
        nan = np.isnan(v)
        v = np.where(nan, 0.0, v)
        sx = sx + v
        skx = skx + k * v
        cnt = cnt + nan
    return sx, skx, cnt


def ts_slope(x, w):
    """OLS slope against the window position. Per column the sums start at its first non-NaN row, are set directly
    on the first full window and then updated in O(1) (re-summed every 65536 rows) - the reference's recurrence."""
    T, S = x.shape
    mean_t, var_t = (w - 1) / 2.0, float(w * w - 1) / 12.0
    valid = ~np.isnan(x)
    has = valid.any(0)
    vlo = np.where(has, valid.argmax(0), T)
    vhi = np.where(has, T - 1 - valid[::-1].argmax(0), -1)
    first = vlo + w - 1
    live = vhi - vlo + 1 >= w
    x0 = np.where(valid, x, 0.0)
    sx, skx, cnt = np.zeros(S), np.zeros(S), np.zeros(S, np.int64)
    c = np.flatnonzero(live)
    sx[c], skx[c], cnt[c] = _slope_sums(x, vlo[c], c, w)
    out = np.full(x.shape, NAN)
    for t in range(T):
        step = live & (t > first) & (t <= vhi)
        if step.any():
            skx = np.where(step, skx - sx + x0[t - w] + (w - 1) * x0[t], skx)
            sx = np.where(step, sx + x0[t] - x0[t - w], sx)
            cnt = np.where(step, cnt + ~valid[t] - ~valid[t - w], cnt)
            c = np.flatnonzero(step & (((t - vlo) & 65535) == 0))
            if len(c):
                sx[c], skx[c], cnt[c] = _slope_sums(x, np.full(len(c), t - w + 1), c, w)
        emit = (step | (live & (t == first))) & (cnt == 0)
        out[t] = np.where(emit, (skx - mean_t * sx) / w / var_t, NAN)
    return out


def ts_corr(a, b, w):
    """Pearson (ddof 1) from five running sums; a pair with either side NaN counts as missing."""
    bad = _both(a, b)
    a0, b0 = np.where(bad, 0.0, a), np.where(bad, 0.0, b)
    sa, sb, sab, saa, sbb = _running([a0, b0, a0 * b0, a0 * a0, b0 * b0], w)
    ma, mb = sa / w, sb / w
    cov = (sab - ma * sb) / (w - 1)
    va, vb = (saa - ma * sa) / (w - 1), (sbb - mb * sb) / (w - 1)
    ok = _nan_free(bad, w) & (va > 0) & (vb > 0)
    return np.where(ok, cov / np.sqrt(np.where(ok, va * vb, 1.0)), NAN)


def ts_cov(a, b, w):
    bad = _both(a, b)
    a0, b0 = np.where(bad, 0.0, a), np.where(bad, 0.0, b)
    sa, sb, sab = _running([a0, b0, a0 * b0], w)
    return np.where(_nan_free(bad, w), (sab - sa / w * sb) / (w - 1), NAN)


# ---------------------------------------------------------------- time series: recursions and shifts
def ts_ref(x, w):
    out = np.full(x.shape, NAN)
    out[w:] = x[:-w]
    return out


def ts_delta(x, w):
    return x - ts_ref(x, w)


def ts_ema(x, w):
    """ewm(span=w, adjust=False) seeded by the first value; a NaN row keeps the state and outputs NaN; fewer than w
    values seen -> NaN."""
    a = 2.0 / (w + 1)
    S = x.shape[1]
    out = np.empty(x.shape)
    ema, seeded, cnt = np.zeros(S), np.zeros(S, bool), np.zeros(S, np.int64)
    for t in range(len(x)):
        v = x[t]
        ok = ~np.isnan(v)
        ema = np.where(ok, np.where(seeded, a * v + (1.0 - a) * ema, v), ema)
        seeded |= ok
        cnt = cnt + ok
        out[t] = np.where(ok & (cnt >= w), ema, NAN)
    return out


def ts_backfill(x, w):
    """The last non-NaN value within the w-1 rows before (no full-window rule: filling NaN is its purpose)."""
    T, S = x.shape
    valid = ~np.isnan(x)
    rows = np.arange(T)[:, None]
    last = np.maximum.accumulate(np.where(valid, rows, -1), axis=0)
    fill = x[np.maximum(last, 0), np.arange(S)]
    return np.where(valid, x, np.where((last >= 0) & (rows - last <= w - 1), fill, NAN))


# ---------------------------------------------------------------- time series: window rescans
def _moments(w, powers):
    """fn for _rescan: the window mean (pass 1) and the central moment sums of `powers` (pass 2), each / w."""
    def fn(rows):
        mean = _seq(rows) / w
        m = [0.0] * len(powers)
        for r in rows:
            d = r - mean
            d2 = d * d
            for i, p in enumerate(powers):
                m[i] = m[i] + (d2 if p == 2 else d2 * d if p == 3 else d2 * d2)
        return mean, [v / w for v in m]
    return fn


def ts_skew(x, w):
    def fn(rows):
        _, (m2, m3) = _moments(w, (2, 3))(rows)
        pos = m2 > 0
        return np.where(pos, m3 / np.where(pos, m2, 1.0) ** 1.5, NAN)
    return _rescan([x], w, fn)


def ts_kurt(x, w):
    def fn(rows):
        _, (m2, m4) = _moments(w, (2, 4))(rows)
        pos = m2 > 0
        return np.where(pos, m4 / np.where(pos, m2 * m2, 1.0) - 3.0, NAN)
    return _rescan([x], w, fn)


def ts_zscore(x, w):
    """(x - mean) / population std; std <= 1e-12 -> NaN."""
    def fn(rows):
        mean, (ss,) = _moments(w, (2,))(rows)
        sd = np.sqrt(ss)
        ok = sd > 1e-12
        return np.where(ok, (rows[-1] - mean) / np.where(ok, sd, 1.0), NAN)
    return _rescan([x], w, fn)


def ts_snr(x, w):
    """mean / population std; std <= 1e-12 -> NaN."""
    def fn(rows):
        mean, (ss,) = _moments(w, (2,))(rows)
        sd = np.sqrt(ss)
        ok = sd > 1e-12
        return np.where(ok, mean / np.where(ok, sd, 1.0), NAN)
    return _rescan([x], w, fn)


def ts_mad(x, w):
    """Mean absolute deviation from the window mean of its own scalar running sum (not ts_mean's lane order)."""
    mean = _running([np.where(np.isnan(x), 0.0, x)], w)[0][w - 1:] / w
    return _rescan([x], w, lambda rows: _seq(np.abs(r - mean) for r in rows) / w)


def ts_decay_exp_window(x, w, f):
    """Weights f^i (i = 0 is the current row) normalised by their sum."""
    wt, p = [], 1.0
    for _ in range(w):
        wt.append(p)
        p *= f
    wsum = _seq(wt)
    return _rescan([x], w, lambda rows: _seq(wt[i] * rows[w - 1 - i] for i in range(w)) / wsum)


def _counts(rows):
    """(#values below the current one, #equal to it incl. itself) over the window."""
    cur = rows[-1]
    return _seq((r < cur).astype(np.int64) for r in rows), _seq((r == cur).astype(np.int64) for r in rows)


def ts_rank(x, w):
    """Average rank of the current value in its trailing window, in [0,1]."""
    def fn(rows):
        less, eq = _counts(rows)
        return (less + 0.5 * (eq - 1)) / (w - 1)
    return _rescan([x], w, fn)


_A = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02,
      -3.066479806614716e+01, 2.506628277459239e+00)
_B = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01,
      -1.328068155288572e+01)
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00,
      4.374664141464968e+00, 2.938163982698783e+00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00)
_erfc = np.vectorize(math.erfc, otypes=[np.float64])


def _ndtri(p):
    """Inverse normal CDF on (0,1): Acklam's rational approximation plus one Halley step (the reference's fm_ndtri)."""
    def tail(q):
        return ((((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5])
                / ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0))
    q = p - 0.5
    r = q * q
    mid = ((((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q
           / (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0))
    plow = 0.02425
    xv = np.where(p < plow, tail(np.sqrt(-2.0 * np.log(p))),
                  np.where(p > 1.0 - plow, -tail(np.sqrt(-2.0 * np.log(1.0 - p))), mid))
    e = 0.5 * _erfc(-xv / math.sqrt(2.0)) - p
    u = e * math.sqrt(2.0 * math.pi) * np.exp(xv * xv / 2.0)
    return xv - u / (1.0 + xv * u / 2.0)


def ts_quantile(x, w, driver):
    """Plotting position p = (average rank + 0.5) / w through the inverse CDF of `driver` (0 gaussian, 1 uniform,
    2 cauchy)."""
    dr = math.floor(driver + 0.5)

    def fn(rows):
        less, eq = _counts(rows)
        p = np.where(eq > 0, (less + 0.5 * (eq - 1) + 0.5) / w, NAN)      # eq == 0: the current value is NaN
        return p if dr == 1 else np.tan(math.pi * (p - 0.5)) if dr == 2 else _ndtri(p)
    return _rescan([x], w, fn)


def ts_regression(y, x, w, rettype):
    """OLS y ~ alpha + beta*x over the window, re-summed per row: 0 residual, 1 alpha, 2 beta, 3 fit, 4 R^2; x
    constant over the window -> NaN."""
    rt = math.floor(rettype + 0.5)

    def fn(ys, xs):
        sx = sy = sxx = sxy = syy = 0.0
        for yv, xv in zip(ys, xs):
            sx, sy = sx + xv, sy + yv
            sxx, sxy, syy = sxx + xv * xv, sxy + xv * yv, syy + yv * yv
        n = float(w)
        dxx = n * sxx - sx * sx
        ok = dxx != 0
        beta = (n * sxy - sx * sy) / np.where(ok, dxx, 1.0)
        alpha = (sy - beta * sx) / n
        fit = alpha + beta * xs[-1]
        if rt == 4:
            dyy = n * syy - sy * sy
            cov = n * sxy - sx * sy
            ok = ok & (dyy != 0)
            v = cov * cov / np.where(ok, dxx * dyy, 1.0)
        else:
            v = {1: alpha, 2: beta, 3: fit}.get(rt, ys[-1] - fit)
        return np.where(ok, v, NAN)
    return _rescan([y, x], w, fn)


# ---------------------------------------------------------------- time series: window extrema
def ts_max(x, w):
    return _extrema(x, w)[1]


def ts_min(x, w):
    return _extrema(x, w)[0]


def ts_chanpos(x, w):
    """(x - min) / (max - min); a flat window is 0.5."""
    lo, hi, ok = _extrema(x, w)
    wide = hi > lo
    return np.where(ok, np.where(wide, (x - lo) / np.where(wide, hi - lo, 1.0), 0.5), NAN)


def ts_scale(x, w, k):
    """(x - min) / (max - min) + k; a flat window is NaN (unlike ts_chanpos)."""
    lo, hi, ok = _extrema(x, w)
    den = hi - lo
    ok = ok & (den != 0)
    return np.where(ok, (x - lo) / np.where(ok, den, 1.0) + k, NAN)


def ts_ddown(x, w):
    """x / window max - 1; a max <= 0 -> NaN."""
    _, hi, ok = _extrema(x, w)
    ok = ok & (hi > 0)
    return np.where(ok, x / np.where(ok, hi, 1.0) - 1.0, NAN)


def _arg_extreme(x, w, pick):
    """Rows since the window's extreme; ties go to the most recent row (the reference's monotone deque)."""
    out = np.full(x.shape, NAN)
    if len(x) >= w:
        out[w - 1:] = pick(_windows(x, w)[..., ::-1], axis=-1)
    return np.where(_nan_free(np.isnan(x), w), out, NAN)


def ts_arg_max(x, w):
    return _arg_extreme(x, w, np.argmax)


def ts_arg_min(x, w):
    return _arg_extreme(x, w, np.argmin)


def ts_median(x, w):
    T, S = x.shape
    out = np.full(x.shape, NAN)
    step = max(1, (1 << 22) // (S * w))            # rows per block: bounds the sorted copy of the windows
    for lo in range(w - 1, T, step):
        v = np.sort(_windows(x[lo - w + 1:lo + step], w), axis=-1)
        m = v[..., w // 2]
        out[lo:lo + step] = m if w % 2 else 0.5 * (m + v[..., w // 2 - 1])
    return np.where(_nan_free(np.isnan(x), w), out, NAN)


IMPL = {
    'abs': np.abs, 'neg': np.negative, 'sign': sign, 'log': log, 'square': lambda x: x * x, 'sqrt': sqrt,
    'tanh': np.tanh, 'inv': inv, 's_log_1p': lambda x: np.copysign(np.log1p(np.abs(x)), x),
    'not': lambda x: np.where(np.isnan(x), NAN, np.where(x > 0, 0.0, 1.0)),
    'is_nan': lambda x: np.isnan(x).astype(np.float64),
    'add': np.add, 'sub': np.subtract, 'mul': np.multiply, 'div': div,
    'max': lambda a, b: np.where(_both(a, b), NAN, np.where(a > b, a, b)),
    'min': lambda a, b: np.where(_both(a, b), NAN, np.where(a < b, a, b)),
    'gt': lambda a, b: _flag(a, b, a > b), 'lt': lambda a, b: _flag(a, b, a < b),
    'and': lambda a, b: _flag(a, b, (a > 0) & (b > 0)), 'or': lambda a, b: _flag(a, b, (a > 0) | (b > 0)),
    'equal': lambda a, b: _flag(a, b, a == b), 'not_equal': lambda a, b: _flag(a, b, a != b),
    'greater_equal': lambda a, b: _flag(a, b, a >= b), 'less_equal': lambda a, b: _flag(a, b, a <= b),
    'add_const': lambda x, k: x + k, 'mul_const': lambda x, k: x * k, 'pow_const': pow_const,
    'left_tail': lambda x, k: np.where(x > k, NAN, x), 'right_tail': lambda x, k: np.where(x < k, NAN, x),
    'hump': hump,
    'if_else': lambda c, a, b: np.where(np.isnan(c), NAN, np.where(c > 0, a, b)), 'trade_when': trade_when,
    'cs_rank': cs_rank, 'cs_zscore': cs_zscore, 'cs_demean': cs_demean, 'cs_scale': cs_scale,
    'ts_mean': ts_mean, 'ts_std': ts_std, 'ts_max': ts_max, 'ts_min': ts_min, 'ts_sum': ts_sum, 'ts_rank': ts_rank,
    'ts_arg_max': ts_arg_max, 'ts_arg_min': ts_arg_min, 'ts_ema': ts_ema, 'ts_wma': ts_wma, 'ts_ref': ts_ref,
    'ts_delta': ts_delta, 'ts_skew': ts_skew, 'ts_kurt': ts_kurt, 'ts_mad': ts_mad, 'ts_slope': ts_slope,
    'ts_chanpos': ts_chanpos, 'ts_ddown': ts_ddown, 'ts_zscore': ts_zscore, 'ts_snr': ts_snr,
    'ts_backfill': ts_backfill, 'ts_median': ts_median, 'ts_av_diff': ts_av_diff,
    'ts_decay_exp_window': ts_decay_exp_window, 'ts_scale': ts_scale, 'ts_quantile': ts_quantile,
    'ts_corr': ts_corr, 'ts_cov': ts_cov, 'ts_regression': ts_regression,
}
