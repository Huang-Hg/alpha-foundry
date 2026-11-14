/* foundry/ops/src/ops_ts.cpp — Rolling kernels (per-symbol time series)
 *
 * unary: fm_ts_sum / mean / std / mad / skew / kurt / slope / max / min / arg_max / arg_min /
 *        ref / delta / ema / wma / rank / chanpos / ddown / zscore / snr
 * pair:  fm_ts_corr / fm_ts_cov
 *
 * All parallelised across symbols via OpenMP; incremental window updates (sum / sum^2 / monotonic
 * deque / Fenwick BIT).
 * Sum/mean recompute centered windows in the same order as CUDA to preserve discrete decisions.
 *
 * 2D strided addressing: x[t * ldx + s] / out[t * ldo + s] (ld = row stride; see ops.hpp).
 * Per-column accumulation order along t is independent of ld -> bitwise identical for any ld.
 */

#include "ops.hpp"

#include <math.h>
#include <stdint.h>

#include <algorithm>    /* std::sort */
#include <vector>       /* per-thread / per-column scratch */

#ifdef _OPENMP
#include <omp.h>
#endif


/* Match the CUDA moment kernel exactly at decision boundaries: each window
 * accumulates oldest-to-newest deviations from its oldest observation. An
 * explicit fused multiply-add restores the anchor, then division rounds once.
 * Rolling subtraction drifts across a column and can turn an exact flat window
 * into a positive/negative signal. Fast-math reassociation/reciprocal transforms
 * would reintroduce the same discrete decision error, so these kernels are strict.
 */
#pragma GCC push_options
#pragma GCC optimize ("no-fast-math", "fp-contract=off")
static void rolling_sum_impl(const double* x, double* out, int64_t T, int64_t S, int64_t w,
                             int divide, int64_t ldx, int64_t ldo) {
    #pragma omp parallel for schedule(static)
    for (int64_t s = 0; s < S; s++) {
        for (int64_t t = 0; t < std::min(T, w - 1); t++) out[t * ldo + s] = NAN;
        for (int64_t t = w - 1; t < T; t++) {
            const double anchor = x[(t - w + 1) * ldx + s];
            double centered = 0.;
            bool missing = false;
            for (int64_t k = 0; k < w; k++) {
                const double value = x[(t - w + 1 + k) * ldx + s];
                if (isnan(value)) { missing = true; break; }
                centered += value - anchor;
            }
            const double sum = fma((double)w, anchor, centered);
            out[t * ldo + s] = missing ? NAN : (divide ? sum / (double)w : sum);
        }
    }
}
void fm_ts_sum (const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo) { rolling_sum_impl(x, out, T, S, w, 0, ldx, ldo); }
void fm_ts_mean(const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo) { rolling_sum_impl(x, out, T, S, w, 1, ldx, ldo); }
#pragma GCC pop_options

/* Per-column valid-range trimming: in wide panels, most columns have long pre-listing / post-delisting
 * all-NaN runs. Rolling kernels run only on [vlo, vhi] = first/last valid row and write NaN outside.
 * Outputs outside [vlo, vhi] are guaranteed NaN (window contains NaN), so this is **bitwise identical**
 * to the untrimmed version (not an approximation); scan cost = O(skipped rows), O(1) on dense columns.
 * Full-window check is written t - vlo >= w - 1 (accumulated from vlo; [0, vlo) is all-NaN and contributes
 * nothing to sum, only already-evicted cnt_nan). */
static inline void _col_range(const double* x, int64_t T, int64_t ldx, int64_t s,
                              int64_t* vlo, int64_t* vhi) {
    int64_t l = 0;     while (l < T && isnan(x[l * ldx + s])) l++;
    int64_t h = T - 1; while (h >= l && isnan(x[h * ldx + s])) h--;
    *vlo = l; *vhi = h;
}


/* std = sqrt((sum_sq - mean * sum) / (w - 1)), sample (ddof=1). */
static void rolling_std_impl(const double* x, double* out, int64_t T, int64_t S, int64_t w,
                             int64_t ldx, int64_t ldo, const int64_t* rng_lo, const int64_t* rng_hi) {
    #pragma omp parallel for schedule(static)
    for (int64_t s = 0; s < S; s++) {
        int64_t vlo, vhi;
        if (rng_lo) { vlo = rng_lo[s]; vhi = rng_hi[s]; }   /* Caller-precomputed range */
        else        { _col_range(x, T, ldx, s, &vlo, &vhi); }
        if (vlo > vhi) { for (int64_t t = 0; t < T; t++) out[t * ldo + s] = NAN; continue; }
        for (int64_t t = 0; t < vlo; t++)      out[t * ldo + s] = NAN;
        for (int64_t t = vhi + 1; t < T; t++)  out[t * ldo + s] = NAN;
        double sum = 0.0, sum_sq = 0.0;
        int64_t cnt_nan = 0;
        for (int64_t t = vlo; t <= vhi; t++) {
            double v = x[t * ldx + s];
            if (isnan(v)) cnt_nan++;
            else { sum += v; sum_sq += v * v; }
            if (t - vlo >= w) {
                double old = x[(t - w) * ldx + s];
                if (isnan(old)) cnt_nan--;
                else { sum -= old; sum_sq -= old * old; }
            }
            if (t - vlo >= w - 1 && cnt_nan == 0) {
                double mean = sum / (double)w;
                double var  = (sum_sq - mean * sum) / (double)(w - 1);
                if (var < 0.0) var = 0.0;  /* Clamp tiny negative values from numerical noise */
                out[t * ldo + s] = sqrt(var);
            } else {
                out[t * ldo + s] = NAN;
            }
        }
    }
}

void fm_ts_std(const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo, const int64_t* rng_lo, const int64_t* rng_hi) { rolling_std_impl(x, out, T, S, w, ldx, ldo, rng_lo, rng_hi); }


/* ts_mad: rolling mean absolute deviation = mean(|x - rolling_mean|).
 * More robust than ts_std (L1 vs L2 dispersion). Requires 2 passes, evaluated only when cnt_nan == 0. */
void fm_ts_mad(const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo, const int64_t* rng_lo, const int64_t* rng_hi) {
    #pragma omp parallel for schedule(static)
    for (int64_t s = 0; s < S; s++) {
        int64_t vlo, vhi;
        if (rng_lo) { vlo = rng_lo[s]; vhi = rng_hi[s]; }   /* Caller-precomputed range */
        else        { _col_range(x, T, ldx, s, &vlo, &vhi); }
        if (vlo > vhi) { for (int64_t t = 0; t < T; t++) out[t * ldo + s] = NAN; continue; }
        for (int64_t t = 0; t < vlo; t++)      out[t * ldo + s] = NAN;
        for (int64_t t = vhi + 1; t < T; t++)  out[t * ldo + s] = NAN;
        double sum = 0.0;
        int64_t cnt_nan = 0;
        for (int64_t t = vlo; t <= vhi; t++) {
            double v = x[t * ldx + s];
            if (isnan(v)) cnt_nan++; else sum += v;
            if (t - vlo >= w) {
                double old = x[(t - w) * ldx + s];
                if (isnan(old)) cnt_nan--; else sum -= old;
            }
            if (t - vlo >= w - 1 && cnt_nan == 0) {
                double mean = sum / (double)w;
                double mad = 0.0;
                for (int64_t k = t - w + 1; k <= t; k++) {
                    mad += fabs(x[k * ldx + s] - mean);
                }
                out[t * ldo + s] = mad / (double)w;
            } else {
                out[t * ldo + s] = NAN;
            }
        }
    }
}


/* ============================================================================
 * Fused trend operators — 4 literature-aligned single-node trend primitives (composing from
 * existing ops takes >= 4 nodes, unreachable within short formula budgets):
 *   ts_chanpos  Donchian channel position (x - min) / (max - min) in [0, 1] (Zarattini breakout ensemble)
 *   ts_ddown    Drawdown from rolling high x / max - 1 <= 0 (George-Hwang nearness-to-high)
 *   ts_zscore   Rolling z = (x - mu) / sigma (normalised trend; ddof=0)
 *   ts_snr      Trend signal-to-noise mu / sigma (rolling t-stat; AdaptiveTrend quality gate)
 * Same pattern as ts_mad: sliding NaN count + O(w) window rescan when valid (numerically stable,
 * avoids incremental power-sum drift); any NaN in window -> NaN; degenerate window (range/sigma ~ 0)
 * -> chanpos=0.5, zscore/snr/ddown=NaN.
 * ============================================================================ */
#define TS_FUSED_LOOP(...)                                                           \
    _Pragma("omp parallel for schedule(static)")                                     \
    for (int64_t s = 0; s < S; s++) {                                                \
        int64_t vlo, vhi;                                                            \
        if (rng_lo) { vlo = rng_lo[s]; vhi = rng_hi[s]; }                            \
        else        { _col_range(x, T, ldx, s, &vlo, &vhi); }                        \
        if (vlo > vhi) { for (int64_t t = 0; t < T; t++) out[t * ldo + s] = NAN; continue; } \
        for (int64_t t = 0; t < vlo; t++)      out[t * ldo + s] = NAN;               \
        for (int64_t t = vhi + 1; t < T; t++)  out[t * ldo + s] = NAN;               \
        int64_t cnt_nan = 0;                                                         \
        for (int64_t t = vlo; t <= vhi; t++) {                                       \
            double v = x[t * ldx + s];                                               \
            if (isnan(v)) cnt_nan++;                                                 \
            if (t - vlo >= w) {                                                      \
                double old = x[(t - w) * ldx + s];                                   \
                if (isnan(old)) cnt_nan--;                                           \
            }                                                                        \
            if (t - vlo >= w - 1 && cnt_nan == 0) { __VA_ARGS__ }                    \
            else out[t * ldo + s] = NAN;                                             \
        }                                                                            \
    }

/* ---------------------------------------------------------------------------
 * Per-column rolling min/max — dual monotonic deques, O(1) amortised. Shared by chanpos / ddown / ts_scale (large w).
 *
 * **Bitwise identical** to window rescan: min/max are exact selections (picking a value from the window)
 * with no arithmetic — deque and rescan pick the exact same double, so downstream (v - mn) / (mx - mn) is
 * bit-for-bit identical.
 * (Contrast: window sums in zscore/snr/skew/kurt **cannot** be converted to incremental updates without
 * reordering floating-point additions, and CPU f64 is the system-wide golden reference.)
 *
 * ok[t]=1 <=> full window with no NaNs, matching TS_FUSED_LOOP's criterion verbatim (including rng
 * trimming and the `t - vlo >= w - 1` origin convention). Caller provides deque scratch (w int64s each).
 * --------------------------------------------------------------------------- */
void fm_win_minmax_col(const double* x, int64_t T, int64_t ldx, int64_t s, int64_t w,
                            const int64_t* rng_lo, const int64_t* rng_hi,
                            int64_t* dq_lo, int64_t* dq_hi,
                            double* mn, double* mx, unsigned char* ok) {
    for (int64_t t = 0; t < T; t++) ok[t] = 0;
    int64_t vlo, vhi;
    if (rng_lo) { vlo = rng_lo[s]; vhi = rng_hi[s]; }
    else        { _col_range(x, T, ldx, s, &vlo, &vhi); }
    if (vlo > vhi) return;
    int64_t hlo = 0, tlo = 0, hhi = 0, thi = 0;
    int64_t cnt_nan = 0;
    for (int64_t t = vlo; t <= vhi; t++) {
        int64_t lo = t - w + 1;             /* lo-pop must precede push (same ring-slot invariant as rolling_minmax_impl) */
        while (tlo > hlo && dq_lo[hlo % w] < lo) hlo++;
        while (thi > hhi && dq_hi[hhi % w] < lo) hhi++;
        double v = x[t * ldx + s];
        if (isnan(v)) {
            cnt_nan++;
        } else {
            while (tlo > hlo && x[dq_lo[(tlo - 1) % w] * ldx + s] >= v) tlo--;
            dq_lo[tlo % w] = t; tlo++;
            while (thi > hhi && x[dq_hi[(thi - 1) % w] * ldx + s] <= v) thi--;
            dq_hi[thi % w] = t; thi++;
        }
        if (t - vlo >= w) {
            double old = x[(t - w) * ldx + s];
            if (isnan(old)) cnt_nan--;
        }
        if (t - vlo >= w - 1 && cnt_nan == 0 && tlo > hlo && thi > hhi) {
            mn[t] = x[dq_lo[hlo % w] * ldx + s];
            mx[t] = x[dq_hi[hhi % w] * ldx + s];
            ok[t] = 1;
        }
    }
}

/* Outer wrapper for chanpos / ddown: per-thread scratch, extract column extrema then apply per-op formula. */
#define TS_MINMAX_LOOP(...)                                                          \
    _Pragma("omp parallel")                                                          \
    {                                                                                \
        std::vector<int64_t> _dq((size_t)(2 * w));                                   \
        std::vector<double> _mn((size_t)T), _mx((size_t)T);                          \
        std::vector<unsigned char> _ok((size_t)T);                                   \
        _Pragma("omp for schedule(static)")                                          \
        for (int64_t s = 0; s < S; s++) {                                            \
            fm_win_minmax_col(x, T, ldx, s, w, rng_lo, rng_hi,                         \
                            _dq.data(), _dq.data() + w,                              \
                            _mn.data(), _mx.data(), _ok.data());                     \
            for (int64_t t = 0; t < T; t++) {                                        \
                if (!_ok[t]) { out[t * ldo + s] = NAN; continue; }                   \
                double v = x[t * ldx + s], mn = _mn[t], mx = _mx[t];                 \
                (void)v; (void)mn; (void)mx;                                         \
                __VA_ARGS__                                                          \
            }                                                                        \
        }                                                                            \
    }

/* The deque path is **slower on small windows**: with only a few elements, rescanning is a handful of
 * cache-hot comparisons, whereas the deque maintains ring buffers plus three length-T scratch arrays;
 * measured crossover is around d ~ 230. => Dispatch by window size (analogous to `w < 32 -> naive` in
 * fm_ts_rank). Both branches are **bitwise identical** (exact extremum selection); validation window sets
 * must span both sides of the threshold. */
#define TS_MINMAX_W_MIN 256

void fm_ts_chanpos(const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo, const int64_t* rng_lo, const int64_t* rng_hi) {
    if (w < TS_MINMAX_W_MIN) {
        TS_FUSED_LOOP({
            double mn = v, mx = v;
            for (int64_t k = t - w + 1; k <= t; k++) {
                double u = x[k * ldx + s];
                if (u < mn) mn = u;
                if (u > mx) mx = u;
            }
            out[t * ldo + s] = (mx > mn) ? (v - mn) / (mx - mn) : 0.5;
        })
        return;
    }
    TS_MINMAX_LOOP({
        out[t * ldo + s] = (mx > mn) ? (v - mn) / (mx - mn) : 0.5;
    })
}

void fm_ts_ddown(const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo, const int64_t* rng_lo, const int64_t* rng_hi) {
    if (w < TS_MINMAX_W_MIN) {
        TS_FUSED_LOOP({
            double mx = v;
            for (int64_t k = t - w + 1; k <= t; k++) {
                double u = x[k * ldx + s];
                if (u > mx) mx = u;
            }
            out[t * ldo + s] = (mx > 0.0) ? v / mx - 1.0 : NAN;
        })
        return;
    }
    TS_MINMAX_LOOP({
        out[t * ldo + s] = (mx > 0.0) ? v / mx - 1.0 : NAN;
    })
}

#undef TS_MINMAX_LOOP

void fm_ts_zscore(const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo, const int64_t* rng_lo, const int64_t* rng_hi) {
    TS_FUSED_LOOP({
        double sum = 0.0;
        for (int64_t k = t - w + 1; k <= t; k++) sum += x[k * ldx + s];
        double mean = sum / (double)w, ss = 0.0;
        for (int64_t k = t - w + 1; k <= t; k++) {
            double d = x[k * ldx + s] - mean;
            ss += d * d;
        }
        double sd = sqrt(ss / (double)w);
        out[t * ldo + s] = (sd > 1e-12) ? (v - mean) / sd : NAN;
    })
}

void fm_ts_snr(const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo, const int64_t* rng_lo, const int64_t* rng_hi) {
    TS_FUSED_LOOP({
        double sum = 0.0;
        for (int64_t k = t - w + 1; k <= t; k++) sum += x[k * ldx + s];
        double mean = sum / (double)w, ss = 0.0;
        for (int64_t k = t - w + 1; k <= t; k++) {
            double d = x[k * ldx + s] - mean;
            ss += d * d;
        }
        double sd = sqrt(ss / (double)w);
        out[t * ldo + s] = (sd > 1e-12) ? mean / sd : NAN;
    })
}

#undef TS_FUSED_LOOP


/* ts_skew: rolling 3rd standardised central moment = E[((x - mu) / sigma)^3].
 * **Two-pass direct centering per window** (pass 1 window mean -> pass 2 sum((v - mean)^p)): numerically
 * stable, eliminates e2 - mu^2 cancellation, and matches GPU moment_core's per-window two-pass algorithm
 * and summation order -> bitwise/tight parity. Incremental rolling power sums accumulate drift on long
 * columns (observed up to ~40% divergence vs GPU on kurt), so O(w)/output two-pass is used: slower than
 * incremental, but accurate. */
void fm_ts_skew(const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo, const int64_t* rng_lo, const int64_t* rng_hi) {
    #pragma omp parallel for schedule(static)
    for (int64_t s = 0; s < S; s++) {
        int64_t vlo, vhi;
        if (rng_lo) { vlo = rng_lo[s]; vhi = rng_hi[s]; }   /* Caller-precomputed range */
        else        { _col_range(x, T, ldx, s, &vlo, &vhi); }
        for (int64_t t = 0; t < T; t++) out[t * ldo + s] = NAN;   /* Pre-fill NaN (warmup / out-of-range / NaN windows) */
        if (vlo > vhi) continue;
        for (int64_t t = vlo + w - 1; t <= vhi; t++) {
            double mean = 0.0; int bad = 0;
            for (int64_t k = t - w + 1; k <= t; k++) {           /* pass 1: window mean (skip if any NaN) */
                double v = x[k * ldx + s];
                if (isnan(v)) { bad = 1; break; }
                mean += v;
            }
            if (bad) continue;
            mean /= (double)w;
            double m2 = 0.0, m3 = 0.0;
            for (int64_t k = t - w + 1; k <= t; k++) {           /* pass 2: direct central moments (no cancellation) */
                double d = x[k * ldx + s] - mean, d2 = d * d;
                m2 += d2; m3 += d2 * d;
            }
            m2 /= (double)w; m3 /= (double)w;
            if (m2 > 0.0) out[t * ldo + s] = m3 / pow(m2, 1.5);  /* Constant window m2 == 0 -> keep NaN */
        }
    }
}


/* ts_kurt: rolling 4th standardised central moment = E[((x - mu) / sigma)^4] - 3 (excess kurtosis). Two-pass central moments; see fm_ts_skew. */
void fm_ts_kurt(const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo, const int64_t* rng_lo, const int64_t* rng_hi) {
    #pragma omp parallel for schedule(static)
    for (int64_t s = 0; s < S; s++) {
        int64_t vlo, vhi;
        if (rng_lo) { vlo = rng_lo[s]; vhi = rng_hi[s]; }   /* Caller-precomputed range */
        else        { _col_range(x, T, ldx, s, &vlo, &vhi); }
        for (int64_t t = 0; t < T; t++) out[t * ldo + s] = NAN;
        if (vlo > vhi) continue;
        for (int64_t t = vlo + w - 1; t <= vhi; t++) {
            double mean = 0.0; int bad = 0;
            for (int64_t k = t - w + 1; k <= t; k++) {
                double v = x[k * ldx + s];
                if (isnan(v)) { bad = 1; break; }
                mean += v;
            }
            if (bad) continue;
            mean /= (double)w;
            double m2 = 0.0, m4 = 0.0;
            for (int64_t k = t - w + 1; k <= t; k++) {
                double d = x[k * ldx + s] - mean, d2 = d * d;
                m2 += d2; m4 += d2 * d2;
            }
            m2 /= (double)w; m4 /= (double)w;
            if (m2 > 0.0) out[t * ldo + s] = m4 / (m2 * m2) - 3.0;
        }
    }
}


/* ts_slope: rolling OLS slope = cov(t_idx, x) / var(t_idx) over t_idx in {0, 1, ..., w-1}.
 * Precomputed constants: mean_t = (w - 1) / 2, var_t = (w^2 - 1) / 12. */
/* Incremental O(T*S) implementation (instead of O(T*S*w)): slope = cov(k, x) / var_k, var_k = (w^2 - 1) / 12;
 *   cov_tx = (S_kx - mean_k * S_x) / w, where S_x = sum(x_k), S_kx = sum(k * x_k) (k = window-local index 0..w-1).
 * O(1) sliding-window recurrence (shifting all window positions by -1 subtracts one S_x):
 *   S_kx(t) = S_kx(t-1) - S_x(t-1) + xc[t-w] + (w-1)*xc[t]   (uses pre-update S_x)
 *   S_x (t) = S_x (t-1) + xc[t] - xc[t-w]
 * NaN -> 0 in sums + cnt_nan gates output to NaN-free full windows (same as ts_std). Recomputes O(w)
 * every _SLOPE_RESYNC steps to bound incremental summation drift (~T/RESYNC times total, negligible). */
#define _SLOPE_RESYNC 65536
void fm_ts_slope(const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo) {
    double mean_t = (double)(w - 1) / 2.0;
    double var_t  = (double)(w * w - 1) / 12.0;  /* Exact integer w^2 - 1 before division */
    if (var_t <= 0.0) {
        for (int64_t t = 0; t < T; t++)
            for (int64_t s = 0; s < S; s++) out[t * ldo + s] = NAN;
        return;
    }
    #pragma omp parallel for schedule(static)
    for (int64_t s = 0; s < S; s++) {
        /* slope scans its own per-column valid range (does not accept caller rng): incremental S_kx/S_x
         * values drift with accumulation origin + resync phase, so an external range start vs the column's
         * true start would introduce ~1 ULP difference and break bit-identity; scanning the column's actual
         * start is bitwise identical to the untrimmed loop. Scan cost O(skipped rows) is negligible. */
        int64_t vlo, vhi; _col_range(x, T, ldx, s, &vlo, &vhi);
        if (vlo > vhi) { for (int64_t t = 0; t < T; t++) out[t * ldo + s] = NAN; continue; }
        for (int64_t t = 0; t < vlo; t++)      out[t * ldo + s] = NAN;
        for (int64_t t = vhi + 1; t < T; t++)  out[t * ldo + s] = NAN;
        if (vhi - vlo + 1 < w) {               /* Column shorter than window: full window never reached -> all NaN */
            for (int64_t t = vlo; t <= vhi; t++) out[t * ldo + s] = NAN;
            continue;
        }
        /* First full window t = vlo + w - 1: direct O(w) bootstrap of S_x / S_kx / cnt_nan */
        double S_x = 0.0, S_kx = 0.0;
        int64_t cnt_nan = 0;
        for (int64_t k = 0; k < w; k++) {
            double v = x[(vlo + k) * ldx + s];
            if (isnan(v)) cnt_nan++;
            else { S_x += v; S_kx += (double)k * v; }
        }
        for (int64_t t = vlo; t < vlo + w - 1; t++) out[t * ldo + s] = NAN;
        out[(vlo + w - 1) * ldo + s] = (cnt_nan == 0)
            ? (S_kx - mean_t * S_x) / (double)w / var_t : NAN;
        /* Main loop t = vlo + w .. vhi: O(1) recurrence + periodic resync to bound drift */
        for (int64_t t = vlo + w; t <= vhi; t++) {
            if (((t - vlo) & (_SLOPE_RESYNC - 1)) == 0) {
                S_x = 0.0; S_kx = 0.0; cnt_nan = 0;
                for (int64_t k = 0; k < w; k++) {
                    double v = x[(t - w + 1 + k) * ldx + s];
                    if (isnan(v)) cnt_nan++;
                    else { S_x += v; S_kx += (double)k * v; }
                }
            } else {
                double xnv = x[t * ldx + s], xov = x[(t - w) * ldx + s];
                int nn = isnan(xnv), no = isnan(xov);
                double xn = nn ? 0.0 : xnv, xo = no ? 0.0 : xov;
                S_kx = S_kx - S_x + xo + (double)(w - 1) * xn;  /* Uses pre-update S_x */
                S_x  = S_x + xn - xo;
                if (nn) cnt_nan++;
                if (no) cnt_nan--;
            }
            out[t * ldo + s] = (cnt_nan == 0)
                ? (S_kx - mean_t * S_x) / (double)w / var_t : NAN;
        }
    }
}


/* Monotonic deque, O(N) amortised. is_max=1 maintains a decreasing deque with max at front.
 * At most w indices live in the window -> ring buffer of size w (legacy T*8B alloc wasted 23 MB
 * across 12 threads at T=245k, S=16; w*8B uses only ~2.3 KB/thread).
 *
 * Critical ordering: lo-pop (advancing head) MUST precede push. Otherwise tail % w can collide with
 *   head % w when tail - head == w, overwriting the still-live entry referenced by head.
 *   Running lo-pop first guarantees tail - head <= w - 1 at push time, so the push slot never aliases head. */
static void rolling_minmax_impl(const double* x, double* out, int64_t T, int64_t S, int64_t w,
                                int is_max, int64_t ldx, int64_t ldo,
                                const int64_t* rng_lo, const int64_t* rng_hi) {
    #pragma omp parallel
    {
        std::vector<int64_t> deque(w);                          /* Per-thread ring buffer (RAII) */
        #pragma omp for schedule(static)
        for (int64_t s = 0; s < S; s++) {
            int64_t vlo, vhi;
            if (rng_lo) { vlo = rng_lo[s]; vhi = rng_hi[s]; }   /* Caller-precomputed range */
            else        { _col_range(x, T, ldx, s, &vlo, &vhi); }
            if (vlo > vhi) { for (int64_t t = 0; t < T; t++) out[t * ldo + s] = NAN; continue; }
            for (int64_t t = 0; t < vlo; t++)      out[t * ldo + s] = NAN;
            for (int64_t t = vhi + 1; t < T; t++)  out[t * ldo + s] = NAN;
            int64_t head = 0, tail = 0;
            int64_t cnt_nan = 0;
            for (int64_t t = vlo; t <= vhi; t++) {
                /* 1. lo-pop: evict expired entries outside [t-w+1, t-1] first (all deque indices >= vlo) */
                int64_t lo = t - w + 1;
                while (tail > head && deque[head % w] < lo) head++;
                /* 2. Push current value into deque (skip NaN, still counting cnt_nan) */
                double v = x[t * ldx + s];
                if (isnan(v)) {
                    cnt_nan++;
                } else {
                    while (tail > head) {
                        double bk = x[deque[(tail - 1) % w] * ldx + s];
                        if ((is_max && bk <= v) || (!is_max && bk >= v)) tail--;
                        else break;
                    }
                    deque[tail % w] = t; tail++;
                }
                /* 3. Decrement cnt_nan when an outgoing value is NaN */
                if (t - vlo >= w) {
                    double old = x[(t - w) * ldx + s];
                    if (isnan(old)) cnt_nan--;
                }
                /* 4. Emit: no NaN in window and non-empty deque -> front value */
                if (t - vlo >= w - 1 && cnt_nan == 0 && tail > head) {
                    out[t * ldo + s] = x[deque[head % w] * ldx + s];
                } else {
                    out[t * ldo + s] = NAN;
                }
            }
        }
    }
}

void fm_ts_max(const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo, const int64_t* rng_lo, const int64_t* rng_hi) { rolling_minmax_impl(x, out, T, S, w, 1, ldx, ldo, rng_lo, rng_hi); }
void fm_ts_min(const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo, const int64_t* rng_lo, const int64_t* rng_hi) { rolling_minmax_impl(x, out, T, S, w, 0, ldx, ldo, rng_lo, rng_hi); }


/* arg_max / arg_min: reuse monotonic deque, emitting `t - deque[head]` = bars elapsed since max/min
 * (in [0, w-1]; returns 0 when current bar is the extremum). Same size-w ring buffer and lo-pop-before-push invariant. */
static void rolling_argminmax_impl(const double* x, double* out, int64_t T, int64_t S, int64_t w,
                                   int is_max, int64_t ldx, int64_t ldo) {
    #pragma omp parallel
    {
        std::vector<int64_t> deque(w);                          /* Per-thread ring buffer (RAII) */
        #pragma omp for schedule(static)
        for (int64_t s = 0; s < S; s++) {
            int64_t head = 0, tail = 0;
            int64_t cnt_nan = 0;
            for (int64_t t = 0; t < T; t++) {
                int64_t lo = t - w + 1;
                while (tail > head && deque[head % w] < lo) head++;
                double v = x[t * ldx + s];
                if (isnan(v)) cnt_nan++;
                else {
                    while (tail > head) {
                        double bk = x[deque[(tail - 1) % w] * ldx + s];
                        if ((is_max && bk <= v) || (!is_max && bk >= v)) tail--;
                        else break;
                    }
                    deque[tail % w] = t; tail++;
                }
                if (t >= w) {
                    double old = x[(t - w) * ldx + s];
                    if (isnan(old)) cnt_nan--;
                }
                if (t >= w - 1 && cnt_nan == 0 && tail > head) {
                    out[t * ldo + s] = (double)(t - deque[head % w]);
                } else {
                    out[t * ldo + s] = NAN;
                }
            }
        }
    }
}

void fm_ts_arg_max(const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo) { rolling_argminmax_impl(x, out, T, S, w, 1, ldx, ldo); }
void fm_ts_arg_min(const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo) { rolling_argminmax_impl(x, out, T, S, w, 0, ldx, ldo); }


/* ----- Lag / difference / EMA / WMA ----- */

void fm_ts_ref(const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo) {
    #pragma omp parallel for schedule(static)
    for (int64_t s = 0; s < S; s++) {
        for (int64_t t = 0; t < T; t++) {
            out[t * ldo + s] = (t >= w) ? x[(t - w) * ldx + s] : NAN;
        }
    }
}

void fm_ts_delta(const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo) {
    #pragma omp parallel for schedule(static)
    for (int64_t s = 0; s < S; s++) {
        for (int64_t t = 0; t < T; t++) {
            if (t >= w) {
                double cur = x[t * ldx + s], prv = x[(t - w) * ldx + s];
                out[t * ldo + s] = (isnan(cur) || isnan(prv)) ? NAN : (cur - prv);
            } else {
                out[t * ldo + s] = NAN;
            }
        }
    }
}


/* Matches pandas ewm(span=k, adjust=False, min_periods=k):
 *   alpha = 2 / (span + 1)
 *   Seeds on the first non-NaN value and recurses thereafter; cnt tracks non-NaN count, emitting NaN while cnt < span.
 *   NaN inputs leave the internal ema state unchanged and emit NaN at that row.
 */
void fm_ts_ema(const double* x, double* out, int64_t T, int64_t S, int64_t span, int64_t ldx, int64_t ldo) {
    double alpha = 2.0 / (double)(span + 1);
    #pragma omp parallel for schedule(static)
    for (int64_t s = 0; s < S; s++) {
        double ema = 0.0;
        int has_seed = 0;
        int64_t cnt = 0;
        for (int64_t t = 0; t < T; t++) {
            double v = x[t * ldx + s];
            if (isnan(v)) {
                out[t * ldo + s] = NAN;
                continue;
            }
            if (!has_seed) { ema = v; has_seed = 1; }
            else           { ema = alpha * v + (1.0 - alpha) * ema; }
            cnt++;
            out[t * ldo + s] = (cnt >= span) ? ema : NAN;
        }
    }
}


/* WMA: weight[i] = i + 1 (oldest = 1, newest = w). Weight sum = w*(w+1)/2.
 *
 * O(T*S) sliding-window implementation (was O(T*S*W), ~250x faster at W=288):
 *   Let SUM[t]  = sum_{i=0..w-1} x[t-w+1+i]
 *       WSUM[t] = sum_{i=0..w-1} (i+1) * x[t-w+1+i]
 *   Then WSUM[t+1] - WSUM[t] = -SUM[t] + w * x[t+1]   (uses OLD SUM containing the outgoing item)
 *        SUM[t+1]  = SUM[t] - x[t-w+1] + x[t+1]
 *   Both updates are O(1).
 *
 * NaN handling: treat NaN as 0 in SUM/WSUM while counting NaNs separately in cnt_nan; cnt_nan > 0 -> emit NaN.
 *               The SUM/WSUM recurrence identity still holds because NaN contributes 0 on both sides.
 */
void fm_ts_wma(const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo) {
    double wsum = (double)w * ((double)w + 1.0) / 2.0;
    #pragma omp parallel for schedule(static)
    for (int64_t s = 0; s < S; s++) {
        double SUM = 0.0, WSUM = 0.0;
        int64_t cnt_nan = 0;
        for (int64_t t = 0; t < T; t++) {
            double v = x[t * ldx + s];
            int missing = isnan(v);
            double clean = missing ? 0.0 : v;
            if (t < w) {
                if (!missing) { SUM += v; WSUM += (double)(t + 1) * v; }
            } else {
                double drop = x[(t - w) * ldx + s];
                int drop_nan = isnan(drop);
                WSUM = WSUM - SUM + (double)w * clean;
                SUM = SUM - (drop_nan ? 0.0 : drop) + clean;
                cnt_nan -= drop_nan;
            }
            cnt_nan += missing;
            out[t * ldo + s] = (t >= w - 1 && cnt_nan == 0) ? (WSUM / wsum) : NAN;
        }
    }
}


/* Exact rolling rank. Only the current trailing window enters the ordered set;
 * no full-sample compression, bucket boundaries, or future values. */
void fm_ts_rank(const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo) {
    #pragma omp parallel for schedule(static)
    for(int64_t s=0;s<S;++s) {
        std::vector<double> window;
        window.reserve((size_t)w);
        int64_t missing=0;
        for(int64_t t=0;t<T;++t) {
            if(t>=w) {
                double old=x[(t-w)*ldx+s];
                if(isnan(old)) --missing;
                else window.erase(std::lower_bound(window.begin(),window.end(),old));
            }
            double v=x[t*ldx+s];
            if(isnan(v)) ++missing;
            else window.insert(std::upper_bound(window.begin(),window.end(),v),v);
            if(t<w-1 || missing) {out[t*ldo+s]=NAN;continue;}
            auto lo=std::lower_bound(window.begin(),window.end(),v);
            auto hi=std::upper_bound(lo,window.end(),v);
            out[t*ldo+s]=((double)(lo-window.begin())+.5*(double)(hi-lo-1))/(double)(w-1);
        }
    }
}

/* ============================================================================
 * Rolling pair — fm_ts_corr / fm_ts_cov
 * Row-streaming layout: each thread owns a contiguous symbol strip [s0, s1) and advances row by row,
 * keeping sliding-window accumulators in contiguous L1-resident arrays.
 * Stride-1 access across s within each row utilises full cache lines; column-wise sweeps use only
 * 1/8 of each cache line (~8x DRAM traffic). Per-symbol t-accumulation order matches column-wise
 * traversal -> bitwise identical.
 * NaN on either side increments cnt[s]; any NaN in window emits NaN (matches ts_std).
 *
 * Range: uses caller-supplied rng when provided; otherwise intersects the valid ranges of columns a and b.
 * Rows t < vlo or t > vhi emit NaN inline during the row sweep.
 * ============================================================================ */

void fm_ts_corr(const double* a, const double* b, double* out, int64_t T, int64_t S, int64_t w,
                int64_t lda_, int64_t ldb, int64_t ldo, const int64_t* rng_lo, const int64_t* rng_hi) {
    std::vector<int64_t> vbuf;                                  /* Fallback [lo|hi] when rng is NULL (RAII) */
    const int64_t *vlo, *vhi;
    if (rng_lo) { vlo = rng_lo; vhi = rng_hi; }
    else {
        vbuf.resize((size_t)S * 2);
        int64_t* lo = vbuf.data(); int64_t* hi = vbuf.data() + S;
        #pragma omp parallel for schedule(static)
        for (int64_t s = 0; s < S; s++) {
            int64_t la, ha, lb, hb;
            _col_range(a, T, lda_, s, &la, &ha);
            _col_range(b, T, ldb,  s, &lb, &hb);
            lo[s] = la > lb ? la : lb;
            hi[s] = ha < hb ? ha : hb;
        }
        vlo = lo; vhi = hi;
    }
    std::vector<double>  acc_buf((size_t)S * 5);                /* 5 sliding accumulators (RAII) */
    std::vector<int64_t> cnt_buf(S), constant_a(S), constant_b(S);
    double*  acc = acc_buf.data();
    int64_t* cnt = cnt_buf.data();
    double *sa = acc, *sb = acc + S, *sab = acc + 2 * S, *saa = acc + 3 * S, *sbb = acc + 4 * S;
    #pragma omp parallel
    {
        int nth = omp_get_num_threads(), tid = omp_get_thread_num();
        int64_t s0 = S * (int64_t)tid / nth, s1 = S * (int64_t)(tid + 1) / nth;
        for (int64_t s = s0; s < s1; s++) { sa[s] = sb[s] = sab[s] = saa[s] = sbb[s] = 0.0; cnt[s] = 0; }
        for (int64_t t = 0; t < T; t++) {
            const double* ar = a + t * lda_;
            const double* br = b + t * ldb;
            double* orow = out + t * ldo;
            for (int64_t s = s0; s < s1; s++) {
                int64_t lo = vlo[s], hi = vhi[s];
                if (t < lo || t > hi) { orow[s] = NAN; continue; }
                double va = ar[s], vb = br[s];
                constant_a[s] = t > lo && va == a[(t-1)*lda_+s] ? constant_a[s]+1 : 1;
                constant_b[s] = t > lo && vb == b[(t-1)*ldb+s] ? constant_b[s]+1 : 1;
                int valid = !isnan(va) && !isnan(vb);
                if (!valid) cnt[s]++;
                else { sa[s] += va; sb[s] += vb; sab[s] += va * vb; saa[s] += va * va; sbb[s] += vb * vb; }
                if (t - lo >= w) {
                    double oa = a[(t - w) * lda_ + s], ob = b[(t - w) * ldb + s];
                    int o_valid = !isnan(oa) && !isnan(ob);
                    if (!o_valid) cnt[s]--;
                    else { sa[s] -= oa; sb[s] -= ob; sab[s] -= oa * ob; saa[s] -= oa * oa; sbb[s] -= ob * ob; }
                }
                if (t - lo >= w - 1 && cnt[s] == 0) {
                    double ma = sa[s] / (double)w, mb = sb[s] / (double)w;
                    double cov = (sab[s] - ma * sb[s]) / (double)(w - 1);
                    double va_ = (saa[s] - ma * sa[s]) / (double)(w - 1);
                    double vb_ = (sbb[s] - mb * sb[s]) / (double)(w - 1);
                    if (constant_a[s] < w && constant_b[s] < w && va_ > 0.0 && vb_ > 0.0) orow[s] = cov / sqrt(va_ * vb_);
                    else                        orow[s] = NAN;
                } else {
                    orow[s] = NAN;
                }
            }
        }
    }
}

void fm_ts_cov(const double* a, const double* b, double* out, int64_t T, int64_t S, int64_t w,
               int64_t lda_, int64_t ldb, int64_t ldo, const int64_t* rng_lo, const int64_t* rng_hi) {
    std::vector<int64_t> vbuf;                                  /* Fallback [lo|hi] when rng is NULL (RAII) */
    const int64_t *vlo, *vhi;
    if (rng_lo) { vlo = rng_lo; vhi = rng_hi; }
    else {
        vbuf.resize((size_t)S * 2);
        int64_t* lo = vbuf.data(); int64_t* hi = vbuf.data() + S;
        #pragma omp parallel for schedule(static)
        for (int64_t s = 0; s < S; s++) {
            int64_t la, ha, lb, hb;
            _col_range(a, T, lda_, s, &la, &ha);
            _col_range(b, T, ldb,  s, &lb, &hb);
            lo[s] = la > lb ? la : lb;
            hi[s] = ha < hb ? ha : hb;
        }
        vlo = lo; vhi = hi;
    }
    std::vector<double>  acc_buf((size_t)S * 3);                /* 3 sliding accumulators (RAII) */
    std::vector<int64_t> cnt_buf(S), constant_a(S), constant_b(S);
    double*  acc = acc_buf.data();
    int64_t* cnt = cnt_buf.data();
    double *sa = acc, *sb = acc + S, *sab = acc + 2 * S;
    #pragma omp parallel
    {
        int nth = omp_get_num_threads(), tid = omp_get_thread_num();
        int64_t s0 = S * (int64_t)tid / nth, s1 = S * (int64_t)(tid + 1) / nth;
        for (int64_t s = s0; s < s1; s++) { sa[s] = sb[s] = sab[s] = 0.0; cnt[s] = 0; }
        for (int64_t t = 0; t < T; t++) {
            const double* ar = a + t * lda_;
            const double* br = b + t * ldb;
            double* orow = out + t * ldo;
            for (int64_t s = s0; s < s1; s++) {
                int64_t lo = vlo[s], hi = vhi[s];
                if (t < lo || t > hi) { orow[s] = NAN; continue; }
                double va = ar[s], vb = br[s];
                constant_a[s] = t > lo && va == a[(t-1)*lda_+s] ? constant_a[s]+1 : 1;
                constant_b[s] = t > lo && vb == b[(t-1)*ldb+s] ? constant_b[s]+1 : 1;
                int valid = !isnan(va) && !isnan(vb);
                if (!valid) cnt[s]++;
                else { sa[s] += va; sb[s] += vb; sab[s] += va * vb; }
                if (t - lo >= w) {
                    double oa = a[(t - w) * lda_ + s], ob = b[(t - w) * ldb + s];
                    int o_valid = !isnan(oa) && !isnan(ob);
                    if (!o_valid) cnt[s]--;
                    else { sa[s] -= oa; sb[s] -= ob; sab[s] -= oa * ob; }
                }
                if (t - lo >= w - 1 && cnt[s] == 0) {
                    double ma = sa[s] / (double)w;
                    // A constant operand has exactly zero covariance. Raw moments
                    // otherwise manufacture tiny signals from cancellation.
                    orow[s] = constant_a[s] >= w || constant_b[s] >= w ? 0. :
                        (sab[s] - ma * sb[s]) / (double)(w - 1);
                } else {
                    orow[s] = NAN;
                }
            }
        }
    }
}
