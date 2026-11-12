/* foundry/ops/src/ops_ts.cpp — Rolling kernels (per-symbol time series)
 *
 * unary: fm_ts_sum / mean / std / mad / skew / kurt / slope / max / min / arg_max / arg_min /
 *        ref / delta / ema / wma / rank / chanpos / ddown / zscore / snr
 * pair:  fm_ts_corr / fm_ts_cov
 *
 * All parallelised across symbols via OpenMP; incremental window updates (sum / sum^2 / monotonic
 * deque / Fenwick BIT).
 * AVX2 SIMD lane-packing for ts_sum/ts_mean (for other ops, a serial t-loop is already cache-friendly).
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


/* SIMD enablement: compiled in whenever gcc/clang defines __AVX2__. 4-lane path for 4*k symbols, scalar tail otherwise. */
#if defined(__AVX2__)
#include <immintrin.h>
#define FM_HAS_AVX2 1
#else
#define FM_HAS_AVX2 0
#endif

#if FM_HAS_AVX2
/* AVX2 lane-pack helper: treat NaN as zero + accumulate cnt_nan (int64x4) for x[t*ldx + s_block .. s_block+3]. */
static inline void _fm_unpack_clean(__m256d v, __m256d* out_clean, __m256i* out_inc) {
    /* nan_mask all-ones lanes = NaN */
    __m256d nan_mask = _mm256_cmp_pd(v, v, _CMP_UNORD_Q);
    /* clean = nan ? 0 : v */
    *out_clean = _mm256_blendv_pd(v, _mm256_setzero_pd(), nan_mask);
    /* inc = nan_mask ? 1 : 0 */
    __m256i ones = _mm256_set1_epi64x(1);
    *out_inc = _mm256_and_si256(_mm256_castpd_si256(nan_mask), ones);
}

/* 4-symbol lane-packed kernel for ts_sum / ts_mean. */
static void rolling_sum_simd_block(const double* x, double* out,
                                   int64_t T, int64_t w,
                                   int64_t s_block, int divide,
                                   int64_t ldx, int64_t ldo) {
    __m256d vsum = _mm256_setzero_pd();
    __m256i vcnt = _mm256_setzero_si256();
    const __m256d vnan_out = _mm256_set1_pd(NAN);
    const __m256d vw = _mm256_set1_pd((double)w);

    /* Warmup t = 0 .. w-2: accumulate into sum/cnt, emit NaN */
    for (int64_t t = 0; t < w - 1; t++) {
        __m256d v = _mm256_loadu_pd(&x[t * ldx + s_block]);
        __m256d clean; __m256i inc;
        _fm_unpack_clean(v, &clean, &inc);
        vsum = _mm256_add_pd(vsum, clean);
        vcnt = _mm256_add_epi64(vcnt, inc);
        _mm256_storeu_pd(&out[t * ldo + s_block], vnan_out);
    }
    /* Main loop t >= w-1: add current -> emit -> evict oldest. */
    for (int64_t t = w - 1; t < T; t++) {
        __m256d v_new = _mm256_loadu_pd(&x[t * ldx + s_block]);
        __m256d clean_new; __m256i inc_new;
        _fm_unpack_clean(v_new, &clean_new, &inc_new);
        vsum = _mm256_add_pd(vsum, clean_new);
        vcnt = _mm256_add_epi64(vcnt, inc_new);

        /* cnt == 0 mask -> blend mean / NaN */
        __m256i cnt_zero = _mm256_cmpeq_epi64(vcnt, _mm256_setzero_si256());
        __m256d val = divide ? _mm256_div_pd(vsum, vw) : vsum;
        __m256d res = _mm256_blendv_pd(vnan_out, val, _mm256_castsi256_pd(cnt_zero));
        _mm256_storeu_pd(&out[t * ldo + s_block], res);

        /* Evict t - w + 1 (= 0 when t = w-1; never reached when t < w-1) */
        if (t >= w - 1) {
            int64_t old_t = t - w + 1;
            if (old_t >= 0) {
                __m256d v_drop = _mm256_loadu_pd(&x[old_t * ldx + s_block]);
                __m256d clean_drop; __m256i inc_drop;
                _fm_unpack_clean(v_drop, &clean_drop, &inc_drop);
                vsum = _mm256_sub_pd(vsum, clean_drop);
                vcnt = _mm256_sub_epi64(vcnt, inc_drop);
            }
        }
    }
}
#endif  /* FM_HAS_AVX2 */

/* Scalar fallback / remainder when S is not divisible by 4 */
static void rolling_sum_scalar_one(const double* x, double* out, int64_t T,
                                   int64_t w, int64_t s, int divide,
                                   int64_t ldx, int64_t ldo) {
    double sum = 0.0;
    int64_t cnt_nan = 0;
    for (int64_t t = 0; t < T; t++) {
        double v = x[t * ldx + s];
        if (isnan(v)) cnt_nan++; else sum += v;
        if (t >= w) {
            double old = x[(t - w) * ldx + s];
            if (isnan(old)) cnt_nan--; else sum -= old;
        }
        if (t >= w - 1 && cnt_nan == 0) out[t * ldo + s] = divide ? sum / (double)w : sum;
        else                            out[t * ldo + s] = NAN;
    }
}

static void rolling_sum_impl(const double* x, double* out, int64_t T, int64_t S, int64_t w,
                             int divide, int64_t ldx, int64_t ldo) {
#if FM_HAS_AVX2
    int64_t S_simd = S - (S & 3);    /* Multiple-of-4 prefix */
    #pragma omp parallel for schedule(static)
    for (int64_t s_block = 0; s_block < S_simd; s_block += 4) {
        rolling_sum_simd_block(x, out, T, w, s_block, divide, ldx, ldo);
    }
    /* Scalar tail */
    #pragma omp parallel for schedule(static)
    for (int64_t s = S_simd; s < S; s++) {
        rolling_sum_scalar_one(x, out, T, w, s, divide, ldx, ldo);
    }
#else
    #pragma omp parallel for schedule(static)
    for (int64_t s = 0; s < S; s++) {
        rolling_sum_scalar_one(x, out, T, w, s, divide, ldx, ldo);
    }
#endif
}

void fm_ts_sum (const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo) { rolling_sum_impl(x, out, T, S, w, 0, ldx, ldo); }
void fm_ts_mean(const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo) { rolling_sum_impl(x, out, T, S, w, 1, ldx, ldo); }


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


/* ----- ts_rank bucketization helpers ----- */

#define _RANK_BUCKETS 128
struct _ts_rank_vi { double v; int64_t orig; };
typedef struct _ts_rank_vi _ts_rank_vi_t;

static inline void _bit_add(int64_t* bit, int64_t size, int64_t i, int64_t delta) {
    for (; i <= size; i += i & -i) bit[i] += delta;
}
static inline int64_t _bit_prefix(const int64_t* bit, int64_t i) {
    int64_t s = 0;
    for (; i > 0; i -= i & -i) s += bit[i];
    return s;
}

/* Bucketize one column: bucket_id[t] in {1..B} for finite values, 0 for NaN. */
static void _bucketize_column(
    const double* x, int64_t T, int64_t ldx, int64_t s_idx, int B,
    int64_t* bucket_id, int64_t* n_finite_out)
{
    std::vector<_ts_rank_vi_t> arr(T);                          /* (value, orig_t) (RAII) */
    int64_t n = 0;
    for (int64_t t = 0; t < T; t++) {
        double v = x[t * ldx + s_idx];
        if (!isnan(v)) { arr[n].v = v; arr[n].orig = t; n++; }
        else { bucket_id[t] = 0; }
    }
    *n_finite_out = n;
    if (n == 0) return;
    /* Sort ascending by value; tie order is unspecified — bucket IDs follow sorted position,
     * and the bucketed kernel is a +-1/(2B) approximation anyway (continuous data has no exact ties). */
    std::sort(arr.begin(), arr.begin() + n,
              [](const _ts_rank_vi_t& a, const _ts_rank_vi_t& b) { return a.v < b.v; });
    for (int64_t k = 0; k < n; k++) {
        int64_t b = (k * B) / n + 1;
        if (b > B) b = B;
        bucket_id[arr[k].orig] = b;
    }
}


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
        /* Force NaN on warmup rows 0 .. w-2 */
        for (int64_t t = 0; t < w - 1; t++) out[t * ldo + s] = NAN;

        /* t = w-1: O(w) bootstrap of SUM / WSUM */
        double SUM = 0.0, WSUM = 0.0;
        int64_t cnt_nan = 0;
        for (int64_t i = 0; i < w; i++) {
            double v = x[i * ldx + s];
            if (isnan(v)) cnt_nan++;
            else { SUM += v; WSUM += (double)(i + 1) * v; }
        }
        out[(w - 1) * ldo + s] = (cnt_nan == 0) ? (WSUM / wsum) : NAN;

        /* t >= w: O(1)/step sliding update */
        for (int64_t t = w; t < T; t++) {
            double drop = x[(t - w) * ldx + s];
            double new_v = x[t * ldx + s];
            int drop_nan = isnan(drop), new_nan = isnan(new_v);
            double drop_clean = drop_nan ? 0.0 : drop;
            double new_clean  = new_nan  ? 0.0 : new_v;

            /* Update WSUM using OLD SUM (includes drop) */
            WSUM = WSUM - SUM + (double)w * new_clean;
            SUM  = SUM - drop_clean + new_clean;
            if (drop_nan) cnt_nan--;
            if (new_nan)  cnt_nan++;

            out[t * ldo + s] = (cnt_nan == 0) ? (WSUM / wsum) : NAN;
        }
    }
}


/* ts_rank: normalised average rank of the current value within the window, in [0, 1].
 *
 * O(T*S*log T) scheme — per-column coordinate/bucket compression + sliding Fenwick BIT
 * (was O(T*S*W), ~10-15x faster at W=288):
 *
 * Per column:
 *   1. Collect finite (value, orig_t) -> sort by value -> rank index comp[t] in {1..M}; NaN -> 0
 *   2. BIT[1..M]: add(c, +1) / add(c, -1) / prefix(c) are all O(log M)
 *   3. Over window [t-w+1 .. t]:
 *        cnt_less = prefix(c - 1)
 *        cnt_eq   = prefix(c) - cnt_less       (includes current value)
 *        rank     = (cnt_less + 0.5 * (cnt_eq - 1)) / (w - 1)
 *   4. Side-channel cnt_nan tracks NaNs; cnt_nan > 0 -> emit NaN
 *
 * Tie handling: equal values share the same compressed index, so prefix differences count ties directly.
 */

/* Small W uses O(T*S*W) naive scan — W inner iterations are cache-hot with tiny constant factor. */
static void _ts_rank_naive(const double* x, double* out, int64_t T, int64_t S, int64_t w,
                           int64_t ldx, int64_t ldo) {
    #pragma omp parallel for schedule(static)
    for (int64_t s = 0; s < S; s++) {
        for (int64_t t = 0; t < T; t++) {
            if (t < w - 1) { out[t * ldo + s] = NAN; continue; }
            double cur = x[t * ldx + s];
            if (isnan(cur)) { out[t * ldo + s] = NAN; continue; }
            int64_t cnt_less = 0, cnt_eq = 0;
            int has_nan = 0;
            for (int64_t i = 0; i < w; i++) {
                double v = x[(t - i) * ldx + s];
                if (isnan(v)) { has_nan = 1; break; }
                if (v < cur) cnt_less++;
                else if (v == cur) cnt_eq++;
            }
            if (has_nan) { out[t * ldo + s] = NAN; continue; }
            out[t * ldo + s] = ((double)cnt_less + 0.5 * (double)(cnt_eq - 1))
                             / (double)(w - 1);
        }
    }
}

/* Large W uses bucketized BIT — B=128 quantile buckets, BIT depth log2(B) = 7.
 * Same-bucket values are treated as ties; rank error is +-1/(2B) ~ 0.4% with negligible effect on rank IC. */
static void _ts_rank_bucket(const double* x, double* out, int64_t T, int64_t S, int64_t w,
                            int64_t ldx, int64_t ldo) {
    const int B = _RANK_BUCKETS;
    #pragma omp parallel for schedule(static)
    for (int64_t s = 0; s < S; s++) {
        std::vector<int64_t> bucket_buf(T, 0);                  /* Zero-initialised (RAII) */
        int64_t* bucket_id = bucket_buf.data();
        int64_t n_finite;
        _bucketize_column(x, T, ldx, s, B, bucket_id, &n_finite);
        if (n_finite == 0) {
            for (int64_t t = 0; t < T; t++) out[t * ldo + s] = NAN;
            continue;
        }
        std::vector<int64_t> bit_buf(B + 2, 0);                 /* Zero-initialised */
        int64_t* bit = bit_buf.data();
        int64_t cnt_nan = 0;

        for (int64_t t = 0; t < w - 1; t++) {
            out[t * ldo + s] = NAN;
            int64_t b = bucket_id[t];
            if (b == 0) cnt_nan++;
            else        _bit_add(bit, B, b, +1);
        }
        for (int64_t t = w - 1; t < T; t++) {
            int64_t b = bucket_id[t];
            if (b == 0) cnt_nan++;
            else        _bit_add(bit, B, b, +1);

            if (cnt_nan > 0 || b == 0) {
                out[t * ldo + s] = NAN;
            } else {
                int64_t cnt_less = _bit_prefix(bit, b - 1);
                int64_t cnt_eq   = _bit_prefix(bit, b) - cnt_less;
                out[t * ldo + s] = ((double)cnt_less + 0.5 * (double)(cnt_eq - 1))
                                 / (double)(w - 1);
            }

            int64_t old_t = t - w + 1;
            int64_t ob = bucket_id[old_t];
            if (ob == 0) cnt_nan--;
            else         _bit_add(bit, B, ob, -1);
        }
    }
}

/* Entry point: W < 32 uses naive scan (cache-friendly, tiny constant); W >= 32 uses bucketized BIT. */
void fm_ts_rank(const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo) {
    if (w < 32) _ts_rank_naive (x, out, T, S, w, ldx, ldo);
    else        _ts_rank_bucket(x, out, T, S, w, ldx, ldo);
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
    std::vector<int64_t> cnt_buf(S);
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
                    if (va_ > 0.0 && vb_ > 0.0) orow[s] = cov / sqrt(va_ * vb_);
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
    std::vector<int64_t> cnt_buf(S);
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
                    orow[s] = (sab[s] - ma * sb[s]) / (double)(w - 1);
                } else {
                    orow[s] = NAN;
                }
            }
        }
    }
}
