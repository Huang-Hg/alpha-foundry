/* foundry/ops/src/ops.hpp — C API of the formula-operator CPU reference backend
 * (implemented in ops_xs.cpp / ops_ts.cpp / ops_inject_ts.cpp).
 *
 * Layout:
 *   (T,S) panels are row-major double: data[t * ld + s] = (time t, asset s). ld = row stride in elements (ld >= S);
 *   ld == S is C-contiguous, ld > S is a zero-copy column-block view of a wide panel.
 *   Elementwise / ts / pair operators take an independent ld per input and for out (appended at the end);
 *   cross-sectional (cs) operators require C-contiguous full-width panels (cross-sectional semantics span the full row).
 *   Missing values are IEEE 754 NaN and propagate elementwise.
 *
 * Rolling semantics:
 *   pandas `rolling(w).f()` with `min_periods=w`. Any NaN inside the window -> NaN output.
 *   Warm-up rows t < w - 1 -> NaN output.
 *
 * NaN propagation:
 *   Binary operators (add/mul/...) output NaN if either operand is NaN.
 *   div outputs NaN when b == 0 (never inf: downstream uniformly uses NaN for "undefined").
 */

#ifndef FM_OPS_H
#define FM_OPS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ===== Elementwise unary ===== */
void fm_abs   (const double* x, double* out, int64_t T, int64_t S, int64_t ldx, int64_t ldo);
void fm_neg   (const double* x, double* out, int64_t T, int64_t S, int64_t ldx, int64_t ldo);
void fm_sign  (const double* x, double* out, int64_t T, int64_t S, int64_t ldx, int64_t ldo);
void fm_log   (const double* x, double* out, int64_t T, int64_t S, int64_t ldx, int64_t ldo);  /* log(x) for x>0, else NaN */
void fm_sqrt_ (const double* x, double* out, int64_t T, int64_t S, int64_t ldx, int64_t ldo);  /* x>=0 else NaN */
void fm_square(const double* x, double* out, int64_t T, int64_t S, int64_t ldx, int64_t ldo);
void fm_tanh  (const double* x, double* out, int64_t T, int64_t S, int64_t ldx, int64_t ldo);  /* tanh(x) in (-1, 1) */
void fm_inv   (const double* x, double* out, int64_t T, int64_t S, int64_t ldx, int64_t ldo);  /* 1/x; x==0 -> NaN */
void fm_s_log_1p(const double* x, double* out, int64_t T, int64_t S, int64_t ldx, int64_t ldo);  /* sign(x)*log(1+|x|) */

/* ===== Elementwise binary ===== */
void fm_add  (const double* a, const double* b, double* out, int64_t T, int64_t S, int64_t lda, int64_t ldb, int64_t ldo);
void fm_sub  (const double* a, const double* b, double* out, int64_t T, int64_t S, int64_t lda, int64_t ldb, int64_t ldo);
void fm_mul  (const double* a, const double* b, double* out, int64_t T, int64_t S, int64_t lda, int64_t ldb, int64_t ldo);
void fm_div  (const double* a, const double* b, double* out, int64_t T, int64_t S, int64_t lda, int64_t ldb, int64_t ldo);
void fm_max_b(const double* a, const double* b, double* out, int64_t T, int64_t S, int64_t lda, int64_t ldb, int64_t ldo);  /* max(a,b) */
void fm_min_b(const double* a, const double* b, double* out, int64_t T, int64_t S, int64_t lda, int64_t ldb, int64_t ldo);  /* min(a,b) */
void fm_gt   (const double* a, const double* b, double* out, int64_t T, int64_t S, int64_t lda, int64_t ldb, int64_t ldo);  /* (a>b)?1:0, NaN propagates */
void fm_lt   (const double* a, const double* b, double* out, int64_t T, int64_t S, int64_t lda, int64_t ldb, int64_t ldo);  /* (a<b)?1:0, NaN propagates */

/* ===== Ternary bool combinators (cond/entry, a/alpha, b/exit) ===== */
void fm_if_else   (const double* c, const double* a, const double* b, double* out, int64_t T, int64_t S, int64_t ldc, int64_t lda, int64_t ldb, int64_t ldo);
void fm_trade_when(const double* e, const double* a, const double* x, double* out, int64_t T, int64_t S, int64_t lde, int64_t lda, int64_t ldx, int64_t ldo);

/* ===== Logical family =====
 * NaN propagation matches gt/lt, except fm_is_nan (whose purpose is testing for NaN). */
void fm_not   (const double* x, double* out, int64_t T, int64_t S, int64_t ldx, int64_t ldo);  /* x>0?0:1, NaN->NaN */
void fm_is_nan(const double* x, double* out, int64_t T, int64_t S, int64_t ldx, int64_t ldo);  /* NaN?1:0, does NOT propagate NaN */
void fm_and          (const double* a, const double* b, double* out, int64_t T, int64_t S, int64_t lda, int64_t ldb, int64_t ldo);
void fm_or           (const double* a, const double* b, double* out, int64_t T, int64_t S, int64_t lda, int64_t ldb, int64_t ldo);
void fm_equal        (const double* a, const double* b, double* out, int64_t T, int64_t S, int64_t lda, int64_t ldb, int64_t ldo);
void fm_not_equal    (const double* a, const double* b, double* out, int64_t T, int64_t S, int64_t lda, int64_t ldb, int64_t ldo);
void fm_greater_equal(const double* a, const double* b, double* out, int64_t T, int64_t S, int64_t lda, int64_t ldb, int64_t ldo);
void fm_less_equal   (const double* a, const double* b, double* out, int64_t T, int64_t S, int64_t lda, int64_t ldb, int64_t ldo);

/* ===== Elementwise binary_const (panel op scalar k, 1 SIMD pass) ===== */
void fm_add_const(const double* x, double k, double* out, int64_t T, int64_t S, int64_t ldx, int64_t ldo);
void fm_mul_const(const double* x, double k, double* out, int64_t T, int64_t S, int64_t ldx, int64_t ldo);
/* k is continuous (unconstrained by the search constant ladder). */
void fm_left_tail (const double* x, double k, double* out, int64_t T, int64_t S, int64_t ldx, int64_t ldo);  /* >k -> NaN */
void fm_right_tail(const double* x, double k, double* out, int64_t T, int64_t S, int64_t ldx, int64_t ldo);  /* <k -> NaN */
/* hump: update only when |x[t] - out[t-1]| > k, else hold previous value -> dampens turnover. Per-column state machine (like trade_when). */
void fm_hump      (const double* x, double k, double* out, int64_t T, int64_t S, int64_t ldx, int64_t ldo);
/* signed_power(x, k) = sign(x)*|x|^k; 0 -> 0 (avoids 0^k = NaN/inf when k < 0). NaN propagates. */
void fm_pow_const(const double* x, double k, double* out, int64_t T, int64_t S, int64_t ldx, int64_t ldo);

/* ===== Rolling unary, per-symbol time series =====
 * std/max/min/skew/kurt/mad/chanpos/ddown/zscore/snr/corr/cov take trailing rng_lo/rng_hi: precomputed valid row range
 * [vlo, vhi] per column (int64 x S); if non-NULL, skips in-column scanning; if NULL, scans in place. Outputs outside
 * the range are necessarily NaN, so trimmed and untrimmed runs match bitwise. Other ts operators do not trim. */
void fm_ts_mean (const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo);
void fm_ts_std  (const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo, const int64_t* rng_lo, const int64_t* rng_hi);
void fm_ts_sum  (const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo);
void fm_ts_max  (const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo, const int64_t* rng_lo, const int64_t* rng_hi);
void fm_ts_min  (const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo, const int64_t* rng_lo, const int64_t* rng_hi);
void fm_ts_ref  (const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo);  /* x.shift(w) */
void fm_ts_delta(const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo);  /* x - x.shift(w) */
void fm_ts_ema  (const double* x, double* out, int64_t T, int64_t S, int64_t span, int64_t ldx, int64_t ldo);
void fm_ts_wma  (const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo);
void fm_ts_rank   (const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo);  /* in [0,1] */
void fm_ts_arg_max(const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo);  /* bars-since-max in [0,w-1] */
void fm_ts_arg_min(const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo);
void fm_ts_skew   (const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo, const int64_t* rng_lo, const int64_t* rng_hi);  /* 3rd standardized moment */
void fm_ts_kurt   (const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo, const int64_t* rng_lo, const int64_t* rng_hi);  /* excess kurtosis */
void fm_ts_mad    (const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo, const int64_t* rng_lo, const int64_t* rng_hi);  /* mean abs deviation */
void fm_ts_slope  (const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo);  /* OLS slope vs t; incremental sum drifts with start row, so no external range */
/* Per-column rolling min/max (dual monotonic deques, O(1) amortised; implemented in ops_ts.cpp).
 * Shared by chanpos / ddown / ts_scale for window extrema.
 * Caller supplies scratch: w int64s each for dq_lo/dq_hi, T elements each for mn/mx/ok.
 * ok[t] == 1 iff window is full and NaN-free; mn/mx are undefined when ok[t] == 0. */
void fm_win_minmax_col(const double* x, int64_t T, int64_t ldx, int64_t s, int64_t w,
                       const int64_t* rng_lo, const int64_t* rng_hi,
                       int64_t* dq_lo, int64_t* dq_hi,
                       double* mn, double* mx, unsigned char* ok);

/* Fused trend primitives (multi-step composition in a single pass). */
void fm_ts_chanpos(const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo, const int64_t* rng_lo, const int64_t* rng_hi);  /* Donchian channel position (x-min)/(max-min) in [0,1] */
void fm_ts_ddown  (const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo, const int64_t* rng_lo, const int64_t* rng_hi);  /* drawdown from rolling high x/max-1 <= 0 (nearness-to-high) */
void fm_ts_zscore (const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo, const int64_t* rng_lo, const int64_t* rng_hi);  /* rolling z = (x-mean)/std, ddof=0 */
void fm_ts_snr    (const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo, const int64_t* rng_lo, const int64_t* rng_hi);  /* trend signal-to-noise mean/std (rolling t-stat) */

/* ===== BRAIN time_series family (implemented in ops_inject_ts.cpp) =====
 * First three match standard ts signature (kind='ts'); the rest take an additional continuous scalar parameter. */
void fm_ts_backfill(const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo);  /* last valid value within d rows (no min_periods) */
void fm_ts_median  (const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo);  /* rolling median */
void fm_ts_av_diff (const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo);  /* x - mean(x,d), mean ignores NaN (BRAIN definition) */
/* panel + window + scalar (kind='ts_param') */
void fm_ts_decay_exp_window(const double* x, double* out, int64_t T, int64_t S, int64_t w, double f, int64_t ldx, int64_t ldo);  /* normalised weights f^i */
void fm_ts_scale           (const double* x, double* out, int64_t T, int64_t S, int64_t w, double k, int64_t ldx, int64_t ldo);  /* chanpos + k */
void fm_ts_quantile        (const double* x, double* out, int64_t T, int64_t S, int64_t w, double driver, int64_t ldx, int64_t ldo);  /* 0=gaussian 1=uniform 2=cauchy */
/* panel + panel + window + scalar (kind='pair_param') */
void fm_ts_regression(const double* y, const double* x, double* out, int64_t T, int64_t S, int64_t w, double rettype, int64_t ldy, int64_t ldx, int64_t ldo);  /* 0=resid 1=intercept 2=slope 3=fitted 4=R^2 */

/* ===== Rolling pair ===== */
void fm_ts_corr(const double* a, const double* b, double* out, int64_t T, int64_t S, int64_t w, int64_t lda_, int64_t ldb, int64_t ldo, const int64_t* rng_lo, const int64_t* rng_hi);
void fm_ts_cov (const double* a, const double* b, double* out, int64_t T, int64_t S, int64_t w, int64_t lda_, int64_t ldb, int64_t ldo, const int64_t* rng_lo, const int64_t* rng_hi);

/* ===== Cross-sectional ===== */
/* Per row t across S symbols: average-tie rank normalised to [0, 1]. NaN cells stay NaN. */
void fm_cs_rank   (const double* x, double* out, int64_t T, int64_t S);
/* Per row t across S symbols: z-score (x - mean) / std (ddof=1). NaN cells stay NaN. */
void fm_cs_zscore (const double* x, double* out, int64_t T, int64_t S);
/* Per row t across S symbols: demean x - row_mean. NaN cells stay NaN. */
void fm_cs_demean (const double* x, double* out, int64_t T, int64_t S);
/* Per row t across S symbols: L1-normalise x / sum_s |x|. NaN cells stay NaN. */
void fm_cs_scale  (const double* x, double* out, int64_t T, int64_t S);

#ifdef __cplusplus
}
#endif

#endif
