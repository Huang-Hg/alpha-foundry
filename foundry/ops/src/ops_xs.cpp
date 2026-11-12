/* foundry/ops/src/ops_xs.cpp — elementwise (windowless) and cross-sectional operators
 *
 * unary:        fm_abs / fm_neg / fm_sign / fm_log / fm_sqrt_ / fm_square / fm_tanh / fm_inv / fm_s_log_1p / fm_not / fm_is_nan
 * binary:       fm_add / fm_sub / fm_mul / fm_div / fm_max_b / fm_min_b / fm_gt / fm_lt / logical family
 * binary_const: fm_add_const / fm_mul_const / fm_pow_const (signed_power) / fm_left_tail / fm_right_tail / fm_hump
 * ternary:      fm_if_else / fm_trade_when
 * cross-sectional: fm_cs_rank / fm_cs_zscore / fm_cs_demean / fm_cs_scale
 *
 * Design principles:
 *   - Arithmetic ops (add/sub/mul, unary abs/neg/square/tanh): IEEE 754 NaN propagates implicitly without explicit
 *     isnan branches -> auto-vectorised under -O3 -march=native.
 *   - Comparison/selection ops (max/min/log/sqrt/inv/sign): NaN comparisons evaluate to false and would swallow NaN
 *     into 0 or the wrong branch, so explicit isnan checks are required.
 *   - div/pow_const: explicit branches for special values (b == 0, base == 0); IEEE propagates NaN elsewhere.
 *   - build.sh must retain -fno-finite-math-only; otherwise GCC >= 14 optimises away isnan/isfinite.
 *   - 2D strided addressing (ldx/ldo row strides): elementwise expressions are independent of ld; ld == S is C-contiguous.
 */

#include "ops.hpp"

#include <math.h>
#include <stdint.h>

#include <algorithm>    /* std::sort */
#include <utility>      /* std::pair */
#include <vector>       /* per-thread scratch */


/* ----- unary ----- */

#define FM_UNARY(name, expr) \
void name(const double* x, double* out, int64_t T, int64_t S, int64_t ldx, int64_t ldo) { \
    _Pragma("omp parallel for schedule(static)") \
    for (int64_t t = 0; t < T; t++) { \
        const double* xr  = x   + t * ldx; \
        double*       outr = out + t * ldo; \
        for (int64_t s = 0; s < S; s++) { \
            double v = xr[s]; \
            outr[s] = (expr); \
        } \
    } \
}

FM_UNARY(fm_abs,    fabs(v))
FM_UNARY(fm_neg,    -v)
FM_UNARY(fm_sign,   isnan(v) ? NAN : (v > 0.0) ? 1.0 : (v < 0.0) ? -1.0 : 0.0)
FM_UNARY(fm_log,    (isnan(v) || v <= 0.0) ? NAN : log(v))
FM_UNARY(fm_sqrt_,  (isnan(v) || v < 0.0) ? NAN : sqrt(v))
FM_UNARY(fm_square, isnan(v) ? NAN : v * v)
FM_UNARY(fm_tanh,   isnan(v) ? NAN : tanh(v))
FM_UNARY(fm_inv,    (isnan(v) || v == 0.0) ? NAN : (1.0 / v))
/* s_log_1p: sign(x) * log(1 + |x|), compresses heavy tails while preserving sign; defined on all reals. */
FM_UNARY(fm_s_log_1p, isnan(v) ? NAN : (v >= 0.0) ? log1p(v) : -log1p(-v))


/* ----- binary ----- */

#define FM_BIN_ARITH(name, expr) \
void name(const double* a, const double* b, double* out, int64_t T, int64_t S, \
          int64_t lda, int64_t ldb, int64_t ldo) { \
    _Pragma("omp parallel for schedule(static)") \
    for (int64_t t = 0; t < T; t++) { \
        const double* ar  = a   + t * lda; \
        const double* br  = b   + t * ldb; \
        double*       outr = out + t * ldo; \
        for (int64_t s = 0; s < S; s++) { \
            double va = ar[s], vb = br[s]; \
            outr[s] = (expr); \
        } \
    } \
}

#define FM_BIN_NAN(name, expr) \
void name(const double* a, const double* b, double* out, int64_t T, int64_t S, \
          int64_t lda, int64_t ldb, int64_t ldo) { \
    _Pragma("omp parallel for schedule(static)") \
    for (int64_t t = 0; t < T; t++) { \
        const double* ar  = a   + t * lda; \
        const double* br  = b   + t * ldb; \
        double*       outr = out + t * ldo; \
        for (int64_t s = 0; s < S; s++) { \
            double va = ar[s], vb = br[s]; \
            outr[s] = (isnan(va) || isnan(vb)) ? NAN : (expr); \
        } \
    } \
}

FM_BIN_ARITH(fm_add, va + vb)
FM_BIN_ARITH(fm_sub, va - vb)
FM_BIN_ARITH(fm_mul, va * vb)
/* div: vb == 0.0 must be checked explicitly (IEEE 754 div-by-0 yields +-Inf, whereas we require NaN); NaN in va/vb propagates naturally. */
FM_BIN_ARITH(fm_div, (vb == 0.0) ? NAN : (va / vb))

FM_BIN_NAN(fm_max_b, (va > vb) ? va : vb)
FM_BIN_NAN(fm_min_b, (va < vb) ? va : vb)
/* Conditional gates: 0/1 indicator output, NaN propagates. */
FM_BIN_NAN(fm_gt,    (va > vb) ? 1.0 : 0.0)
FM_BIN_NAN(fm_lt,    (va < vb) ? 1.0 : 0.0)


/* ---------- Ternary bool combinators (event-driven entry/exit) ---------- */

/* if_else: cond NaN -> NaN; cond > 0 -> a, else b (NaN in the selected branch propagates naturally). */
void fm_if_else(const double* c, const double* a, const double* b, double* out,
                int64_t T, int64_t S, int64_t ldc, int64_t lda, int64_t ldb, int64_t ldo) {
    #pragma omp parallel for schedule(static)
    for (int64_t t = 0; t < T; t++) {
        const double* cr  = c + t * ldc;
        const double* ar  = a + t * lda;
        const double* br  = b + t * ldb;
        double*       outr = out + t * ldo;
        for (int64_t s = 0; s < S; s++) {
            double vc = cr[s];
            outr[s] = isnan(vc) ? NAN : (vc > 0.0 ? ar[s] : br[s]);
        }
    }
}

/* trade_when (WorldQuant semantics, per-column state machine, causal: depends only on <= t):
 *   exit > 0 -> 0; else if entry > 0 -> alpha[t]; else hold previous value (0 before t = 0).
 *   NaN in entry/exit evaluates > 0 as false -> holds; NaN in alpha propagates on entry (cleared on exit). */
void fm_trade_when(const double* e, const double* a, const double* x, double* out,
                   int64_t T, int64_t S, int64_t lde, int64_t lda, int64_t ldx, int64_t ldo) {
    #pragma omp parallel for schedule(static)
    for (int64_t s = 0; s < S; s++) {
        double prev = 0.0;
        for (int64_t t = 0; t < T; t++) {
            double v = (x[t * ldx + s] > 0.0) ? 0.0
                     : ((e[t * lde + s] > 0.0) ? a[t * lda + s] : prev);
            out[t * ldo + s] = v;
            prev = v;
        }
    }
}


/* ---------- BRAIN logical family (complete conditional logic) ----------
 * NaN convention matches gt/lt (NaN on either side -> NaN), except fm_is_nan — whose purpose is testing for NaN,
 * so propagating NaN would prevent it from ever returning 1. */

FM_UNARY(fm_not,    isnan(v) ? NAN : ((v > 0.0) ? 0.0 : 1.0))
FM_UNARY(fm_is_nan, isnan(v) ? 1.0 : 0.0)

FM_BIN_NAN(fm_and,           (va > 0.0 && vb > 0.0) ? 1.0 : 0.0)
FM_BIN_NAN(fm_or,            (va > 0.0 || vb > 0.0) ? 1.0 : 0.0)
FM_BIN_NAN(fm_equal,         (va == vb) ? 1.0 : 0.0)
FM_BIN_NAN(fm_not_equal,     (va != vb) ? 1.0 : 0.0)
FM_BIN_NAN(fm_greater_equal, (va >= vb) ? 1.0 : 0.0)
FM_BIN_NAN(fm_less_equal,    (va <= vb) ? 1.0 : 0.0)


/* ----- binary_const (panel op scalar k, 1 SIMD pass) ----- */

#define FM_BIN_CONST(name, expr) \
void name(const double* x, double k, double* out, int64_t T, int64_t S, \
          int64_t ldx, int64_t ldo) { \
    _Pragma("omp parallel for schedule(static)") \
    for (int64_t t = 0; t < T; t++) { \
        const double* xr  = x   + t * ldx; \
        double*       outr = out + t * ldo; \
        for (int64_t s = 0; s < S; s++) { \
            double v = xr[s]; \
            outr[s] = (expr); \
        } \
    } \
}

FM_BIN_CONST(fm_add_const, v + k)
FM_BIN_CONST(fm_mul_const, v * k)

/* panel op continuous scalar (BRAIN transformational family): tail truncation. NaN stays NaN naturally
 * (NaN comparisons are false -> takes the else branch and returns v = NaN). */
FM_BIN_CONST(fm_left_tail,  (v > k) ? NAN : v)   /* NaN above maximum, keeping the left tail */
FM_BIN_CONST(fm_right_tail, (v < k) ? NAN : v)   /* NaN below minimum, keeping the right tail */

/* hump(x, k) (BRAIN time_series family): update only when |x[t] - out[t-1]| > k, else hold previous value -> dampens turnover.
 * Per-column state machine, causal (depends only on <= t), independent across columns -> OpenMP over columns (like fm_trade_when).
 * When previous state is undefined, the first finite value seeds it; x[t] == NaN is not a move and holds previous value.
 * k < 0 updates every step (degenerates to identity with forward-fill on NaN), left to caller semantics. */
void fm_hump(const double* x, double k, double* out, int64_t T, int64_t S,
             int64_t ldx, int64_t ldo) {
    #pragma omp parallel for schedule(static)
    for (int64_t s = 0; s < S; s++) {
        double prev = NAN;
        for (int64_t t = 0; t < T; t++) {
            double v = x[t * ldx + s];
            if (!isnan(v) && (isnan(prev) || fabs(v - prev) > k)) prev = v;
            out[t * ldo + s] = prev;
        }
    }
}

/* signed_power inner loop macro: 0 -> 0 (prevents 0^k = inf/NaN when k < 0); otherwise sign(v)*P(av), av = |v|.
 * NaN propagates naturally through fabs/P (fabs(NaN) = NaN); the zero check cannot be omitted. P is specialised per k. */
#define FM_SIGNED_POW_LOOP(P) \
    _Pragma("omp parallel for schedule(static)") \
    for (int64_t t = 0; t < T; t++) { \
        const double* xr  = x   + t * ldx; \
        double*       outr = out + t * ldo; \
        for (int64_t s = 0; s < S; s++) { \
            double v = xr[s]; \
            double av = fabs(v); \
            if (av == 0.0) { outr[s] = 0.0; continue; } \
            double p = (P); \
            outr[s] = (v < 0.0) ? -p : p; \
        } \
    }

void fm_pow_const(const double* x, double k, double* out, int64_t T, int64_t S,
                  int64_t ldx, int64_t ldo) {
    /* Specialise common exponents k in {+-2, +-0.5} with mul/sqrt instead of libm pow (~5-11x faster). Bitwise identical:
     * av*av == pow(av, 2) exactly; sqrt == pow(av, 0.5) under IEEE rounding; negative exponents take the reciprocal.
     * All other k fall back to generic pow. */
    if      (k ==  2.0) { FM_SIGNED_POW_LOOP(av * av) }
    else if (k == -2.0) { FM_SIGNED_POW_LOOP(1.0 / (av * av)) }
    else if (k ==  0.5) { FM_SIGNED_POW_LOOP(sqrt(av)) }
    else if (k == -0.5) { FM_SIGNED_POW_LOOP(1.0 / sqrt(av)) }
    else                { FM_SIGNED_POW_LOOP(pow(av, k)) }
}

/* ---------- Cross-sectional ----------
 * One cross-sectional reduction per row t, OpenMP-parallel across independent rows t.
 * cs_rank: compact finite values, std::sort ascending by value, assign average rank to equal-value runs.
 * Tie-invariant: intra-group order does not affect output (all ties receive the same average rank) -> value-only compare is bitwise deterministic.
 */

/* (value, orig_s) pair; std::sort ascending by value (tie order unspecified, output tie-invariant). */
typedef std::pair<double, int64_t> _vi_pair;

static void cs_rank_one_row(const double* row, double* out_row, int64_t S, _vi_pair* buf) {
    int64_t n_valid = 0;
    for (int64_t s = 0; s < S; s++) {
        double v = row[s];
        if (!isnan(v)) buf[n_valid++] = { v, s };
    }
    if (n_valid < 2) {
        for (int64_t s = 0; s < S; s++) out_row[s] = NAN;
        return;
    }
    std::sort(buf, buf + n_valid,
              [](const _vi_pair& a, const _vi_pair& b) { return a.first < b.first; });
    /* Average rank across tied runs (independent of intra-tie order -> bit-identical). */
    int64_t i = 0;
    while (i < n_valid) {
        int64_t j = i;
        while (j + 1 < n_valid && buf[j+1].first == buf[i].first) j++;
        double avg_norm = ((double)(i + j) / 2.0) / (double)(n_valid - 1);
        for (int64_t k = i; k <= j; k++) out_row[buf[k].second] = avg_norm;
        i = j + 1;
    }
    /* Restore NaN positions. */
    for (int64_t s = 0; s < S; s++) {
        if (isnan(row[s])) out_row[s] = NAN;
    }
}

void fm_cs_rank(const double* x, double* out, int64_t T, int64_t S) {
    #pragma omp parallel
    {
        std::vector<_vi_pair> buf(S);            /* per-thread scratch (RAII) */
        #pragma omp for schedule(static)
        for (int64_t t = 0; t < T; t++)
            cs_rank_one_row(&x[t * S], &out[t * S], S, buf.data());
    }
}

void fm_cs_zscore(const double* x, double* out, int64_t T, int64_t S) {
    #pragma omp parallel for schedule(static)
    for (int64_t t = 0; t < T; t++) {
        double sum = 0.0, sum_sq = 0.0;
        int64_t n = 0;
        for (int64_t s = 0; s < S; s++) {
            double v = x[t * S + s];
            if (!isnan(v)) { sum += v; sum_sq += v * v; n++; }
        }
        if (n < 2) {
            for (int64_t s = 0; s < S; s++) out[t * S + s] = NAN;
            continue;
        }
        double mean = sum / (double)n;
        double var  = (sum_sq - mean * sum) / (double)(n - 1);
        if (var <= 0.0) {
            for (int64_t s = 0; s < S; s++) out[t * S + s] = NAN;
            continue;
        }
        double sd = sqrt(var);
        for (int64_t s = 0; s < S; s++) {
            double v = x[t * S + s];
            out[t * S + s] = isnan(v) ? NAN : ((v - mean) / sd);
        }
    }
}


void fm_cs_demean(const double* x, double* out, int64_t T, int64_t S) {
    #pragma omp parallel for schedule(static)
    for (int64_t t = 0; t < T; t++) {
        double sum = 0.0;
        int64_t n = 0;
        for (int64_t s = 0; s < S; s++) {
            double v = x[t * S + s];
            if (!isnan(v)) { sum += v; n++; }
        }
        if (n == 0) {
            for (int64_t s = 0; s < S; s++) out[t * S + s] = NAN;
            continue;
        }
        double mean = sum / (double)n;
        for (int64_t s = 0; s < S; s++) {
            double v = x[t * S + s];
            out[t * S + s] = isnan(v) ? NAN : (v - mean);
        }
    }
}


/* cs_scale: cross-sectional L1 normalisation = x[t,s] / sum_s |x[t,s]| (standard scaling for dollar-neutral signals). */
void fm_cs_scale(const double* x, double* out, int64_t T, int64_t S) {
    #pragma omp parallel for schedule(static)
    for (int64_t t = 0; t < T; t++) {
        double sum_abs = 0.0;
        for (int64_t s = 0; s < S; s++) {
            double v = x[t * S + s];
            if (!isnan(v)) sum_abs += fabs(v);
        }
        if (sum_abs <= 1e-12) {
            for (int64_t s = 0; s < S; s++) out[t * S + s] = NAN;
            continue;
        }
        for (int64_t s = 0; s < S; s++) {
            double v = x[t * S + s];
            out[t * S + s] = isnan(v) ? NAN : (v / sum_abs);
        }
    }
}

