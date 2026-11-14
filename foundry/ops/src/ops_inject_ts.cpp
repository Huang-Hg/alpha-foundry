/* foundry/ops/src/ops_inject_ts.cpp — BRAIN time_series family rolling operators
 *
 * Shares the layout convention ((T, S) row-major double, row stride ld, NaN for missing) and
 * rolling semantics (min_periods=w; any NaN in window -> output NaN; warmup t < w-1 -> output NaN)
 * with ops_ts.cpp. The only two exceptions are documented at their definitions (ts_av_diff and
 * ts_backfill ignore NaNs per BRAIN specification).
 */

#include "ops.hpp"

#define _USE_MATH_DEFINES        /* Needed for M_PI on MSVC/ucrt (must precede math.h) */
#include <math.h>                /* Matches ops_xs.cpp: unqualified isnan / M_PI */
#include <vector>
#include <algorithm>

/* ===================== No extra parameters (kind='ts', same signature as existing ts ops) ===================== */

/* ts_backfill(x, d): keep x[t] if valid; otherwise look back up to d-1 rows for the most recent
 * valid value, or NaN if none is found.
 * [BRAIN] "Returns the first valid value of the input x by looking through lookback days (d)".
 * **Deliberately exempt from min_periods** — its purpose is filling NaNs, so requiring a full
 * valid window would defeat it. Causal: strictly backward-looking. */
void fm_ts_backfill(const double* x, double* out, int64_t T, int64_t S, int64_t w,
                    int64_t ldx, int64_t ldo) {
    /* Tracking only the **last valid row index** updates monotonically along t => O(1)/point,
     * avoiding an O(w) backward scan at every NaN. Pure value transport, no arithmetic. */
    #pragma omp parallel for schedule(static)
    for (int64_t s = 0; s < S; s++) {
        int64_t last = -1;                        /* Row index of most recent valid value (< t) */
        for (int64_t t = 0; t < T; t++) {
            double v = x[t * ldx + s];
            if (!isnan(v)) { last = t; out[t * ldo + s] = v; continue; }
            /* Look back over [t-w+1, t-1] <=> age <= w-1 rows */
            out[t * ldo + s] = (last >= 0 && t - last <= w - 1) ? x[last * ldx + s] : NAN;
        }
    }
}

/* ts_median(x, d): rolling median. min_periods=w; any NaN in window -> NaN (matches ts_mean).
 *
 * Maintains a **sorted buffer of valid window values**: binary search + memmove erase/insert per step,
 * with sliding cnt_nan counting NaNs. Asymptotically O(w), but memmove is contiguous SIMD memory motion
 * with an order-of-magnitude smaller constant than per-point nth_element.
 *
 * Exact selection: srt[mid] for odd w, 0.5*(srt[mid] + srt[mid-1]) for even w — selects the exact same
 * pair of doubles as nth_element, bitwise identical. */
void fm_ts_median(const double* x, double* out, int64_t T, int64_t S, int64_t w,
                  int64_t ldx, int64_t ldo) {
    #pragma omp parallel for schedule(static)
    for (int64_t s = 0; s < S; s++) {
        std::vector<double> srt;                  /* Valid values in window, ascending */
        srt.reserve((size_t)w);
        int64_t cnt_nan = 0;
        for (int64_t t = 0; t < T; t++) {
            double v = x[t * ldx + s];            /* 1. Insert incoming value */
            if (isnan(v)) cnt_nan++;
            else srt.insert(std::lower_bound(srt.begin(), srt.end(), v), v);
            if (t >= w) {                         /* 2. Evict outgoing value */
                double old = x[(t - w) * ldx + s];
                if (isnan(old)) cnt_nan--;
                else srt.erase(std::lower_bound(srt.begin(), srt.end(), old));
            }
            if (t < w - 1 || cnt_nan > 0) { out[t * ldo + s] = NAN; continue; }
            size_t mid = (size_t)w / 2;           /* srt.size() == w here */
            double m = srt[mid];
            if (w % 2 == 0) m = 0.5 * (m + srt[mid - 1]);
            out[t * ldo + s] = m;
        }
    }
}

/* ts_av_diff(x, d) = x - mean(x over d). [BRAIN] "deals with NaNs carefully. That is NaNs
 * are ignored during mean computation" => **mean ignores NaNs** (opposite to ts_mean's propagation
 * semantics; intentional BRAIN definition, not a typo). All-NaN window -> NaN; x[t] NaN -> NaN. */
void fm_ts_av_diff(const double* x, double* out, int64_t T, int64_t S, int64_t w,
                   int64_t ldx, int64_t ldo) {
    #pragma omp parallel for schedule(static)
    for (int64_t s = 0; s < S; s++) {
        double sum = 0.0;
        int64_t cnt = 0;
        for (int64_t t = 0; t < T; t++) {
            double v = x[t * ldx + s];
            if (!isnan(v)) { sum += v; cnt++; }
            if (t >= w) {
                double old = x[(t - w) * ldx + s];
                if (!isnan(old)) { sum -= old; cnt--; }
            }
            out[t * ldo + s] = (cnt > 0 && !isnan(v)) ? (v - sum / (double)cnt) : NAN;
        }
    }
}


/* ===================== panel + window + scalar (kind='ts_param') ===================== */

/* ts_decay_exp_window(x, d, factor=f): exponentially decaying weighted average over the past d rows,
 * weights w_i = f^i (i=0 is current row), normalised. Degenerates to equal-weight mean at f=1;
 * smaller f in (0, 1) places more weight on recent rows.
 * min_periods=d; any NaN in window -> NaN (matches standard rolling ops). */
void fm_ts_decay_exp_window(const double* x, double* out, int64_t T, int64_t S, int64_t w,
                            double f, int64_t ldx, int64_t ldo) {
    std::vector<double> wt((size_t)w);
    double wsum = 0.0, p = 1.0;
    for (int64_t i = 0; i < w; i++) { wt[(size_t)i] = p; wsum += p; p *= f; }
    #pragma omp parallel for schedule(static)
    for (int64_t s = 0; s < S; s++) {
        for (int64_t t = 0; t < T; t++) {
            if (t < w - 1) { out[t * ldo + s] = NAN; continue; }
            double acc = 0.0; bool bad = false;
            for (int64_t i = 0; i < w; i++) {          /* i=0 -> current row */
                double v = x[(t - i) * ldx + s];
                if (isnan(v)) { bad = true; break; }
                acc += wt[(size_t)i] * v;
            }
            out[t * ldo + s] = bad ? NAN : (acc / wsum);
        }
    }
}

/* ts_scale(x, d, constant=k) = (x - ts_min(x,d)) / (ts_max(x,d) - ts_min(x,d)) + k.
 * [BRAIN] Time-series scale-down. Zero denominator (constant window) -> NaN, matching existing convention.
 * Note: at k=0 the formula matches ts_chanpos; k shifts [0, 1] away from clip boundaries. */
void fm_ts_scale(const double* x, double* out, int64_t T, int64_t S, int64_t w,
                 double k, int64_t ldx, int64_t ldo) {
    /* Large windows delegate window extrema to ops_ts.cpp's dual monotonic deque (shared with chanpos/ddown);
     * min/max are exact selections, so both branches are bitwise identical.
     * Note: the helper's full-window/NaN criterion (t - vlo >= w - 1 and no NaN in window) is equivalent
     * to the small-window branch's `t >= w - 1` + bad flag — if a window straddles a leading all-NaN run,
     * both implementations emit NaN. */
    if (w < 256) {                 /* Rescan is faster on small windows; threshold matches TS_MINMAX_W_MIN in ops_ts.cpp */
        #pragma omp parallel for schedule(static)
        for (int64_t s = 0; s < S; s++) {
            for (int64_t t = 0; t < T; t++) {
                if (t < w - 1) { out[t * ldo + s] = NAN; continue; }
                double lo = INFINITY, hi = -INFINITY; bool bad = false;
                for (int64_t u = t - w + 1; u <= t; u++) {
                    double v = x[u * ldx + s];
                    if (isnan(v)) { bad = true; break; }
                    if (v < lo) lo = v;
                    if (v > hi) hi = v;
                }
                double den = hi - lo;
                out[t * ldo + s] = (bad || den == 0.0) ? NAN
                                 : ((x[t * ldx + s] - lo) / den + k);
            }
        }
        return;
    }
    #pragma omp parallel
    {
        std::vector<int64_t> dq((size_t)(2 * w));
        std::vector<double> mn((size_t)T), mx((size_t)T);
        std::vector<unsigned char> ok((size_t)T);
        #pragma omp for schedule(static)
        for (int64_t s = 0; s < S; s++) {
            fm_win_minmax_col(x, T, ldx, s, w, NULL, NULL,
                              dq.data(), dq.data() + w, mn.data(), mx.data(), ok.data());
            for (int64_t t = 0; t < T; t++) {
                double den = ok[t] ? (mx[t] - mn[t]) : 0.0;
                out[t * ldo + s] = (!ok[t] || den == 0.0) ? NAN
                                 : ((x[t * ldx + s] - mn[t]) / den + k);
            }
        }
    }
}

/* Inverse standard normal CDF (Acklam rational approximation + one Halley refinement, |error| < 1e-15).
 * Used by ts_quantile's gaussian driver. Standard math.h lacks erfinv, so bundled inline. */
static inline double fm_ndtri(double p) {
    static const double a[6] = {-3.969683028665376e+01, 2.209460984245205e+02,
                                -2.759285104469687e+02, 1.383577518672690e+02,
                                -3.066479806614716e+01, 2.506628277459239e+00};
    static const double b[5] = {-5.447609879822406e+01, 1.615858368580409e+02,
                                -1.556989798598866e+02, 6.680131188771972e+01,
                                -1.328068155288572e+01};
    static const double c[6] = {-7.784894002430293e-03, -3.223964580411365e-01,
                                -2.400758277161838e+00, -2.549732539343734e+00,
                                 4.374664141464968e+00,  2.938163982698783e+00};
    static const double d[4] = { 7.784695709041462e-03,  3.224671290700398e-01,
                                 2.445134137142996e+00,  3.754408661907416e+00};
    const double plow = 0.02425, phigh = 1.0 - plow;
    double q, r, xv;
    if (p <= 0.0) return -INFINITY;
    if (p >= 1.0) return  INFINITY;
    if (p < plow) {
        q = sqrt(-2.0 * log(p));
        xv = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
             ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    } else if (p > phigh) {
        q = sqrt(-2.0 * log(1.0 - p));
        xv = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
              ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    } else {
        q = p - 0.5; r = q * q;
        xv = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
             (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0);
    }
    /* Halley refinement: e = Phi(x) - p, u = e * sqrt(2*pi) * exp(x^2 / 2) */
    double e = 0.5 * erfc(-xv / sqrt(2.0)) - p;
    double u = e * sqrt(2.0 * M_PI) * exp(xv * xv / 2.0);
    return xv - u / (1.0 + xv * u / 2.0);
}

/* ts_quantile(x, d, driver): window ordinal position mapped through the driver distribution's inverse CDF.
 * driver encoding: 0=gaussian (default) / 1=uniform / 2=cauchy — BRAIN uses strings, DSL uses numeric codes.
 *
 * **Endpoint convention is critical**: the empirical quantile of the window min/max is 0 / 1, which maps
 * to -/+inf under the inverse CDF. Using plotting position p = (r + 0.5) / d (r = 0-based average rank
 * in [0, d-1]) keeps p inside [1/(2d), 1 - 1/(2d)] => finite, symmetric tails. Improper endpoint handling
 * previously cost 0.40 Sharpe on the rank family in this repo.
 * Ties receive average rank (matching fm_ts_rank). min_periods=d; any NaN in window -> NaN. */
static inline void _q_bit_add(int64_t* bit, int64_t n, int64_t i, int64_t d) {
    for (; i <= n; i += i & -i) bit[i] += d;
}
static inline int64_t _q_bit_prefix(const int64_t* bit, int64_t i) {
    int64_t s = 0;
    for (; i > 0; i -= i & -i) s += bit[i];
    return s;
}

/* Ordinal counts -> driver inverse CDF. less/eq are integer counts, so downstream FP is independent of how they were counted. */
static inline double _q_emit(int64_t less, int64_t eq, int64_t w, int dr) {
    double r = (double)less + 0.5 * (double)(eq - 1);   /* Average rank for ties */
    double p = (r + 0.5) / (double)w;                    /* Plotting position, finite endpoints */
    if (dr == 1) return p;                                      /* uniform */
    if (dr == 2) return tan(M_PI * (p - 0.5));                  /* cauchy  */
    return fm_ndtri(p);                                         /* gaussian */
}

void fm_ts_quantile(const double* x, double* out, int64_t T, int64_t S, int64_t w,
                    double driver, int64_t ldx, int64_t ldo) {
    /* Large windows use **exact value compression + Fenwick BIT** (O(log M)/point instead of O(w) scan).
     *
     * **Unlike** fm_ts_rank's BIT path (which uses 128 quantile buckets with +-1/(2B) rank error for speed),
     * we compress onto **exact distinct values** here — less/eq are the exact same integer pair as per-point
     * scanning, and subsequent r/p/inverse-CDF floating-point ops are identical => bitwise parity between
     * branches. Approximation is unacceptable here because quantile readings feed directly into exposure.
     *
     * Small windows (w < 32) scan per point: 4 BIT ops per step (~48 operations at O(log M)) lose to scanning
     * 5..31 values; dispatch threshold matches fm_ts_rank. */
    int dr = (int)llround(driver);
    if (w < 32) {
        #pragma omp parallel for schedule(static)
        for (int64_t s = 0; s < S; s++) {
            for (int64_t t = 0; t < T; t++) {
                if (t < w - 1) { out[t * ldo + s] = NAN; continue; }
                double cur = x[t * ldx + s];
                if (isnan(cur)) { out[t * ldo + s] = NAN; continue; }
                int64_t less = 0, eq = 0; bool bad = false;
                for (int64_t u = t - w + 1; u <= t; u++) {
                    double v = x[u * ldx + s];
                    if (isnan(v)) { bad = true; break; }
                    if (v < cur) less++;
                    else if (v == cur) eq++;
                }
                out[t * ldo + s] = bad ? NAN : _q_emit(less, eq, w, dr);
            }
        }
        return;
    }
    #pragma omp parallel for schedule(static)
    for (int64_t s = 0; s < S; s++) {
        /* 1. Per-column compression: sort & deduplicate finite values -> comp[t] in {1..M}, NaN -> 0 */
        std::vector<double> vals;
        vals.reserve((size_t)T);
        for (int64_t t = 0; t < T; t++) {
            double v = x[t * ldx + s];
            if (!isnan(v)) vals.push_back(v);
        }
        if (vals.empty()) {
            for (int64_t t = 0; t < T; t++) out[t * ldo + s] = NAN;
            continue;
        }
        std::sort(vals.begin(), vals.end());
        vals.erase(std::unique(vals.begin(), vals.end()), vals.end());
        int64_t M = (int64_t)vals.size();
        std::vector<int64_t> comp((size_t)T, 0);
        for (int64_t t = 0; t < T; t++) {
            double v = x[t * ldx + s];
            if (!isnan(v))
                comp[(size_t)t] =
                    (int64_t)(std::lower_bound(vals.begin(), vals.end(), v) - vals.begin()) + 1;
        }
        /* 2. Sliding window: BIT counts + side-channel cnt_nan. */
        std::vector<int64_t> bit((size_t)M + 2, 0);
        int64_t cnt_nan = 0;
        for (int64_t t = 0; t < T; t++) {
            int64_t c = comp[(size_t)t];
            if (c == 0) cnt_nan++;
            else        _q_bit_add(bit.data(), M, c, +1);
            if (t >= w) {
                int64_t oc = comp[(size_t)(t - w)];
                if (oc == 0) cnt_nan--;
                else         _q_bit_add(bit.data(), M, oc, -1);
            }
            if (t < w - 1 || cnt_nan > 0) { out[t * ldo + s] = NAN; continue; }
            int64_t less = _q_bit_prefix(bit.data(), c - 1);
            int64_t eq   = _q_bit_prefix(bit.data(), c) - less;
            out[t * ldo + s] = _q_emit(less, eq, w, dr);
        }
    }
}


/* ===================== panel + panel + window + scalar (kind='pair_param') ===================== */

/* ts_regression(y, x, d, rettype): rolling OLS of y ~ alpha + beta * x over window d.
 * min_periods=d; any NaN on either side in window -> NaN; zero denominator (constant x in window) -> NaN.
 *
 * rettype encoding: 0=residual y-y_hat . 1=intercept alpha . 2=slope beta . 3=fitted y_hat . 4=R^2.
 * WARNING: **0-3 are named after BRAIN's documented quantities, but BRAIN's numeric rettype table is not
 * recorded in this repo**, so numeric-code parity with BRAIN is not guaranteed; 4 (R^2) is a local extension.
 * Verify codes manually when porting expressions across platforms.
 *
 * Implemented via **per-point O(d) rescan** rather than sliding sums: sliding sums require accumulator-reset
 * branches when NaNs enter the window, and cancellation error is hardest to control on Syy/Sxx. Used on a
 * small number of refined candidates (not a mass-screening hot path), so correctness comes first. Any future
 * sliding-sum replacement must be validated against this reference implementation first. */
void fm_ts_regression(const double* y, const double* x, double* out, int64_t T, int64_t S,
                      int64_t w, double rettype, int64_t ldy, int64_t ldx, int64_t ldo) {
    int rt = (int)llround(rettype);
    #pragma omp parallel for schedule(static)
    for (int64_t s = 0; s < S; s++) {
        for (int64_t t = 0; t < T; t++) {
            if (t < w - 1) { out[t * ldo + s] = NAN; continue; }
            double Sx = 0, Sy = 0, Sxx = 0, Sxy = 0, Syy = 0;
            bool bad = false;
            for (int64_t u = t - w + 1; u <= t; u++) {
                double xv = x[u * ldx + s], yv = y[u * ldy + s];
                if (isnan(xv) || isnan(yv)) { bad = true; break; }
                Sx += xv; Sy += yv; Sxx += xv * xv; Sxy += xv * yv; Syy += yv * yv;
            }
            if (bad) { out[t * ldo + s] = NAN; continue; }
            double n = (double)w;
            double dxx = n * Sxx - Sx * Sx;
            if (dxx == 0.0) { out[t * ldo + s] = NAN; continue; }
            double beta  = (n * Sxy - Sx * Sy) / dxx;
            double alpha = (Sy - beta * Sx) / n;
            double xt = x[t * ldx + s], yt = y[t * ldy + s];
            double fit = alpha + beta * xt;
            double v;
            switch (rt) {
                case 1:  v = alpha; break;
                case 2:  v = beta;  break;
                case 3:  v = fit;   break;
                case 4: {
                    double dyy = n * Syy - Sy * Sy;
                    double cov = n * Sxy - Sx * Sy;
                    v = (dyy == 0.0) ? NAN : (cov * cov) / (dxx * dyy);
                    break;
                }
                default: v = yt - fit; break;      /* 0 = residual */
            }
            out[t * ldo + s] = v;
        }
    }
}
