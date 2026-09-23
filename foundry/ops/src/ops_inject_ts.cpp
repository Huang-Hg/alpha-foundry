/* foundry/ops/src/ops_inject_ts.cpp — BRAIN time_series 族时序算子
 *
 * 与 ops_ts.cpp 的算子同 layout 约定((T,S) 行优先 double,ld 行步长,NaN 空缺)、
 * 同 rolling 语义(min_periods=w;窗口内任一 NaN → 输出 NaN;t<w-1 暖机输出 NaN),
 * 唯二例外在各自注释里写明(ts_av_diff / ts_backfill 按 BRAIN 定义忽略 NaN)。
 */

#include "ops.hpp"

#define _USE_MATH_DEFINES        /* MSVC/ucrt 下 M_PI 需此宏(须在 math.h 之前) */
#include <math.h>                /* 与 ops_xs.cpp 一致:非限定 isnan / M_PI */
#include <vector>
#include <algorithm>

/* ===================== 无额外参数(kind='ts',签名同既有 ts 算子) ===================== */

/* ts_backfill(x, d):x[t] 有效则原样;否则回看至多 d-1 行取最近的有效值,找不到 → NaN。
 * [BRAIN] "Returns the first valid value of the input x by looking through lookback days (d)"。
 * **刻意不套 min_periods 语义** —— 它的用途就是填 NaN,套了等于自我否定。因果:只回看。 */
void fm_ts_backfill(const double* x, double* out, int64_t T, int64_t S, int64_t w,
                    int64_t ldx, int64_t ldo) {
    /* "最近有效值"只需**记住上一个有效行号**—— 沿 t 前进时它天然单调更新 ⇒ O(1)/点,
     * 无需每遇 NaN 回扫 O(w)。这个算子只搬运数值,不做任何算术。 */
    #pragma omp parallel for schedule(static)
    for (int64_t s = 0; s < S; s++) {
        int64_t last = -1;                        /* 最近一个有效行的行号(必 < t) */
        for (int64_t t = 0; t < T; t++) {
            double v = x[t * ldx + s];
            if (!isnan(v)) { last = t; out[t * ldo + s] = v; continue; }
            /* 回看 [t−w+1, t−1] ⇔ 距今 ≤ w−1 行 */
            out[t * ldo + s] = (last >= 0 && t - last <= w - 1) ? x[last * ldx + s] : NAN;
        }
    }
}

/* ts_median(x, d):滚动中位数。min_periods=w,窗口内任一 NaN → NaN(与 ts_mean 一致)。
 *
 * 维持一个**窗内有效值的有序缓冲**:每步二分定位后 memmove 删旧、memmove 插新,NaN 用滑动
 * cnt_nan 计数。渐近 O(w),但 memmove 是 SIMD 连续搬运,常数比每点 nth_element 小一个量级。
 *
 * 中位数是精确选择:奇数窗取 srt[mid],偶数窗取 0.5*(srt[mid] + srt[mid-1]),
 * 与 nth_element 写法选出的是同一对 double,结果逐位相同。 */
void fm_ts_median(const double* x, double* out, int64_t T, int64_t S, int64_t w,
                  int64_t ldx, int64_t ldo) {
    #pragma omp parallel for schedule(static)
    for (int64_t s = 0; s < S; s++) {
        std::vector<double> srt;                  /* 窗内有效值,升序 */
        srt.reserve((size_t)w);
        int64_t cnt_nan = 0;
        for (int64_t t = 0; t < T; t++) {
            double v = x[t * ldx + s];            /* 1. 新值入窗 */
            if (isnan(v)) cnt_nan++;
            else srt.insert(std::lower_bound(srt.begin(), srt.end(), v), v);
            if (t >= w) {                         /* 2. 旧值出窗 */
                double old = x[(t - w) * ldx + s];
                if (isnan(old)) cnt_nan--;
                else srt.erase(std::lower_bound(srt.begin(), srt.end(), old));
            }
            if (t < w - 1 || cnt_nan > 0) { out[t * ldo + s] = NAN; continue; }
            size_t mid = (size_t)w / 2;           /* 此时 srt.size() == w */
            double m = srt[mid];
            if (w % 2 == 0) m = 0.5 * (m + srt[mid - 1]);
            out[t * ldo + s] = m;
        }
    }
}

/* ts_av_diff(x, d) = x − mean(x over d)。[BRAIN] "deals with NaNs carefully. That is NaNs
 * are ignored during mean computation" ⇒ **均值忽略 NaN**(与 ts_mean 的传播语义相反,
 * 这是 BRAIN 定义,不是笔误)。窗口内全 NaN → NaN;x[t] 本身 NaN → NaN。 */
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


/* ===================== panel + window + 标量(kind='ts_param') ===================== */

/* ts_decay_exp_window(x, d, factor=f):过去 d 行的指数衰减加权平均,
 * 权重 w_i = f^i(i=0 为当前行),归一化。f=1 退化为等权均值;f∈(0,1) 越小越重视近期。
 * min_periods=d,窗口内任一 NaN → NaN(与既有 rolling 一致)。 */
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
            for (int64_t i = 0; i < w; i++) {          /* i=0 → 当前行 */
                double v = x[(t - i) * ldx + s];
                if (isnan(v)) { bad = true; break; }
                acc += wt[(size_t)i] * v;
            }
            out[t * ldo + s] = bad ? NAN : (acc / wsum);
        }
    }
}

/* ts_scale(x, d, constant=k) = (x − ts_min(x,d)) / (ts_max(x,d) − ts_min(x,d)) + k。
 * [BRAIN] 时序版 scale-down。分母 0(窗内恒值)→ NaN,与 ts_chanpos 的既有约定一致。
 * 注:k=0 时逐字等于既有的 ts_chanpos;k 存在的意义是把 [0,1] 平移出 clip 边界。 */
void fm_ts_scale(const double* x, double* out, int64_t T, int64_t S, int64_t w,
                 double k, int64_t ldx, int64_t ldo) {
    /* 大窗的窗内极值走 ops_ts.cpp 的双单调 deque(与 chanpos/ddown 同一份实现);
     * min/max 是精确选择,两支逐位一致。
     * 注:helper 的窗满/NaN 判据(t−vlo ≥ w−1 且窗内无 NaN)与小窗支的
     * `t ≥ w−1` + bad 标志等价 —— 窗口若越过列首全 NaN 段,两种写法都出 NaN。 */
    if (w < 256) {                 /* 小窗重扫更快,阈值与 ops_ts.cpp 的 TS_MINMAX_W_MIN 同源 */
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

/* 逆正态 CDF(Acklam 有理逼近 + 一次 Halley 精修,|误差| < 1e-15)。
 * ts_quantile 的 gaussian driver 用。标准库无 erfinv,故自带。 */
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
    /* Halley 精修:e = Φ(x) − p,u = e·√(2π)·exp(x²/2) */
    double e = 0.5 * erfc(-xv / sqrt(2.0)) - p;
    double u = e * sqrt(2.0 * M_PI) * exp(xv * xv / 2.0);
    return xv - u / (1.0 + xv * u / 2.0);
}

/* ts_quantile(x, d, driver):窗内序数位置 → 经 driver 分布的逆 CDF 映射。
 * driver 编码 0=gaussian(默认)/ 1=uniform / 2=cauchy —— BRAIN 用字符串,DSL 里用数值码。
 *
 * **端点约定是这个算子的要害**:窗内最小/最大值的经验分位是 0 / 1,直接送进逆 CDF 得 ∓inf。
 * 采用绘图位置 p = (r + 0.5) / d(r = 0-based 平均秩 ∈ [0, d−1]),把 p 收进
 * [1/(2d), 1−1/(2d)] ⇒ 尾部有限且对称。本仓曾因端点约定处理不当在 rank 族上差 0.40 Sharpe。
 * 并列值取平均秩(与既有 fm_ts_rank 一致)。min_periods=d,窗内任一 NaN → NaN。 */
static inline void _q_bit_add(int64_t* bit, int64_t n, int64_t i, int64_t d) {
    for (; i <= n; i += i & -i) bit[i] += d;
}
static inline int64_t _q_bit_prefix(const int64_t* bit, int64_t i) {
    int64_t s = 0;
    for (; i > 0; i -= i & -i) s += bit[i];
    return s;
}

/* 序数位置 → driver 逆 CDF。less/eq 是整数计数,由谁数出来不影响后面的浮点运算。 */
static inline double _q_emit(int64_t less, int64_t eq, int64_t w, int dr) {
    double r = (double)less + 0.5 * (double)(eq - 1);   /* 并列取平均秩 */
    double p = (r + 0.5) / (double)w;                    /* 绘图位置,两端有限 */
    if (dr == 1) return p;                                      /* uniform */
    if (dr == 2) return tan(M_PI * (p - 0.5));                  /* cauchy  */
    return fm_ndtri(p);                                         /* gaussian */
}

void fm_ts_quantile(const double* x, double* out, int64_t T, int64_t S, int64_t w,
                    double driver, int64_t ldx, int64_t ldo) {
    /* 大窗走 **精确值压缩 + Fenwick BIT**(O(log M)/点,替代逐点 O(w) 计数)。
     *
     * 与 fm_ts_rank 的 BIT 版**不同**:那边为提速用了 128 个分位桶(rank 误差 ±1/(2B)),
     * 这里压缩到**精确的不同值**上 —— less/eq 与逐点扫描数出来的是同一对整数,
     * 后续 r/p/逆 CDF 的浮点运算相同 ⇒ 两支逐位一致。近似在这里不可接受:
     * quantile 的读数会直接进敞口。
     *
     * 小窗(w<32)逐点扫描:BIT 每步 4 次 O(log M)≈48 次操作,反而不如扫 5~31 个数,
     * 分派点与 fm_ts_rank 一致。 */
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
        /* 1. 逐列压缩:有限值排序去重 → comp[t] ∈ {1..M},NaN → 0 */
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
        /* 2. 滑窗:BIT 计数 + 旁路 cnt_nan(与 _ts_rank_bucket 同结构) */
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


/* ===================== panel + panel + window + 标量(kind='pair_param') ===================== */

/* ts_regression(y, x, d, rettype):窗内 y ~ α + β·x 的滚动 OLS。
 * min_periods=d,窗内任一 NaN(任一侧)→ NaN;分母 0(x 窗内恒值)→ NaN。
 *
 * rettype 编码:0=残差 y−ŷ · 1=截距 α · 2=斜率 β · 3=拟合值 ŷ · 4=R²。
 * ⚠️ **0–3 按 BRAIN 文档化的含义命名,但本仓知识库未收录 BRAIN 的 rettype 数值对照表**,
 * 故不保证与 BRAIN 平台逐码一致;4(R²)是本仓自定义扩展。跨平台移植表达式时须自行核对。
 *
 * 实现取**每点重扫 O(d)** 而非滑动和:滑动和要处理"窗内出现 NaN 需重置累加器"的分支,
 * 而抵消误差在 Syy/Sxx 上恰是最难控的。本算子用在少量深加工候选上(不是海选热路径),
 * 正确性优先。若日后进热路径再换滑动和,须先对拍本实现。 */
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
                default: v = yt - fit; break;      /* 0 = 残差 */
            }
            out[t * ldo + s] = v;
        }
    }
}
