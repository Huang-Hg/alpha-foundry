/* foundry/ops/src/ops_ts.cpp — Rolling kernels(per-symbol time series)
 *
 * unary: fm_ts_sum / mean / std / mad / skew / kurt / slope / max / min / arg_max / arg_min /
 *        ref / delta / ema / wma / rank / chanpos / ddown / zscore / snr
 * pair:  fm_ts_corr / fm_ts_cov
 *
 * 全部跨 symbol OMP 并行;窗口内增量更新(sum / sum² / 单调 deque / Fenwick BIT)。
 * SIMD AVX2 lane-pack 给 ts_sum/ts_mean(其它 op 串行 t 循环已足够 cache-friendly)。
 *
 * 2D strided 寻址:x[t * ldx + s] / out[t * ldo + s](ld = 行步长,见 ops.hpp)。
 * per-column 的 t 向累计顺序与 ld 无关 → 任意 ld 数值逐位一致。
 */

#include "ops.hpp"

#include <math.h>
#include <stdint.h>

#include <algorithm>    /* std::sort */
#include <vector>       /* per-thread / per-column scratch */

#ifdef _OPENMP
#include <omp.h>
#endif


/* SIMD 启用条件:gcc/clang AVX2 即编进。S = 4·k 时走 4-lane,否则 fall back scalar。 */
#if defined(__AVX2__)
#include <immintrin.h>
#define FM_HAS_AVX2 1
#else
#define FM_HAS_AVX2 0
#endif

#if FM_HAS_AVX2
/* AVX2 lane-pack 工具:给 x[t*ldx + s_block .. s_block+3] 处理 NaN-as-zero +
 * cnt_nan(int64x4)累加。 */
static inline void _fm_unpack_clean(__m256d v, __m256d* out_clean, __m256i* out_inc) {
    /* nan_mask 1.0(全 1)位置 = NaN */
    __m256d nan_mask = _mm256_cmp_pd(v, v, _CMP_UNORD_Q);
    /* clean = nan ? 0 : v */
    *out_clean = _mm256_blendv_pd(v, _mm256_setzero_pd(), nan_mask);
    /* inc = nan_mask ? 1 : 0 */
    __m256i ones = _mm256_set1_epi64x(1);
    *out_inc = _mm256_and_si256(_mm256_castpd_si256(nan_mask), ones);
}

/* ts_sum / ts_mean 的 lane-pack 内核(每次处理 4 个相邻 symbol)。 */
static void rolling_sum_simd_block(const double* x, double* out,
                                   int64_t T, int64_t w,
                                   int64_t s_block, int divide,
                                   int64_t ldx, int64_t ldo) {
    __m256d vsum = _mm256_setzero_pd();
    __m256i vcnt = _mm256_setzero_si256();
    const __m256d vnan_out = _mm256_set1_pd(NAN);
    const __m256d vw = _mm256_set1_pd((double)w);

    /* 暖机 t = 0 .. w-2:累入 sum/cnt,但输出 NaN */
    for (int64_t t = 0; t < w - 1; t++) {
        __m256d v = _mm256_loadu_pd(&x[t * ldx + s_block]);
        __m256d clean; __m256i inc;
        _fm_unpack_clean(v, &clean, &inc);
        vsum = _mm256_add_pd(vsum, clean);
        vcnt = _mm256_add_epi64(vcnt, inc);
        _mm256_storeu_pd(&out[t * ldo + s_block], vnan_out);
    }
    /* 主循环 t ≥ w-1:add 当前 → 输出 → 滑出最老。 */
    for (int64_t t = w - 1; t < T; t++) {
        __m256d v_new = _mm256_loadu_pd(&x[t * ldx + s_block]);
        __m256d clean_new; __m256i inc_new;
        _fm_unpack_clean(v_new, &clean_new, &inc_new);
        vsum = _mm256_add_pd(vsum, clean_new);
        vcnt = _mm256_add_epi64(vcnt, inc_new);

        /* cnt == 0 mask → blend mean / NaN */
        __m256i cnt_zero = _mm256_cmpeq_epi64(vcnt, _mm256_setzero_si256());
        __m256d val = divide ? _mm256_div_pd(vsum, vw) : vsum;
        __m256d res = _mm256_blendv_pd(vnan_out, val, _mm256_castsi256_pd(cnt_zero));
        _mm256_storeu_pd(&out[t * ldo + s_block], res);

        /* 滑出 t - w + 1 (= t-w+1, 当 t = w-1 时索引 = 0;当 t < w-1 不会进这里) */
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

/* 标量 fallback / S 不被 4 整除时的尾部 */
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
    int64_t S_simd = S - (S & 3);    /* 4 的倍数部分 */
    #pragma omp parallel for schedule(static)
    for (int64_t s_block = 0; s_block < S_simd; s_block += 4) {
        rolling_sum_simd_block(x, out, T, w, s_block, divide, ldx, ldo);
    }
    /* 尾部 */
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


/* 逐列有效区间裁剪:宽面板里多数列有大段上市前 / 退市后全 NaN。滚动核只跑 [vlo,vhi] = 列首/末
 * 有效行,之外写 NaN。区间外的输出必为 NaN(窗口含 NaN),故与不裁剪版**逐位一致**(非近似);
 * 扫描成本 = O(跳过的行数),稠密列 O(1)。窗口满判据写作 t-vlo>=w-1(累计自 vlo 起,
 * [0,vlo) 全 NaN 本不贡献 sum,只贡献已滑出的 cnt_nan)。*/
static inline void _col_range(const double* x, int64_t T, int64_t ldx, int64_t s,
                              int64_t* vlo, int64_t* vhi) {
    int64_t l = 0;     while (l < T && isnan(x[l * ldx + s])) l++;
    int64_t h = T - 1; while (h >= l && isnan(x[h * ldx + s])) h--;
    *vlo = l; *vhi = h;
}


/* std = sqrt((Σx² − mean·Σx) / (w−1)),sample。 */
static void rolling_std_impl(const double* x, double* out, int64_t T, int64_t S, int64_t w,
                             int64_t ldx, int64_t ldo, const int64_t* rng_lo, const int64_t* rng_hi) {
    #pragma omp parallel for schedule(static)
    for (int64_t s = 0; s < S; s++) {
        int64_t vlo, vhi;
        if (rng_lo) { vlo = rng_lo[s]; vhi = rng_hi[s]; }   /* 调用方预算的区间 */
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
                if (var < 0.0) var = 0.0;  /* 数值噪声偶尔小负 */
                out[t * ldo + s] = sqrt(var);
            } else {
                out[t * ldo + s] = NAN;
            }
        }
    }
}

void fm_ts_std(const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo, const int64_t* rng_lo, const int64_t* rng_hi) { rolling_std_impl(x, out, T, S, w, ldx, ldo, rng_lo, rng_hi); }


/* ts_mad: 滚动 mean absolute deviation = mean(|x - rolling_mean|)。
 * 比 ts_std 更 robust(L1 vs L2 偏差)。需要 2-pass 但只在 cnt_nan==0 时计算。 */
void fm_ts_mad(const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo, const int64_t* rng_lo, const int64_t* rng_hi) {
    #pragma omp parallel for schedule(static)
    for (int64_t s = 0; s < S; s++) {
        int64_t vlo, vhi;
        if (rng_lo) { vlo = rng_lo[s]; vhi = rng_hi[s]; }   /* 调用方预算的区间 */
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
 * 趋势融合算子 —— 文献对齐的 4 个单节点趋势原语(用现有算子拼需 ≥4 个节点,短公式预算内不可达):
 *   ts_chanpos  Donchian 通道位置 (x−min)/(max−min) ∈[0,1](Zarattini 突破集成)
 *   ts_ddown    距滚动高点回撤 x/max−1 ≤0(George-Hwang nearness-to-high)
 *   ts_zscore   滚动 z=(x−μ)/σ(归一趋势;ddof=0)
 *   ts_snr      趋势信噪比 μ/σ(rolling t-stat;AdaptiveTrend 趋势质量门)
 * 实现同 ts_mad 模式:滑动 NaN 计数 + 有效窗 O(w) 重扫(数值稳,免增量幂和漂移),
 * 窗含 NaN → NaN;退化窗(极差/σ≈0)→ chanpos=0.5,zscore/snr/ddown=NaN。
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
 * 逐列滚动 min/max —— 双单调 deque,O(1) 摊销。chanpos / ddown / ts_scale 共用(大窗时)。
 *
 * 与窗内重扫**逐位一致**:min/max 是精确选择(挑出窗内的某一个数),不做任何算术 ——
 * deque 与重扫挑出的是同一个 double,下游 (v−mn)/(mx−mn) 逐字相同。
 * (对比:zscore/snr/skew/kurt 的窗内累加**不能**这样换 —— 改成增量会挪动浮点运算顺序,
 *  而 CPU f64 是全系统的黄金参照。)
 *
 * ok[t]=1 ⇔ 窗满且窗内无 NaN,判据与 TS_FUSED_LOOP 逐字一致(含 rng 裁剪与
 * `t - vlo >= w-1` 的累计起点约定)。deque scratch 由调用方给,各 w 个 int64。
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
        int64_t lo = t - w + 1;             /* lo-pop 必须先于 push(同 rolling_minmax_impl 的环形槽约定) */
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

/* chanpos / ddown 的外壳:每线程一份 scratch,逐列取极值后套各自公式。 */
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

/* deque 版对小窗**反而更慢**:窗内只有几个数时重扫是 cache-hot 的几次比较,而 deque 要维护
 * 环形缓冲 + 三个 T 长临时数组;实测交叉点在 d≈230。⇒ 按窗口分派(同 fm_ts_rank 的
 * `w<32 走 naive`)。两支**互相逐位一致**(都只是挑出窗内极值),验证时窗口集须跨过阈值两侧。 */
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


/* ts_skew: 滚动 3 阶标准化中心矩 = E[((x-μ)/σ)^3]。
 * **窗内两遍直接中心化**(pass1 窗均值 → pass2 Σ(v-mean)^p):数值稳,消 e2−mu² 抵消,
 * 与 GPU moment_core 逐窗两遍同算法同顺序 → 逐位一致。增量滚动幂和在长列上累积漂移
 * (实测与 GPU 分叉可达 kurt~40%),故取 O(w)/输出的两遍法:慢于增量但正确。 */
void fm_ts_skew(const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo, const int64_t* rng_lo, const int64_t* rng_hi) {
    #pragma omp parallel for schedule(static)
    for (int64_t s = 0; s < S; s++) {
        int64_t vlo, vhi;
        if (rng_lo) { vlo = rng_lo[s]; vhi = rng_hi[s]; }   /* 调用方预算的区间 */
        else        { _col_range(x, T, ldx, s, &vlo, &vhi); }
        for (int64_t t = 0; t < T; t++) out[t * ldo + s] = NAN;   /* 先全 NaN(warmup/越界/含 NaN 窗)*/
        if (vlo > vhi) continue;
        for (int64_t t = vlo + w - 1; t <= vhi; t++) {
            double mean = 0.0; int bad = 0;
            for (int64_t k = t - w + 1; k <= t; k++) {           /* pass1:窗均值(含 NaN 窗 → 跳)*/
                double v = x[k * ldx + s];
                if (isnan(v)) { bad = 1; break; }
                mean += v;
            }
            if (bad) continue;
            mean /= (double)w;
            double m2 = 0.0, m3 = 0.0;
            for (int64_t k = t - w + 1; k <= t; k++) {           /* pass2:直接中心矩(无抵消)*/
                double d = x[k * ldx + s] - mean, d2 = d * d;
                m2 += d2; m3 += d2 * d;
            }
            m2 /= (double)w; m3 /= (double)w;
            if (m2 > 0.0) out[t * ldo + s] = m3 / pow(m2, 1.5);  /* 常数窗 m2==0 → 留 NaN */
        }
    }
}


/* ts_kurt: 滚动 4 阶标准化中心矩 = E[((x-μ)/σ)^4] − 3(excess kurtosis)。窗内两遍中心矩,见 fm_ts_skew。 */
void fm_ts_kurt(const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo, const int64_t* rng_lo, const int64_t* rng_hi) {
    #pragma omp parallel for schedule(static)
    for (int64_t s = 0; s < S; s++) {
        int64_t vlo, vhi;
        if (rng_lo) { vlo = rng_lo[s]; vhi = rng_hi[s]; }   /* 调用方预算的区间 */
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


/* ts_slope: 滚动窗口内 OLS 斜率 = cov(t_idx, x) / var(t_idx)。
 * t_idx ∈ {0, 1, ..., w-1};常量项预算:mean_t = (w-1)/2,var_t = (w^2-1)/12。 */
/* 增量实现 O(T·S)(而非 O(T·S·w)):slope = cov(k,x)/var_k,var_k=(w²−1)/12 常数;
 *   cov_tx = (S_kx − mean_k·S_x)/w,S_x=Σx_k、S_kx=Σ k·x_k(k=窗内局部位 0..w−1)。
 * 滑窗 O(1) 递推(整窗位置 −1 = 减一个 S_x):
 *   S_kx(t) = S_kx(t−1) − S_x(t−1) + xc[t−w] + (w−1)·xc[t]   (用更新前的 S_x)
 *   S_x (t) = S_x (t−1) + xc[t] − xc[t−w]
 * NaN→0 入和 + cnt_nan 门控满窗才输出(同 ts_std 半 ffwd 语义)。每 _SLOPE_RESYNC 步
 * O(w) 重算一次,界住增量累加漂移(全程 ~T/RESYNC 次,可忽略)。 */
#define _SLOPE_RESYNC 65536
void fm_ts_slope(const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo) {
    double mean_t = (double)(w - 1) / 2.0;
    double var_t  = (double)(w * w - 1) / 12.0;  /* 整数 w 算 var */
    if (var_t <= 0.0) {
        for (int64_t t = 0; t < T; t++)
            for (int64_t s = 0; s < S; s++) out[t * ldo + s] = NAN;
        return;
    }
    #pragma omp parallel for schedule(static)
    for (int64_t s = 0; s < S; s++) {
        /* slope 自己扫描本列区间(不收外部区间):增量 S_kx/S_x 值随累计起点 + resync
         * 相位漂移,外部区间起点 vs 实际起点会产生 ~1 ULP 差,破 bit-identity;扫到本列实际
         * 起点则与不裁剪版逐位一致。扫描开销 O(跳过的行数),可忽略。 */
        int64_t vlo, vhi; _col_range(x, T, ldx, s, &vlo, &vhi);
        if (vlo > vhi) { for (int64_t t = 0; t < T; t++) out[t * ldo + s] = NAN; continue; }
        for (int64_t t = 0; t < vlo; t++)      out[t * ldo + s] = NAN;
        for (int64_t t = vhi + 1; t < T; t++)  out[t * ldo + s] = NAN;
        if (vhi - vlo + 1 < w) {               /* 列短于窗口:满窗永不达 → 全 NaN */
            for (int64_t t = vlo; t <= vhi; t++) out[t * ldo + s] = NAN;
            continue;
        }
        /* 首满窗 t=vlo+w−1:O(w) 直接算 S_x / S_kx / cnt_nan */
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
        /* 主循环 t=vlo+w..vhi:O(1) 递推 + 周期 resync 界漂移 */
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
                S_kx = S_kx - S_x + xo + (double)(w - 1) * xn;  /* 用更新前 S_x */
                S_x  = S_x + xn - xo;
                if (nn) cnt_nan++;
                if (no) cnt_nan--;
            }
            out[t * ldo + s] = (cnt_nan == 0)
                ? (S_kx - mean_t * S_x) / (double)w / var_t : NAN;
        }
    }
}


/* 单调 deque,O(N) 摊销。is_max=1 → 维持递减 deque,front 是 max。
 * 窗口内最多 w 个索引 → deque 用大小 w 的环形 buffer(原代码 alloc T*8B 浪费,
 * T=245k S=16 12 thread → 23MB → 改 w*8B 仅 2.3KB/thread)。
 *
 * 关键顺序:lo-pop(head 推进)必须在 push 之前。否则 push 时 tail%w 可能撞
 *   head%w(当 tail-head==w 时),覆盖 head 仍引用的有效条目。
 *   lo-pop 先做 → push 时 tail-head ≤ w-1,push slot 永不与 head 冲突。 */
static void rolling_minmax_impl(const double* x, double* out, int64_t T, int64_t S, int64_t w,
                                int is_max, int64_t ldx, int64_t ldo,
                                const int64_t* rng_lo, const int64_t* rng_hi) {
    #pragma omp parallel
    {
        std::vector<int64_t> deque(w);                          /* per-thread 环形 buffer(RAII) */
        #pragma omp for schedule(static)
        for (int64_t s = 0; s < S; s++) {
            int64_t vlo, vhi;
            if (rng_lo) { vlo = rng_lo[s]; vhi = rng_hi[s]; }   /* 调用方预算的区间 */
            else        { _col_range(x, T, ldx, s, &vlo, &vhi); }
            if (vlo > vhi) { for (int64_t t = 0; t < T; t++) out[t * ldo + s] = NAN; continue; }
            for (int64_t t = 0; t < vlo; t++)      out[t * ldo + s] = NAN;
            for (int64_t t = vhi + 1; t < T; t++)  out[t * ldo + s] = NAN;
            int64_t head = 0, tail = 0;
            int64_t cnt_nan = 0;
            for (int64_t t = vlo; t <= vhi; t++) {
                /* 1. lo-pop:先丢出 [t-w+1, t-1] 之外的旧条目(deque 索引均 ≥ vlo) */
                int64_t lo = t - w + 1;
                while (tail > head && deque[head % w] < lo) head++;
                /* 2. 当前值入 deque(NaN 跳过,但仍计 cnt_nan) */
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
                /* 3. NaN 滑出窗口的旧值减计数 */
                if (t - vlo >= w) {
                    double old = x[(t - w) * ldx + s];
                    if (isnan(old)) cnt_nan--;
                }
                /* 4. 输出:窗口内无 NaN 且 deque 非空 → 取 head value */
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


/* arg_max / arg_min:复用单调 deque,emit `t - deque[head]` = max/min 距今多少 bar(∈ [0, w-1])。
 * 当前 bar 即极值时返回 0。deque 同 minmax_impl,大小 w 环形,lo-pop 先于 push。 */
static void rolling_argminmax_impl(const double* x, double* out, int64_t T, int64_t S, int64_t w,
                                   int is_max, int64_t ldx, int64_t ldo) {
    #pragma omp parallel
    {
        std::vector<int64_t> deque(w);                          /* per-thread 环形 buffer(RAII) */
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


/* ----- ts_rank 桶化 helpers ----- */

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

/* 桶化一列。bucket_id[t] ∈ {1..B} 对 finite,0 对 NaN。 */
static void _bucketize_column(
    const double* x, int64_t T, int64_t ldx, int64_t s_idx, int B,
    int64_t* bucket_id, int64_t* n_finite_out)
{
    std::vector<_ts_rank_vi_t> arr(T);                          /* (value, orig_t)(RAII) */
    int64_t n = 0;
    for (int64_t t = 0; t < T; t++) {
        double v = x[t * ldx + s_idx];
        if (!isnan(v)) { arr[n].v = v; arr[n].orig = t; n++; }
        else { bucket_id[t] = 0; }
    }
    *n_finite_out = n;
    if (n == 0) return;
    /* 按值升序;tie 顺序未定 —— 桶号按排序位置赋,
     * 桶版本身 ±1/(2B) 近似,连续数据无 exact-tie → 结果确定。 */
    std::sort(arr.begin(), arr.begin() + n,
              [](const _ts_rank_vi_t& a, const _ts_rank_vi_t& b) { return a.v < b.v; });
    for (int64_t k = 0; k < n; k++) {
        int64_t b = (k * B) / n + 1;
        if (b > B) b = B;
        bucket_id[arr[k].orig] = b;
    }
}


/* ----- 序列引用 / 差分 / EMA / WMA ----- */

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


/* pandas ewm(span=k, adjust=False, min_periods=k) 行为:
 *   alpha = 2 / (span+1)
 *   见到第一个非 NaN 用作 seed,之后递推;cnt 累计非 NaN 个数,cnt < span 输出 NaN。
 *   NaN 输入位置维持 ema 不变,但输出 NaN。
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


/* WMA: weight[i] = i+1, 最旧 = 1, 最新 = w。Σw = w(w+1)/2。
 *
 * O(T·S) 滑窗实现(原 O(T·S·W),W=288 时约 ~250×):
 *   令 SUM[t]  = Σ_{i=0..w-1} x[t-w+1+i]
 *      WSUM[t] = Σ_{i=0..w-1} (i+1)·x[t-w+1+i]
 *   则 WSUM[t+1] − WSUM[t] = −SUM[t] + w·x[t+1]   (含 drop 项的旧 SUM)
 *      SUM[t+1] = SUM[t] − x[t-w+1] + x[t+1]
 *   两条更新都 O(1)。
 *
 * NaN 处理:NaN 当 0 累入 SUM/WSUM,但用 cnt_nan 单独计数;cnt_nan>0 → 输出 NaN。
 *          这样 SUM/WSUM 的递推恒等式仍成立(NaN 贡献两边都是 0)。
 */
void fm_ts_wma(const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo) {
    double wsum = (double)w * ((double)w + 1.0) / 2.0;
    #pragma omp parallel for schedule(static)
    for (int64_t s = 0; s < S; s++) {
        /* 前 w-1 行强制 NaN */
        for (int64_t t = 0; t < w - 1; t++) out[t * ldo + s] = NAN;

        /* t = w-1: bootstrap SUM / WSUM 一次 O(W) */
        double SUM = 0.0, WSUM = 0.0;
        int64_t cnt_nan = 0;
        for (int64_t i = 0; i < w; i++) {
            double v = x[i * ldx + s];
            if (isnan(v)) cnt_nan++;
            else { SUM += v; WSUM += (double)(i + 1) * v; }
        }
        out[(w - 1) * ldo + s] = (cnt_nan == 0) ? (WSUM / wsum) : NAN;

        /* t ≥ w: O(1)/step 滑动 */
        for (int64_t t = w; t < T; t++) {
            double drop = x[(t - w) * ldx + s];
            double new_v = x[t * ldx + s];
            int drop_nan = isnan(drop), new_nan = isnan(new_v);
            double drop_clean = drop_nan ? 0.0 : drop;
            double new_clean  = new_nan  ? 0.0 : new_v;

            /* WSUM 用 OLD SUM(含 drop) */
            WSUM = WSUM - SUM + (double)w * new_clean;
            SUM  = SUM - drop_clean + new_clean;
            if (drop_nan) cnt_nan--;
            if (new_nan)  cnt_nan++;

            out[t * ldo + s] = (cnt_nan == 0) ? (WSUM / wsum) : NAN;
        }
    }
}


/* ts_rank: 当前值在窗口内的归一化平均-rank ∈ [0,1]。
 *
 * O(T·S·log T) 实现 — per-column 一次坐标压缩 + Fenwick BIT 滑窗。
 * (原 O(T·S·W),W=288 时 ~10-15×)
 *
 * Per column:
 *   1. 收集 finite (value, orig_t) → qsort by value,得 unique 值排名 comp[t] ∈ {1..M};NaN → 0
 *   2. BIT[1..M]:add(c,+1) / add(c,-1) / prefix(c) 都是 O(log M)
 *   3. 窗口 [t-w+1..t] 内:
 *        cnt_less = prefix(c-1)
 *        cnt_eq   = prefix(c) - cnt_less       (含自己)
 *        rank     = (cnt_less + 0.5·(cnt_eq-1)) / (w-1)
 *   4. NaN 用旁路 cnt_nan 计数;cnt_nan>0 → 输出 NaN
 *
 * tie 处理:相同 value 共享同一 comp 索引,prefix 差自然给出窗口内同值数。
 */

/* 小 W 走 O(T·S·W) 朴素扫描 — 内层 W 次,缓存极友好,常数小。 */
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

/* 大 W 走 bucketized BIT 版 — B=128 quantile 桶,BIT 深度 log B = 7。
 * 同桶视为同值,rank 误差 ±1/(2B) ≈ 0.4%,rank IC 几乎不变。 */
static void _ts_rank_bucket(const double* x, double* out, int64_t T, int64_t S, int64_t w,
                            int64_t ldx, int64_t ldo) {
    const int B = _RANK_BUCKETS;
    #pragma omp parallel for schedule(static)
    for (int64_t s = 0; s < S; s++) {
        std::vector<int64_t> bucket_buf(T, 0);                  /* calloc 语义(零初始化,RAII) */
        int64_t* bucket_id = bucket_buf.data();
        int64_t n_finite;
        _bucketize_column(x, T, ldx, s, B, bucket_id, &n_finite);
        if (n_finite == 0) {
            for (int64_t t = 0; t < T; t++) out[t * ldo + s] = NAN;
            continue;
        }
        std::vector<int64_t> bit_buf(B + 2, 0);                 /* calloc 语义 */
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

/* 入口:W<32 走 naive(cache 友好,常数小);W≥32 走 bucketized BIT。 */
void fm_ts_rank(const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo) {
    if (w < 32) _ts_rank_naive (x, out, T, S, w, ldx, ldo);
    else        _ts_rank_bucket(x, out, T, S, w, ldx, ldo);
}


/* ============================================================================
 * Rolling pair — fm_ts_corr / fm_ts_cov
 * 行流式:线程持有连续 symbol 条带 [s0,s1),逐行推进,滑窗累加器存成数组(切片连续、L1 常驻)。
 * 行内 s 连续访存(stride 1)→ cache line 满用;逐列扫整列的写法每条 line 只用 1/8,
 * DRAM 流量 ~8×。每个 symbol 的累加 t-顺序与逐列写法相同 → 逐位一致。
 * 任一侧 NaN → cnt[s]++,窗口含 NaN 输出 NaN(同 ts_std)。
 *
 * 范围:有 rng → 直接用调用方给的区间;无 rng → 扫 a,b 两列有效区取交集。
 * t<vlo 或 t>vhi 行流式内逐行写 NaN。
 * ============================================================================ */

void fm_ts_corr(const double* a, const double* b, double* out, int64_t T, int64_t S, int64_t w,
                int64_t lda_, int64_t ldb, int64_t ldo, const int64_t* rng_lo, const int64_t* rng_hi) {
    std::vector<int64_t> vbuf;                                  /* rng 缺省时的 [lo|hi](RAII) */
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
    std::vector<double>  acc_buf((size_t)S * 5);                /* 5 滑窗累加器(RAII) */
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
    std::vector<int64_t> vbuf;                                  /* rng 缺省时的 [lo|hi](RAII) */
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
    std::vector<double>  acc_buf((size_t)S * 3);                /* 3 滑窗累加器(RAII) */
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
