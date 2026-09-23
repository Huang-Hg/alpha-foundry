/* foundry/ops/src/ops_xs.cpp — 逐元素算子(无窗口)与截面算子
 *
 * unary:fm_abs / fm_neg / fm_sign / fm_log / fm_sqrt_ / fm_square / fm_tanh / fm_inv / fm_s_log_1p / fm_not / fm_is_nan
 * binary:fm_add / fm_sub / fm_mul / fm_div / fm_max_b / fm_min_b / fm_gt / fm_lt / logical 族
 * binary_const:fm_add_const / fm_mul_const / fm_pow_const(signed_power)/ fm_left_tail / fm_right_tail / fm_hump
 * ternary:fm_if_else / fm_trade_when
 * cross-sectional:fm_cs_rank / fm_cs_zscore / fm_cs_demean / fm_cs_scale
 *
 * 设计原则:
 *   - 算术 ops (add/sub/mul,unary 的 abs/neg/square/tanh):IEEE 754 NaN 隐式传播,
 *     无显式 isnan 分支 → -O3 -march=native 自动 SIMD vectorize。
 *   - 比较/选择 ops (max/min/log/sqrt/inv/sign):NaN 比较恒 false 会"吞 NaN"成
 *     0/选错值,必须显式 isnan 检查。
 *   - div/pow_const:特殊值(b==0,base==0)显式分支,其余靠 IEEE 自然传播。
 *   - build.sh 须保留 -fno-finite-math-only;否则 GCC ≥ 14 会优化掉 isnan/isfinite。
 *   - 2D strided 寻址(ldx/ldo 行步长):逐元素表达式与 ld 无关,ld == S 即 C 连续。
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
/* s_log_1p: sign(x) · log(1 + |x|),压缩重尾保留符号,定义域全实数。 */
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
/* div:vb == 0.0 须显式检查(IEEE 754 div-by-0 → ±Inf,而我们要 NaN);NaN 由 va/vb 自然传播。 */
FM_BIN_ARITH(fm_div, (vb == 0.0) ? NAN : (va / vb))

FM_BIN_NAN(fm_max_b, (va > vb) ? va : vb)
FM_BIN_NAN(fm_min_b, (va < vb) ? va : vb)
/* 条件门:输出 0/1 指示,NaN 传播。 */
FM_BIN_NAN(fm_gt,    (va > vb) ? 1.0 : 0.0)
FM_BIN_NAN(fm_lt,    (va < vb) ? 1.0 : 0.0)


/* ---------- Ternary bool 组合子(事件驱动的进出场) ---------- */

/* if_else:cond NaN → NaN;cond>0 → a,否则 b(a/b 的 NaN 自然传播)。 */
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

/* trade_when(WorldQuant 语义,逐列时序状态机,因果:仅依赖 ≤t):
 *   exit>0 → 0;否则 entry>0 → alpha[t];否则持有前值(t=0 前值=0)。
 *   entry/exit NaN 比较恒 false → 持有;alpha NaN 在进场时自然传播(exit 可清除)。 */
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


/* ---------- BRAIN logical 族(完整条件逻辑) ----------
 * NaN 约定与 gt/lt 一致(任一侧 NaN → NaN);唯一例外 fm_is_nan —— 它就是用来测 NaN 的,
 * 传播 NaN 会让它永远返回不了 1。 */

FM_UNARY(fm_not,    isnan(v) ? NAN : ((v > 0.0) ? 0.0 : 1.0))
FM_UNARY(fm_is_nan, isnan(v) ? 1.0 : 0.0)

FM_BIN_NAN(fm_and,           (va > 0.0 && vb > 0.0) ? 1.0 : 0.0)
FM_BIN_NAN(fm_or,            (va > 0.0 || vb > 0.0) ? 1.0 : 0.0)
FM_BIN_NAN(fm_equal,         (va == vb) ? 1.0 : 0.0)
FM_BIN_NAN(fm_not_equal,     (va != vb) ? 1.0 : 0.0)
FM_BIN_NAN(fm_greater_equal, (va >= vb) ? 1.0 : 0.0)
FM_BIN_NAN(fm_less_equal,    (va <= vb) ? 1.0 : 0.0)


/* ----- binary_const(panel ⊕ scalar k,1 SIMD pass) ----- */

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

/* panel⊕连续标量(BRAIN transformational 族):尾部截断。NaN 自然保持 NaN
 * (isnan 比较恒 false ⇒ 走 else 分支返回 v,即 NaN)。 */
FM_BIN_CONST(fm_left_tail,  (v > k) ? NAN : v)   /* > maximum 全置 NaN,保左尾 */
FM_BIN_CONST(fm_right_tail, (v < k) ? NAN : v)   /* < minimum 全置 NaN,保右尾 */

/* hump(x, k)(BRAIN time_series 族):|x[t] − out[t−1]| > k 才更新,否则持有前值 ⇒ 压换手。
 * 逐列时序状态机,因果(仅依赖 ≤t),列间独立 ⇒ 按列 OMP(同 fm_trade_when 的并行轴)。
 * 前值未定义时首个有限值直接进入;x[t]=NaN 不构成"变化",持有前值。
 * k<0 时每步都更新(退化为恒等),不做特判 —— 交由调用方语义。 */
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

/* signed_power 内循环模板:0→0(防 0^k=inf/NaN @ k<0);其他 sign(v)·P(av),av=|v|。
 * NaN 由 fabs/P 自然传播(fabs(NaN)=NaN);0 检查不可省。P 表达式按 k 特化。 */
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
    /* 常用 k ∈ {±2, ±0.5} 用 mul/sqrt 替 libm pow(~5-11×)。与 pow 版逐位等价:
     * av*av = pow(av,2) 精确;sqrt = pow(av,.5) IEEE 正确舍入;负指数同式取倒。
     * 其余 k 走通用 pow。 */
    if      (k ==  2.0) { FM_SIGNED_POW_LOOP(av * av) }
    else if (k == -2.0) { FM_SIGNED_POW_LOOP(1.0 / (av * av)) }
    else if (k ==  0.5) { FM_SIGNED_POW_LOOP(sqrt(av)) }
    else if (k == -0.5) { FM_SIGNED_POW_LOOP(1.0 / sqrt(av)) }
    else                { FM_SIGNED_POW_LOOP(pow(av, k)) }
}

/* ---------- Cross-sectional ----------
 * 每 t 一行截面运算,跨 t OMP 并行(每 t 独立)。
 * cs_rank:有效值 compaction 后 std::sort 按值升序 + 等值组平均-rank。
 * tie-invariant:组内顺序不影响输出(等值组赋同一 avg-rank)→ value-only 比较即逐位一致。
 */

/* (value, orig_s) 对;std::sort 按 value 升序(tie 顺序未定但输出 tie-invariant)。 */
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
    /* 同值组取平均 rank(组内顺序无关 → bit-identical) */
    int64_t i = 0;
    while (i < n_valid) {
        int64_t j = i;
        while (j + 1 < n_valid && buf[j+1].first == buf[i].first) j++;
        double avg_norm = ((double)(i + j) / 2.0) / (double)(n_valid - 1);
        for (int64_t k = i; k <= j; k++) out_row[buf[k].second] = avg_norm;
        i = j + 1;
    }
    /* NaN 位置回填 */
    for (int64_t s = 0; s < S; s++) {
        if (isnan(row[s])) out_row[s] = NAN;
    }
}

void fm_cs_rank(const double* x, double* out, int64_t T, int64_t S) {
    #pragma omp parallel
    {
        std::vector<_vi_pair> buf(S);            /* per-thread scratch(RAII) */
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


/* cs_scale: 横截面 L1 归一化 = x[t,s] / sum_s |x[t,s]|(dollar-neutral 信号的常用归一)。 */
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

