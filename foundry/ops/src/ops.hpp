/* foundry/ops/src/ops.hpp —— 公式算子 CPU 参考后端的 C API(实现见 ops_xs / ops_ts / ops_inject_ts.cpp)。
 *
 * Layout:
 *   (T,S) 面板为 double 行优先:data[t * ld + s] = (时间 t, 资产 s)。ld = 行步长(元素数,ld ≥ S),
 *   ld == S 即 C 连续;ld > S 是宽面板的列块视图(零拷贝切列)。
 *   elementwise / ts / pair 的每个输入与 out 各带独立 ld(追加在参数尾部);
 *   cs 只收 C 连续全宽面板(截面语义需要整行)。
 *   空缺值用 NaN(IEEE 754),逐元素 NaN 传播。
 *
 * Rolling 语义:
 *   pandas `rolling(w).f()` with `min_periods=w`。窗口内有任何 NaN → 输出 NaN。
 *   t < w-1 时(暖机期)输出 NaN。
 *
 * NaN 传播:
 *   binary 算子(add/mul/...)任一侧 NaN → NaN。
 *   div 中 b==0 → NaN(不产生 inf:下游统一以 NaN 表示"无定义")。
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
void fm_tanh  (const double* x, double* out, int64_t T, int64_t S, int64_t ldx, int64_t ldo);  /* tanh(x) ∈ (-1, 1) */
void fm_inv   (const double* x, double* out, int64_t T, int64_t S, int64_t ldx, int64_t ldo);  /* 1/x; x==0 → NaN */
void fm_s_log_1p(const double* x, double* out, int64_t T, int64_t S, int64_t ldx, int64_t ldo);  /* sign(x)·log(1+|x|) */

/* ===== Elementwise binary ===== */
void fm_add  (const double* a, const double* b, double* out, int64_t T, int64_t S, int64_t lda, int64_t ldb, int64_t ldo);
void fm_sub  (const double* a, const double* b, double* out, int64_t T, int64_t S, int64_t lda, int64_t ldb, int64_t ldo);
void fm_mul  (const double* a, const double* b, double* out, int64_t T, int64_t S, int64_t lda, int64_t ldb, int64_t ldo);
void fm_div  (const double* a, const double* b, double* out, int64_t T, int64_t S, int64_t lda, int64_t ldb, int64_t ldo);
void fm_max_b(const double* a, const double* b, double* out, int64_t T, int64_t S, int64_t lda, int64_t ldb, int64_t ldo);  /* max(a,b) */
void fm_min_b(const double* a, const double* b, double* out, int64_t T, int64_t S, int64_t lda, int64_t ldb, int64_t ldo);  /* min(a,b) */
void fm_gt   (const double* a, const double* b, double* out, int64_t T, int64_t S, int64_t lda, int64_t ldb, int64_t ldo);  /* (a>b)?1:0,NaN 传播 */
void fm_lt   (const double* a, const double* b, double* out, int64_t T, int64_t S, int64_t lda, int64_t ldb, int64_t ldo);  /* (a<b)?1:0,NaN 传播 */

/* ===== Ternary bool 组合子(cond/entry, a/alpha, b/exit) ===== */
void fm_if_else   (const double* c, const double* a, const double* b, double* out, int64_t T, int64_t S, int64_t ldc, int64_t lda, int64_t ldb, int64_t ldo);
void fm_trade_when(const double* e, const double* a, const double* x, double* out, int64_t T, int64_t S, int64_t lde, int64_t lda, int64_t ldx, int64_t ldo);

/* ===== Logical 族 =====
 * NaN 传播同 gt/lt;fm_is_nan 例外(它就是用来测 NaN 的)。 */
void fm_not   (const double* x, double* out, int64_t T, int64_t S, int64_t ldx, int64_t ldo);  /* x>0?0:1,NaN→NaN */
void fm_is_nan(const double* x, double* out, int64_t T, int64_t S, int64_t ldx, int64_t ldo);  /* NaN?1:0,**不传播** */
void fm_and          (const double* a, const double* b, double* out, int64_t T, int64_t S, int64_t lda, int64_t ldb, int64_t ldo);
void fm_or           (const double* a, const double* b, double* out, int64_t T, int64_t S, int64_t lda, int64_t ldb, int64_t ldo);
void fm_equal        (const double* a, const double* b, double* out, int64_t T, int64_t S, int64_t lda, int64_t ldb, int64_t ldo);
void fm_not_equal    (const double* a, const double* b, double* out, int64_t T, int64_t S, int64_t lda, int64_t ldb, int64_t ldo);
void fm_greater_equal(const double* a, const double* b, double* out, int64_t T, int64_t S, int64_t lda, int64_t ldb, int64_t ldo);
void fm_less_equal   (const double* a, const double* b, double* out, int64_t T, int64_t S, int64_t lda, int64_t ldb, int64_t ldo);

/* ===== Elementwise binary_const(panel ⊕ scalar k,1 SIMD pass) ===== */
void fm_add_const(const double* x, double k, double* out, int64_t T, int64_t S, int64_t ldx, int64_t ldo);
void fm_mul_const(const double* x, double k, double* out, int64_t T, int64_t S, int64_t ldx, int64_t ldo);
/* k 连续取值(不受搜索常数池约束)。 */
void fm_left_tail (const double* x, double k, double* out, int64_t T, int64_t S, int64_t ldx, int64_t ldo);  /* >k → NaN */
void fm_right_tail(const double* x, double k, double* out, int64_t T, int64_t S, int64_t ldx, int64_t ldo);  /* <k → NaN */
/* hump:|x[t]−out[t−1]|>k 才更新,否则持有前值 ⇒ 压换手。逐列时序状态机(同 trade_when)。 */
void fm_hump      (const double* x, double k, double* out, int64_t T, int64_t S, int64_t ldx, int64_t ldo);
/* signed_power(x, k) = sign(x)·|x|^k;0→0(避免 0^k=NaN/inf @ k<0)。NaN 透传。 */
void fm_pow_const(const double* x, double k, double* out, int64_t T, int64_t S, int64_t ldx, int64_t ldo);

/* ===== Rolling unary, per-symbol time series =====
 * std/max/min/skew/kurt/mad/chanpos/ddown/zscore/snr/corr/cov 末尾收 rng_lo/rng_hi:每列预算的有效行区间
 * [vlo,vhi](int64×S),非 NULL 则跳过列内扫描直接用,NULL 则就地扫描。区间外输出必为 NaN,裁剪与否逐位一致。
 * 其余 ts 算子不裁剪。 */
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
void fm_ts_arg_max(const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo);  /* bars-since-max ∈ [0,w-1] */
void fm_ts_arg_min(const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo);
void fm_ts_skew   (const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo, const int64_t* rng_lo, const int64_t* rng_hi);  /* 3rd standardized moment */
void fm_ts_kurt   (const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo, const int64_t* rng_lo, const int64_t* rng_hi);  /* excess kurtosis */
void fm_ts_mad    (const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo, const int64_t* rng_lo, const int64_t* rng_hi);  /* mean abs deviation */
void fm_ts_slope  (const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo);  /* OLS slope vs t;增量值随累计起点漂移,故不收外部区间 */
/* 逐列滚动 min/max(双单调 deque,O(1) 摊销;实现在 ops_ts.cpp)。
 * chanpos / ddown / ts_scale 共用"取窗内极值"这一步。
 * 调用方提供 scratch:dq_lo/dq_hi 各 w 个 int64,mn/mx/ok 各 T 个。
 * ok[t]=1 ⇔ 窗满且窗内无 NaN;ok[t]=0 时 mn/mx 未定义。 */
void fm_win_minmax_col(const double* x, int64_t T, int64_t ldx, int64_t s, int64_t w,
                       const int64_t* rng_lo, const int64_t* rng_hi,
                       int64_t* dq_lo, int64_t* dq_hi,
                       double* mn, double* mx, unsigned char* ok);

/* 趋势融合算子(单 pass 内完成多步组合)。 */
void fm_ts_chanpos(const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo, const int64_t* rng_lo, const int64_t* rng_hi);  /* Donchian 通道位置 (x-min)/(max-min) ∈[0,1] */
void fm_ts_ddown  (const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo, const int64_t* rng_lo, const int64_t* rng_hi);  /* 距滚动高点回撤 x/max-1 ≤0(nearness-to-high)*/
void fm_ts_zscore (const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo, const int64_t* rng_lo, const int64_t* rng_hi);  /* 滚动 z=(x-μ)/σ,ddof=0 */
void fm_ts_snr    (const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo, const int64_t* rng_lo, const int64_t* rng_hi);  /* 趋势信噪比 μ/σ(rolling t-stat 型)*/

/* ===== BRAIN time_series 族(实现在 ops_inject_ts.cpp)=====
 * 前三个签名与既有 ts 算子一致(kind='ts');其后的多带一个连续标量参数。 */
void fm_ts_backfill(const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo);  /* 回看 d 行取最近有效值(**不套 min_periods**) */
void fm_ts_median  (const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo);  /* 滚动中位数 */
void fm_ts_av_diff (const double* x, double* out, int64_t T, int64_t S, int64_t w, int64_t ldx, int64_t ldo);  /* x − mean(x,d),**均值忽略 NaN**(BRAIN 定义) */
/* panel + window + 标量(kind='ts_param') */
void fm_ts_decay_exp_window(const double* x, double* out, int64_t T, int64_t S, int64_t w, double f, int64_t ldx, int64_t ldo);  /* 权重 f^i 归一化 */
void fm_ts_scale           (const double* x, double* out, int64_t T, int64_t S, int64_t w, double k, int64_t ldx, int64_t ldo);  /* chanpos + k */
void fm_ts_quantile        (const double* x, double* out, int64_t T, int64_t S, int64_t w, double driver, int64_t ldx, int64_t ldo);  /* 0=gaussian 1=uniform 2=cauchy */
/* panel + panel + window + 标量(kind='pair_param') */
void fm_ts_regression(const double* y, const double* x, double* out, int64_t T, int64_t S, int64_t w, double rettype, int64_t ldy, int64_t ldx, int64_t ldo);  /* 0=残差 1=截距 2=斜率 3=拟合 4=R² */

/* ===== Rolling pair ===== */
void fm_ts_corr(const double* a, const double* b, double* out, int64_t T, int64_t S, int64_t w, int64_t lda_, int64_t ldb, int64_t ldo, const int64_t* rng_lo, const int64_t* rng_hi);
void fm_ts_cov (const double* a, const double* b, double* out, int64_t T, int64_t S, int64_t w, int64_t lda_, int64_t ldb, int64_t ldo, const int64_t* rng_lo, const int64_t* rng_hi);

/* ===== Cross-sectional ===== */
/* 每个 t 在 S symbols 上做(平均-rank)排名,归一化到 [0,1]。NaN 位置输出 NaN。 */
void fm_cs_rank   (const double* x, double* out, int64_t T, int64_t S);
/* 每个 t 标准化:(x - mean) / std(ddof=1),跨 symbol。NaN 位置输出 NaN。 */
void fm_cs_zscore (const double* x, double* out, int64_t T, int64_t S);
/* 每个 t 中心化:x - row_mean,跨 symbol。NaN 位置输出 NaN。 */
void fm_cs_demean (const double* x, double* out, int64_t T, int64_t S);
/* 每个 t L1 归一化:x / Σ_s|x|。NaN 位置输出 NaN。 */
void fm_cs_scale  (const double* x, double* out, int64_t T, int64_t S);

#ifdef __cplusplus
}
#endif

#endif
