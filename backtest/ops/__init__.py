"""C 算子绑定 — 直接 re-export Python C 扩展模块 `_ops`(.pyd)。

源码 layout:
    ops_elementwise.c — unary + binary + binary_const
    ops_ts.c          — rolling unary(SIMD/deque/BIT)+ rolling pair
    ops_cs.c          — cross-sectional(rank/zscore/demean/scale/finite_validstd)
    ops_metrics.c     — IC / per-t IC / per-t PnL / rank IC / turnover
    _ops_module.c     — Python C-API 绑定
编译产物:
    backtest/ops/_ops.cp313-win_amd64.pyd

Python 端命名约定:
    保留字 abs/log/div 在 C 扩展里写作 abs_/log_/div_;此处 re-export 时改回普通名字。
"""

from backtest.ops import _ops as _c

# Elementwise unary
abs_   = _c.abs_
neg    = _c.neg
sign   = _c.sign
log    = _c.log_
sqrt_  = _c.sqrt_
square = _c.square
tanh_  = _c.tanh_
inv    = _c.inv
s_log_1p = _c.s_log_1p

# Elementwise binary(GT/LT 2026-07-24 恢复:自由书口径下 = regime 条件门;
# 旧删除理由是 cs_zscore+tanh 截面分类器过拟合路径,该栈已随截面口径退役)
add   = _c.add
sub   = _c.sub
mul   = _c.mul
div   = _c.div_
max_b = _c.max_b
min_b = _c.min_b
gt    = _c.gt
lt    = _c.lt

# Ternary bool 组合子(2026-07-24:事件驱动 / 精细调控)
if_else    = _c.if_else       # if_else(cond,a,b):cond>0?a:b(cond NaN→NaN)
trade_when = _c.trade_when    # trade_when(entry,alpha,exit):exit>0→0;entry>0→alpha;否则持有前值

# 注入层 logical 族(2026-08-13:LLM 写 alpha 需完整条件逻辑)。greater/less = 上面的 gt/lt。
# 命名尾下划线避开 Python 保留字 and/or/not。
not_   = _c.not_              # x>0?0:1;NaN→NaN
is_nan = _c.is_nan            # NaN?1:0 —— **唯一不传播 NaN 的算子**
and_          = _c.and_
or_           = _c.or_
equal         = _c.equal
not_equal     = _c.not_equal
greater_equal = _c.greater_equal
less_equal    = _c.less_equal

# f32 → f64 块升格(块化求值器 operand 叶,OMP 并行)
upcast32 = _c.upcast32

# EvalCache dense-pack(无损稀疏存储:非 NaN 位图 + 紧凑值)
pack_sparse   = _c.pack_sparse
unpack_sparse = _c.unpack_sparse

# Elementwise binary_const(panel ⊕ scalar)
add_const = _c.add_const
mul_const = _c.mul_const
pow_const = _c.pow_const  # signed_power: sign(x)·|x|^k, 0→0

# 注入层 panel⊕**连续**标量(2026-08-13,BRAIN transformational + time_series)
left_tail  = _c.left_tail   # >k 全置 NaN(保左尾)
right_tail = _c.right_tail  # <k 全置 NaN(保右尾)
hump       = _c.hump        # |x[t]−out[t−1]|>k 才更新,否则持有前值 ⇒ 压换手(逐列时序状态机)

# 注入层 rolling(2026-08-13,BRAIN time_series 族;实现 ops_inject_ts.cpp)
ts_backfill = _c.ts_backfill   # 回看 d 行取最近有效值(**不套 min_periods**,它就是用来填 NaN 的)
ts_median   = _c.ts_median     # 滚动中位数
ts_av_diff  = _c.ts_av_diff    # x − mean(x,d),**均值忽略 NaN**(BRAIN 定义,与 ts_mean 相反)
ts_decay_exp_window = _c.ts_decay_exp_window   # (x, d, factor) 权重 factor^i 归一化
ts_scale            = _c.ts_scale              # (x, d, k) = chanpos + k
ts_quantile         = _c.ts_quantile           # (x, d, driver) 0=gaussian 1=uniform 2=cauchy
ts_regression       = _c.ts_regression         # (y, x, d, rettype) 0=残差 1=截距 2=斜率 3=拟合 4=R²

# Rolling unary
ts_mean  = _c.ts_mean
ts_std   = _c.ts_std
ts_sum   = _c.ts_sum
ts_max   = _c.ts_max
ts_min   = _c.ts_min
ts_ref   = _c.ts_ref
ts_delta = _c.ts_delta
ts_ema   = _c.ts_ema
ts_wma   = _c.ts_wma
ts_rank    = _c.ts_rank
ts_arg_max = _c.ts_arg_max
ts_arg_min = _c.ts_arg_min
# 新加:WQ101/AlphaGen/AlphaForge 高频
ts_skew  = _c.ts_skew
ts_kurt  = _c.ts_kurt
ts_mad   = _c.ts_mad
ts_slope = _c.ts_slope
# 时序书口径新增融合算子(2026-07-22:通道位置/距高点回撤/滚动z/趋势信噪比)
ts_chanpos = _c.ts_chanpos
ts_ddown   = _c.ts_ddown
ts_zscore  = _c.ts_zscore
ts_snr     = _c.ts_snr

# Rolling pair
ts_corr = _c.ts_corr
ts_cov  = _c.ts_cov

# Cross-sectional
cs_rank   = _c.cs_rank
cs_zscore = _c.cs_zscore
cs_zscore_np = _c.cs_zscore_np   # NaN→0 fill + ddof=0(旧 metrics.cs_zscore_np 的 C 化)
cs_demean = _c.cs_demean
cs_scale  = _c.cs_scale
# evaluator gate1 单 pass:返回 (finite_ratio, valid_ratio_cs)
cs_finite_validstd = _c.cs_finite_validstd
# evaluator 覆盖率门:mean per-bar holdable coverage(挡稀疏退化信号伪高 IC)
cs_holdable_coverage = _c.cs_holdable_coverage

# Metrics
ic       = _c.ic
rank_ic  = _c.rank_ic
icir     = _c.icir         # (icir, mean_ic, std_ic),ddof=0(旧 metrics.icir 的 C 化)
turnover = _c.turnover
per_t_ic  = _c.per_t_ic
per_t_pnl = _c.per_t_pnl   # L1-norm signed PnL per t,AlphaPool D5 行为级 diversity
pnl_corr_vec = _c.pnl_corr_vec   # 候选 vs 池成员 pnl Pearson 向量(C 化 _corr_with_pool,边际 Δens 用)

omp_max_threads = _c.omp_max_threads
omp_set_num_threads = _c.omp_set_num_threads   # 局级 worker fork 后升线程
malloc_trim = _c.malloc_trim                   # glibc brk 堆空洞归还(fork 前/worker 局末)


__all__ = [
    'abs_', 'neg', 'sign', 'log', 'sqrt_', 'square', 'tanh_', 'inv', 's_log_1p',
    'upcast32', 'pack_sparse', 'unpack_sparse',
    'add', 'sub', 'mul', 'div', 'max_b', 'min_b', 'gt', 'lt', 'if_else', 'trade_when',
    'not_', 'is_nan', 'and_', 'or_', 'equal', 'not_equal', 'greater_equal', 'less_equal',
    'add_const', 'mul_const', 'pow_const', 'left_tail', 'right_tail', 'hump',
    'ts_backfill', 'ts_median', 'ts_av_diff',
    'ts_decay_exp_window', 'ts_scale', 'ts_quantile', 'ts_regression',
    'ts_mean', 'ts_std', 'ts_sum', 'ts_max', 'ts_min',
    'ts_ref', 'ts_delta', 'ts_ema', 'ts_wma',
    'ts_rank', 'ts_arg_max', 'ts_arg_min',
    'ts_skew', 'ts_kurt', 'ts_mad', 'ts_slope',
    'ts_chanpos', 'ts_ddown', 'ts_zscore', 'ts_snr',
    'ts_corr', 'ts_cov',
    'cs_rank', 'cs_zscore', 'cs_zscore_np', 'cs_demean', 'cs_scale', 'cs_finite_validstd', 'cs_holdable_coverage',
    'ic', 'rank_ic', 'icir', 'turnover', 'per_t_ic', 'per_t_pnl', 'pnl_corr_vec',
    'omp_max_threads', 'omp_set_num_threads', 'malloc_trim',
]
