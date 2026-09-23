/* foundry/ops/src/module.cpp —— _ops 扩展:把 ops.hpp 的全部算子暴露为 _ops.<name>。
 *
 * 约定:
 *   - elementwise / ts / pair 接受 (T,S) float64 行步长视图(列连续即可,行步长任意正)
 *     → 零拷贝直通 C 内核(列块切片不必拷贝);其余形态(f32 / 列不连续)由
 *     PyArray_FROMANY 物化成 C 连续拷贝。cs 要求 C 连续全宽面板。
 *   - elementwise / ts / pair 末尾可选 out= 位置参数:可写 float64 (T,S) 列连续视图,
 *     内核直写(列块结果直接落进全宽数组,免回拷);返回 out 本身。
 *     缺省时返回新 C 连续数组。
 *   - 内部假设输入合规;只校验 numpy 不能直接 reinterpret 的硬约束(dtype、ndim、shape 一致性)。
 */

#include "npy_glue.h"
#include "ops.hpp"

/* 共享胶水(foundry/include/npy_glue.h)的本地短名:C 连续 / 零拷贝行步长视图 / out 视图 / 分配。 */
#define as_panel     bt_as_2d
#define as_panel_ld  bt_as_panel_ld
#define as_out_ld    bt_as_out_ld
#define alloc_panel  bt_alloc_panel


/* ---------- Elementwise unary 通用模板 ---------- */

#define DEF_UNARY(py_name, c_fn)                                                   \
static PyObject* py_##py_name(PyObject* self, PyObject* args) {                    \
    PyObject* o_x; PyObject* o_out = NULL;                                          \
    if (!PyArg_ParseTuple(args, "O|O", &o_x, &o_out)) return NULL;                  \
    int64_t ldx, ldo;                                                                \
    PyArrayObject* x = as_panel_ld(o_x, &ldx);                                       \
    if (!x) return NULL;                                                             \
    npy_intp T = PyArray_DIM(x, 0), S = PyArray_DIM(x, 1);                           \
    PyArrayObject* out = o_out ? as_out_ld(o_out, T, S, &ldo)                        \
                               : alloc_panel(T, S, &ldo);                            \
    if (!out) { Py_DECREF(x); return NULL; }                                         \
    c_fn((const double*)PyArray_DATA(x), (double*)PyArray_DATA(out),                 \
         (int64_t)T, (int64_t)S, ldx, ldo);                                          \
    Py_DECREF(x);                                                                    \
    return (PyObject*)out;                                                           \
}

DEF_UNARY(abs_,    fm_abs)
DEF_UNARY(neg,     fm_neg)
DEF_UNARY(sign,    fm_sign)
DEF_UNARY(log_,    fm_log)
DEF_UNARY(sqrt_,   fm_sqrt_)
DEF_UNARY(square,  fm_square)
DEF_UNARY(tanh_,   fm_tanh)
DEF_UNARY(inv,     fm_inv)
DEF_UNARY(s_log_1p,fm_s_log_1p)
DEF_UNARY(not_,    fm_not)
DEF_UNARY(is_nan,  fm_is_nan)


/* ---------- Elementwise binary ---------- */

#define DEF_BINARY(py_name, c_fn)                                                   \
static PyObject* py_##py_name(PyObject* self, PyObject* args) {                     \
    PyObject *o_a, *o_b; PyObject* o_out = NULL;                                     \
    if (!PyArg_ParseTuple(args, "OO|O", &o_a, &o_b, &o_out)) return NULL;            \
    int64_t lda, ldb, ldo;                                                            \
    PyArrayObject* a = as_panel_ld(o_a, &lda); if (!a) return NULL;                   \
    PyArrayObject* b = as_panel_ld(o_b, &ldb); if (!b) { Py_DECREF(a); return NULL; } \
    if (PyArray_DIM(a, 0) != PyArray_DIM(b, 0) || PyArray_DIM(a, 1) != PyArray_DIM(b, 1)) { \
        PyErr_SetString(PyExc_ValueError, "panel shape mismatch");                   \
        Py_DECREF(a); Py_DECREF(b); return NULL;                                     \
    }                                                                                 \
    npy_intp T = PyArray_DIM(a, 0), S = PyArray_DIM(a, 1);                            \
    PyArrayObject* out = o_out ? as_out_ld(o_out, T, S, &ldo)                         \
                               : alloc_panel(T, S, &ldo);                             \
    if (!out) { Py_DECREF(a); Py_DECREF(b); return NULL; }                            \
    c_fn((const double*)PyArray_DATA(a), (const double*)PyArray_DATA(b),              \
         (double*)PyArray_DATA(out), (int64_t)T, (int64_t)S, lda, ldb, ldo);          \
    Py_DECREF(a); Py_DECREF(b);                                                       \
    return (PyObject*)out;                                                            \
}

DEF_BINARY(add,   fm_add)
DEF_BINARY(sub,   fm_sub)
DEF_BINARY(mul,   fm_mul)
DEF_BINARY(div_,  fm_div)
DEF_BINARY(max_b, fm_max_b)
DEF_BINARY(min_b, fm_min_b)
DEF_BINARY(gt,    fm_gt)
DEF_BINARY(lt,    fm_lt)
/* logical 族:greater/less 即 gt/lt。 */
DEF_BINARY(and_,          fm_and)
DEF_BINARY(or_,           fm_or)
DEF_BINARY(equal,         fm_equal)
DEF_BINARY(not_equal,     fm_not_equal)
DEF_BINARY(greater_equal, fm_greater_equal)
DEF_BINARY(less_equal,    fm_less_equal)


/* ---------- Ternary(三 panel 入参,bool 组合子) ---------- */

#define DEF_TERNARY(py_name, c_fn)                                                    \
static PyObject* py_##py_name(PyObject* self, PyObject* args) {                       \
    PyObject *o_a, *o_b, *o_c; PyObject* o_out = NULL;                                 \
    if (!PyArg_ParseTuple(args, "OOO|O", &o_a, &o_b, &o_c, &o_out)) return NULL;       \
    int64_t lda, ldb, ldc, ldo;                                                         \
    PyArrayObject* a = as_panel_ld(o_a, &lda); if (!a) return NULL;                     \
    PyArrayObject* b = as_panel_ld(o_b, &ldb); if (!b) { Py_DECREF(a); return NULL; }   \
    PyArrayObject* c = as_panel_ld(o_c, &ldc);                                          \
    if (!c) { Py_DECREF(a); Py_DECREF(b); return NULL; }                                \
    if (PyArray_DIM(a, 0) != PyArray_DIM(b, 0) || PyArray_DIM(a, 1) != PyArray_DIM(b, 1) || \
        PyArray_DIM(a, 0) != PyArray_DIM(c, 0) || PyArray_DIM(a, 1) != PyArray_DIM(c, 1)) { \
        PyErr_SetString(PyExc_ValueError, "panel shape mismatch");                     \
        Py_DECREF(a); Py_DECREF(b); Py_DECREF(c); return NULL;                         \
    }                                                                                   \
    npy_intp T = PyArray_DIM(a, 0), S = PyArray_DIM(a, 1);                              \
    PyArrayObject* out = o_out ? as_out_ld(o_out, T, S, &ldo)                           \
                               : alloc_panel(T, S, &ldo);                               \
    if (!out) { Py_DECREF(a); Py_DECREF(b); Py_DECREF(c); return NULL; }                \
    c_fn((const double*)PyArray_DATA(a), (const double*)PyArray_DATA(b),                \
         (const double*)PyArray_DATA(c), (double*)PyArray_DATA(out),                    \
         (int64_t)T, (int64_t)S, lda, ldb, ldc, ldo);                                   \
    Py_DECREF(a); Py_DECREF(b); Py_DECREF(c);                                           \
    return (PyObject*)out;                                                              \
}

DEF_TERNARY(if_else,    fm_if_else)
DEF_TERNARY(trade_when, fm_trade_when)


/* ---------- Elementwise binary_const(panel ⊕ scalar k)---------- */

#define DEF_BIN_CONST(py_name, c_fn)                                                \
static PyObject* py_##py_name(PyObject* self, PyObject* args) {                     \
    PyObject* o_x; double k; PyObject* o_out = NULL;                                 \
    if (!PyArg_ParseTuple(args, "Od|O", &o_x, &k, &o_out)) return NULL;              \
    int64_t ldx, ldo;                                                                 \
    PyArrayObject* x = as_panel_ld(o_x, &ldx); if (!x) return NULL;                   \
    npy_intp T = PyArray_DIM(x, 0), S = PyArray_DIM(x, 1);                            \
    PyArrayObject* out = o_out ? as_out_ld(o_out, T, S, &ldo)                         \
                               : alloc_panel(T, S, &ldo);                             \
    if (!out) { Py_DECREF(x); return NULL; }                                          \
    c_fn((const double*)PyArray_DATA(x), k, (double*)PyArray_DATA(out),               \
         (int64_t)T, (int64_t)S, ldx, ldo);                                           \
    Py_DECREF(x);                                                                     \
    return (PyObject*)out;                                                            \
}

DEF_BIN_CONST(add_const, fm_add_const)
DEF_BIN_CONST(mul_const, fm_mul_const)
DEF_BIN_CONST(pow_const, fm_pow_const)
DEF_BIN_CONST(left_tail,  fm_left_tail)
DEF_BIN_CONST(right_tail, fm_right_tail)
DEF_BIN_CONST(hump,       fm_hump)


/* ---------- Rolling unary ---------- */

/* 所有 ts unary 绑定统一接受可选尾参 (out, vlo, vhi)(每列有效行区间,int64×S):
 * 非裁剪核(DEF_TS_UNARY)解析后忽略 vlo/vhi;裁剪核(DEF_TS_UNARY_CLIP)传给 c_fn
 * (None → 核内扫描)。统一签名让调用方对所有 ts op 一致传参。
 * vlo/vhi 须 int64 C 连续(调用方保证);直接借数据指针不管引用(调用期有效)。*/
#define _AS_I64_OR_NULL(o) \
    ((o) && (o) != Py_None ? (const int64_t*)PyArray_DATA((PyArrayObject*)(o)) : NULL)

#define DEF_TS_UNARY(py_name, c_fn)                                                 \
static PyObject* py_##py_name(PyObject* self, PyObject* args) {                     \
    PyObject* o_x; long long w; PyObject* o_out = NULL;                              \
    PyObject* o_lo = NULL; PyObject* o_hi = NULL;                                     \
    if (!PyArg_ParseTuple(args, "OL|OOO", &o_x, &w, &o_out, &o_lo, &o_hi)) return NULL; \
    (void)o_lo; (void)o_hi;                          /* 非裁剪核忽略范围 */          \
    int64_t ldx, ldo;                                                                 \
    PyArrayObject* x = as_panel_ld(o_x, &ldx);                                        \
    if (!x) return NULL;                                                              \
    npy_intp T = PyArray_DIM(x, 0), S = PyArray_DIM(x, 1);                            \
    PyArrayObject* out = o_out ? as_out_ld(o_out, T, S, &ldo)                         \
                               : alloc_panel(T, S, &ldo);                             \
    if (!out) { Py_DECREF(x); return NULL; }                                          \
    c_fn((const double*)PyArray_DATA(x), (double*)PyArray_DATA(out),                  \
         (int64_t)T, (int64_t)S, (int64_t)w, ldx, ldo);                               \
    Py_DECREF(x);                                                                     \
    return (PyObject*)out;                                                            \
}

#define DEF_TS_UNARY_CLIP(py_name, c_fn)                                            \
static PyObject* py_##py_name(PyObject* self, PyObject* args) {                     \
    PyObject* o_x; long long w; PyObject* o_out = NULL;                              \
    PyObject* o_lo = NULL; PyObject* o_hi = NULL;                                     \
    if (!PyArg_ParseTuple(args, "OL|OOO", &o_x, &w, &o_out, &o_lo, &o_hi)) return NULL; \
    int64_t ldx, ldo;                                                                 \
    PyArrayObject* x = as_panel_ld(o_x, &ldx);                                        \
    if (!x) return NULL;                                                              \
    npy_intp T = PyArray_DIM(x, 0), S = PyArray_DIM(x, 1);                            \
    PyArrayObject* out = o_out ? as_out_ld(o_out, T, S, &ldo)                         \
                               : alloc_panel(T, S, &ldo);                             \
    if (!out) { Py_DECREF(x); return NULL; }                                          \
    c_fn((const double*)PyArray_DATA(x), (double*)PyArray_DATA(out),                  \
         (int64_t)T, (int64_t)S, (int64_t)w, ldx, ldo,                                \
         _AS_I64_OR_NULL(o_lo), _AS_I64_OR_NULL(o_hi));                               \
    Py_DECREF(x);                                                                     \
    return (PyObject*)out;                                                            \
}

DEF_TS_UNARY(ts_mean,  fm_ts_mean)
DEF_TS_UNARY_CLIP(ts_std,   fm_ts_std)
DEF_TS_UNARY(ts_sum,   fm_ts_sum)
DEF_TS_UNARY_CLIP(ts_max,   fm_ts_max)
DEF_TS_UNARY_CLIP(ts_min,   fm_ts_min)
DEF_TS_UNARY(ts_ref,   fm_ts_ref)
DEF_TS_UNARY(ts_delta, fm_ts_delta)
DEF_TS_UNARY(ts_ema,   fm_ts_ema)
DEF_TS_UNARY(ts_wma,   fm_ts_wma)
DEF_TS_UNARY(ts_rank,    fm_ts_rank)
DEF_TS_UNARY(ts_arg_max, fm_ts_arg_max)
DEF_TS_UNARY(ts_arg_min, fm_ts_arg_min)
DEF_TS_UNARY_CLIP(ts_skew,    fm_ts_skew)
DEF_TS_UNARY_CLIP(ts_kurt,    fm_ts_kurt)
DEF_TS_UNARY_CLIP(ts_mad,     fm_ts_mad)
DEF_TS_UNARY(ts_slope,   fm_ts_slope)        /* 增量值随起点漂移 → 留扫描裁剪保 bit-identity */
DEF_TS_UNARY_CLIP(ts_chanpos, fm_ts_chanpos)
DEF_TS_UNARY_CLIP(ts_ddown,   fm_ts_ddown)
DEF_TS_UNARY_CLIP(ts_zscore,  fm_ts_zscore)
DEF_TS_UNARY_CLIP(ts_snr,     fm_ts_snr)


/* ---------- Rolling pair（corr/cov 均裁剪,统一收 vlo/vhi）---------- */

#define DEF_TS_PAIR(py_name, c_fn)                                                  \
static PyObject* py_##py_name(PyObject* self, PyObject* args) {                     \
    PyObject *o_a, *o_b; long long w; PyObject* o_out = NULL;                        \
    PyObject* o_lo = NULL; PyObject* o_hi = NULL;                                     \
    if (!PyArg_ParseTuple(args, "OOL|OOO", &o_a, &o_b, &w, &o_out, &o_lo, &o_hi)) return NULL; \
    int64_t lda, ldb, ldo;                                                            \
    PyArrayObject* a = as_panel_ld(o_a, &lda); if (!a) return NULL;                   \
    PyArrayObject* b = as_panel_ld(o_b, &ldb); if (!b) { Py_DECREF(a); return NULL; } \
    if (PyArray_DIM(a, 0) != PyArray_DIM(b, 0) || PyArray_DIM(a, 1) != PyArray_DIM(b, 1)) { \
        PyErr_SetString(PyExc_ValueError, "panel shape mismatch");                   \
        Py_DECREF(a); Py_DECREF(b); return NULL;                                     \
    }                                                                                 \
    npy_intp T = PyArray_DIM(a, 0), S = PyArray_DIM(a, 1);                            \
    PyArrayObject* out = o_out ? as_out_ld(o_out, T, S, &ldo)                         \
                               : alloc_panel(T, S, &ldo);                             \
    if (!out) { Py_DECREF(a); Py_DECREF(b); return NULL; }                            \
    c_fn((const double*)PyArray_DATA(a), (const double*)PyArray_DATA(b),              \
         (double*)PyArray_DATA(out), (int64_t)T, (int64_t)S, (int64_t)w,              \
         lda, ldb, ldo, _AS_I64_OR_NULL(o_lo), _AS_I64_OR_NULL(o_hi));               \
    Py_DECREF(a); Py_DECREF(b);                                                       \
    return (PyObject*)out;                                                            \
}

DEF_TS_PAIR(ts_corr, fm_ts_corr)
DEF_TS_PAIR(ts_cov,  fm_ts_cov)


/* ---------- BRAIN time_series 族 ----------
 * 前三个复用 DEF_TS_UNARY(签名与既有 ts 算子一致);其后的多带一个连续标量 k。 */

DEF_TS_UNARY(ts_backfill, fm_ts_backfill)
DEF_TS_UNARY(ts_median,   fm_ts_median)
DEF_TS_UNARY(ts_av_diff,  fm_ts_av_diff)

/* panel + window + 标量:f(x, w, k, out=None) */
#define DEF_TS_PARAM(py_name, c_fn)                                                 \
static PyObject* py_##py_name(PyObject* self, PyObject* args) {                     \
    PyObject* o_x; long long w; double k; PyObject* o_out = NULL;                    \
    PyObject* o_lo = NULL; PyObject* o_hi = NULL;                                     \
    if (!PyArg_ParseTuple(args, "OLd|OOO", &o_x, &w, &k, &o_out, &o_lo, &o_hi)) return NULL; \
    (void)o_lo; (void)o_hi;                                                           \
    int64_t ldx, ldo;                                                                 \
    PyArrayObject* x = as_panel_ld(o_x, &ldx); if (!x) return NULL;                   \
    npy_intp T = PyArray_DIM(x, 0), S = PyArray_DIM(x, 1);                            \
    PyArrayObject* out = o_out ? as_out_ld(o_out, T, S, &ldo)                         \
                               : alloc_panel(T, S, &ldo);                             \
    if (!out) { Py_DECREF(x); return NULL; }                                          \
    c_fn((const double*)PyArray_DATA(x), (double*)PyArray_DATA(out),                  \
         (int64_t)T, (int64_t)S, (int64_t)w, k, ldx, ldo);                            \
    Py_DECREF(x);                                                                     \
    return (PyObject*)out;                                                            \
}

DEF_TS_PARAM(ts_decay_exp_window, fm_ts_decay_exp_window)
DEF_TS_PARAM(ts_scale,            fm_ts_scale)
DEF_TS_PARAM(ts_quantile,         fm_ts_quantile)

/* panel + panel + window + 标量:f(y, x, w, k, out=None) */
static PyObject* py_ts_regression(PyObject* self, PyObject* args) {
    PyObject *o_y, *o_x; long long w; double k; PyObject* o_out = NULL;
    PyObject* o_lo = NULL; PyObject* o_hi = NULL;
    if (!PyArg_ParseTuple(args, "OOLd|OOO", &o_y, &o_x, &w, &k, &o_out, &o_lo, &o_hi)) return NULL;
    (void)o_lo; (void)o_hi;
    int64_t ldy, ldx, ldo;
    PyArrayObject* y = as_panel_ld(o_y, &ldy); if (!y) return NULL;
    PyArrayObject* x = as_panel_ld(o_x, &ldx); if (!x) { Py_DECREF(y); return NULL; }
    if (PyArray_DIM(y, 0) != PyArray_DIM(x, 0) || PyArray_DIM(y, 1) != PyArray_DIM(x, 1)) {
        PyErr_SetString(PyExc_ValueError, "panel shape mismatch");
        Py_DECREF(y); Py_DECREF(x); return NULL;
    }
    npy_intp T = PyArray_DIM(y, 0), S = PyArray_DIM(y, 1);
    PyArrayObject* out = o_out ? as_out_ld(o_out, T, S, &ldo) : alloc_panel(T, S, &ldo);
    if (!out) { Py_DECREF(y); Py_DECREF(x); return NULL; }
    fm_ts_regression((const double*)PyArray_DATA(y), (const double*)PyArray_DATA(x),
                     (double*)PyArray_DATA(out), (int64_t)T, (int64_t)S, (int64_t)w, k,
                     ldy, ldx, ldo);
    Py_DECREF(y); Py_DECREF(x);
    return (PyObject*)out;
}


/* ---------- Cross-sectional ---------- */

#define DEF_CS(py_name, c_fn)                                                       \
static PyObject* py_##py_name(PyObject* self, PyObject* args) {                     \
    PyObject* o_x;                                                                  \
    if (!PyArg_ParseTuple(args, "O", &o_x)) return NULL;                            \
    PyArrayObject* x = as_panel(o_x); if (!x) return NULL;                          \
    npy_intp T = PyArray_DIM(x, 0), S = PyArray_DIM(x, 1);                          \
    int64_t ld_unused;                                                              \
    PyArrayObject* out = alloc_panel(T, S, &ld_unused);                             \
    if (!out) { Py_DECREF(x); return NULL; }                                        \
    c_fn((const double*)PyArray_DATA(x), (double*)PyArray_DATA(out),                \
         (int64_t)T, (int64_t)S);                                                   \
    Py_DECREF(x);                                                                   \
    return (PyObject*)out;                                                          \
}

DEF_CS(cs_rank,   fm_cs_rank)
DEF_CS(cs_zscore, fm_cs_zscore)
DEF_CS(cs_demean, fm_cs_demean)
DEF_CS(cs_scale,  fm_cs_scale)


/* ---------- Method table ---------- */

static PyMethodDef methods[] = {
    {"abs_",   py_abs_,   METH_VARARGS, "|x|"},
    {"neg",    py_neg,    METH_VARARGS, "-x"},
    {"sign",   py_sign,   METH_VARARGS, "sign(x)"},
    {"log_",   py_log_,   METH_VARARGS, "log(x)"},
    {"sqrt_",  py_sqrt_,  METH_VARARGS, "sqrt(x)"},
    {"square", py_square, METH_VARARGS, "x**2"},
    {"tanh_",  py_tanh_,  METH_VARARGS, "tanh(x)"},
    {"inv",      py_inv,      METH_VARARGS, "1/x (NaN if x==0)"},
    {"s_log_1p", py_s_log_1p, METH_VARARGS, "sign(x)·log(1+|x|)"},
    {"add",    py_add,    METH_VARARGS, "a+b"},
    {"sub",    py_sub,    METH_VARARGS, "a-b"},
    {"mul",    py_mul,    METH_VARARGS, "a*b"},
    {"div_",   py_div_,   METH_VARARGS, "a/b (NaN if b==0)"},
    {"max_b",  py_max_b,  METH_VARARGS, "max(a,b) elementwise"},
    {"min_b",  py_min_b,  METH_VARARGS, "min(a,b) elementwise"},
    {"gt",     py_gt,     METH_VARARGS, "(a>b)?1:0 (NaN propagates)"},
    {"lt",     py_lt,     METH_VARARGS, "(a<b)?1:0 (NaN propagates)"},
    {"not_",   py_not_,   METH_VARARGS, "x>0?0:1 (NaN->NaN)"},
    {"is_nan", py_is_nan, METH_VARARGS, "isnan(x)?1:0 (does NOT propagate NaN)"},
    {"and_",          py_and_,          METH_VARARGS, "(a>0 && b>0)?1:0 (NaN propagates)"},
    {"or_",           py_or_,           METH_VARARGS, "(a>0 || b>0)?1:0 (NaN propagates)"},
    {"equal",         py_equal,         METH_VARARGS, "(a==b)?1:0 (NaN propagates)"},
    {"not_equal",     py_not_equal,     METH_VARARGS, "(a!=b)?1:0 (NaN propagates)"},
    {"greater_equal", py_greater_equal, METH_VARARGS, "(a>=b)?1:0 (NaN propagates)"},
    {"less_equal",    py_less_equal,    METH_VARARGS, "(a<=b)?1:0 (NaN propagates)"},
    {"left_tail",  py_left_tail,  METH_VARARGS, "left_tail(x,k): NaN everything > k"},
    {"right_tail", py_right_tail, METH_VARARGS, "right_tail(x,k): NaN everything < k"},
    {"hump",       py_hump,       METH_VARARGS,
     "hump(x,k): update only when |x[t]-out[t-1]|>k, else hold prev (turnover damper)"},
    {"if_else",    py_if_else,    METH_VARARGS, "if_else(cond,a,b): cond>0?a:b (cond NaN->NaN)"},
    {"trade_when", py_trade_when, METH_VARARGS, "trade_when(entry,alpha,exit): exit>0->0; entry>0->alpha; else hold prev"},

    {"add_const", py_add_const, METH_VARARGS, "x + k (1 SIMD pass, NaN propagates)"},
    {"mul_const", py_mul_const, METH_VARARGS, "x * k (1 SIMD pass, NaN propagates)"},
    {"pow_const", py_pow_const, METH_VARARGS, "sign(x)·|x|^k; 0→0; NaN propagates"},

    {"ts_backfill", py_ts_backfill, METH_VARARGS, "ts_backfill(x,d): last valid value within d (no min_periods)"},
    {"ts_median",   py_ts_median,   METH_VARARGS, "rolling median"},
    {"ts_av_diff",  py_ts_av_diff,  METH_VARARGS, "x - mean(x,d) with NaN-ignoring mean (BRAIN)"},
    {"ts_decay_exp_window", py_ts_decay_exp_window, METH_VARARGS,
     "ts_decay_exp_window(x,d,factor): weights factor^i normalised"},
    {"ts_scale",    py_ts_scale,    METH_VARARGS, "ts_scale(x,d,k): (x-min)/(max-min)+k"},
    {"ts_quantile", py_ts_quantile, METH_VARARGS,
     "ts_quantile(x,d,driver): 0=gaussian 1=uniform 2=cauchy; plotting-position p=(r+.5)/d"},
    {"ts_regression", py_ts_regression, METH_VARARGS,
     "ts_regression(y,x,d,rettype): 0=resid 1=intercept 2=slope 3=fitted 4=R2"},
    {"ts_mean",  py_ts_mean,  METH_VARARGS, "rolling mean (T,S,w)"},
    {"ts_std",   py_ts_std,   METH_VARARGS, "rolling std"},
    {"ts_sum",   py_ts_sum,   METH_VARARGS, "rolling sum"},
    {"ts_max",   py_ts_max,   METH_VARARGS, "rolling max"},
    {"ts_min",   py_ts_min,   METH_VARARGS, "rolling min"},
    {"ts_ref",   py_ts_ref,   METH_VARARGS, "x.shift(w)"},
    {"ts_delta", py_ts_delta, METH_VARARGS, "x - x.shift(w)"},
    {"ts_ema",   py_ts_ema,   METH_VARARGS, "ewm(span=w, adjust=False, min_periods=w).mean"},
    {"ts_wma",   py_ts_wma,   METH_VARARGS, "linear-weighted MA"},
    {"ts_rank",    py_ts_rank,    METH_VARARGS, "rolling avg-rank in [0,1]"},
    {"ts_arg_max", py_ts_arg_max, METH_VARARGS, "bars-since rolling-max"},
    {"ts_arg_min", py_ts_arg_min, METH_VARARGS, "bars-since rolling-min"},
    {"ts_skew",    py_ts_skew,    METH_VARARGS, "rolling 3rd standardized moment"},
    {"ts_kurt",    py_ts_kurt,    METH_VARARGS, "rolling excess kurtosis (m4/m2² − 3)"},
    {"ts_mad",     py_ts_mad,     METH_VARARGS, "rolling mean absolute deviation"},
    {"ts_slope",   py_ts_slope,   METH_VARARGS, "rolling OLS slope vs t"},
    {"ts_chanpos", py_ts_chanpos, METH_VARARGS, "Donchian channel position (x-min)/(max-min) in [0,1]"},
    {"ts_ddown",   py_ts_ddown,   METH_VARARGS, "drawdown from rolling max x/max-1 <= 0"},
    {"ts_zscore",  py_ts_zscore,  METH_VARARGS, "rolling z-score (x-mean)/std, ddof=0"},
    {"ts_snr",     py_ts_snr,     METH_VARARGS, "rolling mean/std (trend t-stat)"},

    {"ts_corr",  py_ts_corr,  METH_VARARGS, "rolling corr (T,S,w)"},
    {"ts_cov",   py_ts_cov,   METH_VARARGS, "rolling cov"},

    {"cs_rank",   py_cs_rank,   METH_VARARGS, "cross-sectional avg-rank in [0,1]"},
    {"cs_zscore", py_cs_zscore, METH_VARARGS, "cross-sectional z-score"},
    {"cs_demean", py_cs_demean, METH_VARARGS, "cross-sectional row-demean"},
    {"cs_scale",  py_cs_scale,  METH_VARARGS, "cross-sectional L1-normalize x/Σ|x|"},
    {NULL, NULL, 0, NULL}
};


BT_MODULE(_ops, methods, "Formulaic alpha C ops kernel.")
