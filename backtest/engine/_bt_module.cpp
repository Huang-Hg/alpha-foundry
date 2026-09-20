/* backtest/engine/_bt_module.cpp —— backtest._bt Python 扩展 binding(C++)。
 *
 * 导出:
 *   portfolio_bt_equity(...)  US 权益(EquityPolicy;佣金+借券费,无强平)
 *   portfolio_bt_ashare(...)  中国 A 股(ASharePolicy;印花税卖出单边 + trade_block)
 *   portfolio_bt_qnt(...)     Quantiacs qnt 1:1 复刻(独立递推,engine/qnt_bt.cpp)
 * (旧 portfolio_bt(PerpPolicy)/ stop_latch_scan 随 1h/5m crypto 栈 2026-07-25 删除。)
 * numpy glue / module 样板复用 common/npy_glue.h;引擎核心 engine/portfolio。
 */
#include "npy_glue.h"

#include <cmath>

#include "engine.hpp"
#include "qnt_bt.hpp"


/* equity/turnover/cost (T,) + 标量汇总 → 结果 dict。消费(steal)三个 (T,) 数组的引用。 */
static PyObject* pack_result(PyArrayObject* equity_arr, PyArrayObject* turnover_arr,
                             PyArrayObject* cost_arr, int64_t liq_at,
                             double init_cash, double bars_per_year, npy_intp T) {
    const double* eq = (const double*)PyArray_DATA(equity_arr);
    const double* tu = (const double*)PyArray_DATA(turnover_arr);
    double total_return = (init_cash > 0.0) ? (eq[T - 1] / init_cash - 1.0) : 0.0;

    double sum_lr = 0.0, sum_lr_sq = 0.0;
    int64_t n_r = 0;
    for (npy_intp i = 1; i < T; i++) {
        if (eq[i - 1] <= 0.0 || eq[i] <= 0.0) break;
        double lr = std::log(eq[i] / eq[i - 1]);
        sum_lr += lr; sum_lr_sq += lr * lr; n_r++;
    }
    double sharpe = 0.0;
    if (n_r > 1) {
        double mean = sum_lr / (double)n_r;
        double var  = (sum_lr_sq - mean * sum_lr) / (double)(n_r - 1);
        if (var > 0.0) sharpe = mean / std::sqrt(var) * std::sqrt(bars_per_year);
    }
    double mean_turnover = 0.0;
    if (T > 1) {
        double s = 0.0;
        for (npy_intp i = 1; i < T; i++) s += tu[i];
        mean_turnover = s / (double)(T - 1);
    }

    PyObject* d = PyDict_New();
    PyDict_SetItemString(d, "equity",    (PyObject*)equity_arr);
    PyDict_SetItemString(d, "turnover",  (PyObject*)turnover_arr);
    PyDict_SetItemString(d, "cost_drag", (PyObject*)cost_arr);
    PyObject* v;                       /* SetItemString 不偷引用:标量临时须配对 DECREF,否则每调用漏 4 对象 */
    v = PyLong_FromLongLong((long long)liq_at); PyDict_SetItemString(d, "liq_at",        v); Py_DECREF(v);
    v = PyFloat_FromDouble(total_return);       PyDict_SetItemString(d, "total_return",  v); Py_DECREF(v);
    v = PyFloat_FromDouble(sharpe);             PyDict_SetItemString(d, "sharpe",        v); Py_DECREF(v);
    v = PyFloat_FromDouble(mean_turnover);      PyDict_SetItemString(d, "mean_turnover", v); Py_DECREF(v);
    Py_DECREF(equity_arr); Py_DECREF(turnover_arr); Py_DECREF(cost_arr);
    return d;
}

#define _DAT(a)  ((a) ? (const double*)PyArray_DATA(a) : nullptr)


/* ============================ portfolio_bt_equity(US 权益 / EquityPolicy)============================
 * portfolio_bt_equity(alpha, close, open, init_cash, skip_warmup, cost_rate, daily_borrow, bars_per_year,
 *                     [raw_weights=0]) -> dict
 * close/open = 复权 OHLC(总收益);无 margin-call 强平/impact。long-short(空腿计借券费)。
 * 权重外部构造(raw_weights=1 直传)→ leverage_cap 无意义,不暴露。 */
static PyObject* py_portfolio_bt_equity(PyObject* self, PyObject* args) {
    PyObject *o_alpha, *o_close, *o_open;
    double init_cash, cost_rate, daily_borrow;
    long long skip_warmup_ll;
    int raw_weights = 0;
    double bars_per_year;                               // 必传:MarketProfile 单源年化,无默认

    if (!PyArg_ParseTuple(args, "OOOdLddd|i",
                          &o_alpha, &o_close, &o_open,
                          &init_cash, &skip_warmup_ll, &cost_rate, &daily_borrow, &bars_per_year,
                          &raw_weights)) {
        return NULL;
    }

    PyArrayObject* alpha_arr = bt_as_2d(o_alpha);
    if (!alpha_arr) return NULL;
    PyArrayObject* close_arr = bt_as_2d(o_close);
    if (!close_arr) { Py_DECREF(alpha_arr); return NULL; }
    PyArrayObject* open_arr = bt_as_2d(o_open);
    if (!open_arr) { Py_DECREF(alpha_arr); Py_DECREF(close_arr); return NULL; }

    npy_intp T = PyArray_DIM(alpha_arr, 0), S = PyArray_DIM(alpha_arr, 1);
    if (PyArray_DIM(close_arr, 0) != T || PyArray_DIM(close_arr, 1) != S ||
        PyArray_DIM(open_arr, 0) != T || PyArray_DIM(open_arr, 1) != S) {
        PyErr_SetString(PyExc_ValueError, "close/open shape must match alpha (T,S)");
        Py_DECREF(alpha_arr); Py_DECREF(close_arr); Py_DECREF(open_arr); return NULL;
    }
    bt::EquityPolicy policy(cost_rate, daily_borrow);

    npy_intp shape_T[1] = { T };
    PyArrayObject* equity_arr   = (PyArrayObject*)PyArray_EMPTY(1, shape_T, NPY_DOUBLE, 0);
    PyArrayObject* turnover_arr = (PyArrayObject*)PyArray_EMPTY(1, shape_T, NPY_DOUBLE, 0);
    PyArrayObject* cost_arr     = (PyArrayObject*)PyArray_EMPTY(1, shape_T, NPY_DOUBLE, 0);
    int64_t liq_at = -1;

    Py_BEGIN_ALLOW_THREADS
    bt::run_portfolio(
        _DAT(alpha_arr), _DAT(close_arr), _DAT(open_arr),
        nullptr,                                /* 美股研究口径不建模 halt 冻结 */
        (int64_t)T, (int64_t)S, init_cash, (int64_t)skip_warmup_ll,
        1.0, raw_weights,                       /* lev:raw_weights 旁路不触及 */
        policy,
        (double*)PyArray_DATA(equity_arr), (double*)PyArray_DATA(turnover_arr),
        (double*)PyArray_DATA(cost_arr), &liq_at);
    Py_END_ALLOW_THREADS

    Py_DECREF(alpha_arr); Py_DECREF(close_arr); Py_DECREF(open_arr);
    return pack_result(equity_arr, turnover_arr, cost_arr, liq_at, init_cash, bars_per_year, T);
}


/* ============================ portfolio_bt_ashare(中国 A 股 / ASharePolicy)============================
 * portfolio_bt_ashare(alpha, close, open, init_cash, skip_warmup, commission, stamp_tax_sell, transfer_fee, bars_per_year,
 *                     [raw_weights=0, trade_block=None]) -> dict
 * close/open = qfq 复权 OHLC(总收益);佣金双边 + 印花税卖出单边 + 过户费双边;long-only,无 funding/强平。
 * trade_block:(T,S) int8 涨跌停/停牌方向冻结掩码 | None(bit0=涨停禁买、bit1=跌停禁卖、3=停牌双禁)。
 * 权重外部构造(raw_weights=1 直传)→ leverage_cap 无意义,不暴露。 */
static PyObject* py_portfolio_bt_ashare(PyObject* self, PyObject* args) {
    PyObject *o_alpha, *o_close, *o_open, *o_tblock = Py_None;
    double init_cash, commission, stamp_tax_sell, transfer_fee;
    long long skip_warmup_ll;
    double bars_per_year;             // 必传:MarketProfile.bars_per_year 单源年化,无默认
    int raw_weights = 0;

    if (!PyArg_ParseTuple(args, "OOOdLdddd|iO",
                          &o_alpha, &o_close, &o_open,
                          &init_cash, &skip_warmup_ll, &commission, &stamp_tax_sell, &transfer_fee, &bars_per_year,
                          &raw_weights, &o_tblock)) {
        return NULL;
    }

    PyArrayObject* alpha_arr = bt_as_2d(o_alpha);
    if (!alpha_arr) return NULL;
    PyArrayObject* close_arr = bt_as_2d(o_close);
    if (!close_arr) { Py_DECREF(alpha_arr); return NULL; }
    PyArrayObject* open_arr = bt_as_2d(o_open);
    if (!open_arr) { Py_DECREF(alpha_arr); Py_DECREF(close_arr); return NULL; }

    npy_intp T = PyArray_DIM(alpha_arr, 0), S = PyArray_DIM(alpha_arr, 1);
    if (PyArray_DIM(close_arr, 0) != T || PyArray_DIM(close_arr, 1) != S ||
        PyArray_DIM(open_arr, 0) != T || PyArray_DIM(open_arr, 1) != S) {
        PyErr_SetString(PyExc_ValueError, "close/open shape must match alpha (T,S)");
        Py_DECREF(alpha_arr); Py_DECREF(close_arr); Py_DECREF(open_arr); return NULL;
    }

    PyArrayObject* tblock_arr = NULL;
    if (o_tblock != Py_None) {
        tblock_arr = (PyArrayObject*)PyArray_FROMANY(
            o_tblock, NPY_INT8, 2, 2, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED | NPY_ARRAY_FORCECAST);
        if (!tblock_arr ||
            PyArray_DIM(tblock_arr, 0) != T || PyArray_DIM(tblock_arr, 1) != S) {
            PyErr_SetString(PyExc_ValueError, "trade_block must be int8 (T,S) matching alpha");
            Py_XDECREF(tblock_arr);
            Py_DECREF(alpha_arr); Py_DECREF(close_arr); Py_DECREF(open_arr); return NULL;
        }
    }

    bt::ASharePolicy policy(commission, stamp_tax_sell, transfer_fee);

    npy_intp shape_T[1] = { T };
    PyArrayObject* equity_arr   = (PyArrayObject*)PyArray_EMPTY(1, shape_T, NPY_DOUBLE, 0);
    PyArrayObject* turnover_arr = (PyArrayObject*)PyArray_EMPTY(1, shape_T, NPY_DOUBLE, 0);
    PyArrayObject* cost_arr     = (PyArrayObject*)PyArray_EMPTY(1, shape_T, NPY_DOUBLE, 0);
    int64_t liq_at = -1;

    Py_BEGIN_ALLOW_THREADS
    bt::run_portfolio(
        _DAT(alpha_arr), _DAT(close_arr), _DAT(open_arr),
        tblock_arr ? (const int8_t*)PyArray_DATA(tblock_arr) : nullptr,
        (int64_t)T, (int64_t)S, init_cash, (int64_t)skip_warmup_ll,
        1.0, raw_weights,                       /* lev:raw_weights 旁路不触及 */
        policy,
        (double*)PyArray_DATA(equity_arr), (double*)PyArray_DATA(turnover_arr),
        (double*)PyArray_DATA(cost_arr), &liq_at);
    Py_END_ALLOW_THREADS

    Py_DECREF(alpha_arr); Py_DECREF(close_arr); Py_DECREF(open_arr); Py_XDECREF(tblock_arr);
    return pack_result(equity_arr, turnover_arr, cost_arr, liq_at, init_cash, bars_per_year, T);
}


/* ============================ portfolio_bt_qnt(Quantiacs 1:1)============================
 * portfolio_bt_qnt(w, open_raw, prev_close, open_ff0, close_ff0, slippage, unlocked, i0, ppy) -> dict
 * 全部 (T,S) f64(unlocked 为 uint8);静态量由 rl.backtest_reward.prepare_qnt_static 预算。
 * 返回 rr/equity (T,) + sharpe(arith)/mean_return/volatility/max_drawdown/total_return(窗 [i0:])。 */
static PyObject* py_portfolio_bt_qnt(PyObject* self, PyObject* args) {
    PyObject *o_w, *o_oraw, *o_pcl, *o_off, *o_cff, *o_slip, *o_ul;
    long long i0_ll;
    double ppy;

    if (!PyArg_ParseTuple(args, "OOOOOOOLd",
                          &o_w, &o_oraw, &o_pcl, &o_off, &o_cff, &o_slip, &o_ul,
                          &i0_ll, &ppy)) {
        return NULL;
    }

    PyObject* o_f64[6] = { o_w, o_oraw, o_pcl, o_off, o_cff, o_slip };
    PyArrayObject* arr[6] = {};
    int arg_bad = 0;
    for (int i = 0; i < 6 && !arg_bad; i++) {
        arr[i] = bt_as_2d(o_f64[i]);
        arg_bad = (arr[i] == NULL);
    }
    PyArrayObject* ul_arr = NULL;
    if (!arg_bad) {
        ul_arr = (PyArrayObject*)PyArray_FROMANY(
            o_ul, NPY_UINT8, 2, 2, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED | NPY_ARRAY_FORCECAST);
        arg_bad = (ul_arr == NULL);
    }
    npy_intp T = 0, S = 0;
    if (!arg_bad) {
        T = PyArray_DIM(arr[0], 0); S = PyArray_DIM(arr[0], 1);
        for (int i = 1; i < 6 && !arg_bad; i++)
            arg_bad = (PyArray_DIM(arr[i], 0) != T || PyArray_DIM(arr[i], 1) != S);
        if (!arg_bad)
            arg_bad = (PyArray_DIM(ul_arr, 0) != T || PyArray_DIM(ul_arr, 1) != S);
        if (arg_bad)
            PyErr_SetString(PyExc_ValueError, "all panels must be (T,S) matching w");
    }
    if (arg_bad) {
        for (int i = 0; i < 6; i++) Py_XDECREF(arr[i]);
        Py_XDECREF(ul_arr);
        return NULL;
    }

    npy_intp shape_T[1] = { T };
    PyArrayObject* rr_arr = (PyArrayObject*)PyArray_EMPTY(1, shape_T, NPY_DOUBLE, 0);
    PyArrayObject* eq_arr = (PyArrayObject*)PyArray_EMPTY(1, shape_T, NPY_DOUBLE, 0);
    bt::QntStats st;

    Py_BEGIN_ALLOW_THREADS
    st = bt::qnt_bt(_DAT(arr[0]), _DAT(arr[1]), _DAT(arr[2]), _DAT(arr[3]), _DAT(arr[4]),
                    _DAT(arr[5]), (const uint8_t*)PyArray_DATA(ul_arr),
                    (int64_t)T, (int64_t)S, (int64_t)i0_ll, ppy,
                    (double*)PyArray_DATA(rr_arr), (double*)PyArray_DATA(eq_arr));
    Py_END_ALLOW_THREADS

    for (int i = 0; i < 6; i++) Py_DECREF(arr[i]);
    Py_DECREF(ul_arr);

    PyObject* d = PyDict_New();
    PyDict_SetItemString(d, "rr",     (PyObject*)rr_arr);
    PyDict_SetItemString(d, "equity", (PyObject*)eq_arr);
    PyObject* v;
    v = PyFloat_FromDouble(st.sharpe);       PyDict_SetItemString(d, "sharpe",        v); Py_DECREF(v);
    v = PyFloat_FromDouble(st.mean_return);  PyDict_SetItemString(d, "mean_return",   v); Py_DECREF(v);
    v = PyFloat_FromDouble(st.volatility);   PyDict_SetItemString(d, "volatility",    v); Py_DECREF(v);
    v = PyFloat_FromDouble(st.max_drawdown); PyDict_SetItemString(d, "max_drawdown",  v); Py_DECREF(v);
    v = PyFloat_FromDouble(st.total_return); PyDict_SetItemString(d, "total_return",  v); Py_DECREF(v);
    Py_DECREF(rr_arr); Py_DECREF(eq_arr);
    return d;
}


static PyMethodDef methods[] = {
    {"portfolio_bt_equity", py_portfolio_bt_equity, METH_VARARGS,
     "portfolio_bt_equity(alpha, close, open, init_cash, skip_warmup, cost_rate, daily_borrow, bars_per_year, "
     "[raw_weights=0]) -> dict  (US equity / EquityPolicy)"},
    {"portfolio_bt_ashare", py_portfolio_bt_ashare, METH_VARARGS,
     "portfolio_bt_ashare(alpha, close, open, init_cash, skip_warmup, commission, stamp_tax_sell, transfer_fee, bars_per_year, "
     "[raw_weights=0, trade_block=None]) -> dict  (China A-share / ASharePolicy)"},
    {"portfolio_bt_qnt", py_portfolio_bt_qnt, METH_VARARGS,
     "portfolio_bt_qnt(w, open_raw, prev_close, open_ff0, close_ff0, slippage, unlocked, i0, ppy) -> dict  "
     "(Quantiacs qnt calc_relative_return 1:1 + cryptodaily arithmetic stats)"},
    {NULL, NULL, 0, NULL}
};

BT_MODULE(_bt, methods, "Backtest C++ engine (policy-abstracted: EquityPolicy / ASharePolicy + Quantiacs qnt kernel).")
