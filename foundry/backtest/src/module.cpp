/* foundry/backtest/src/module.cpp — _bt extension: exports simulate (see kernel.hpp for semantics).
 *
 * simulate(w, open_raw, prev_close, open_ff0, close_ff0, cost_buy, cost_sell, live, block, carry) -> rr (T,)
 *   Price/cost/weight panels are (T,S) float64; live and block are (T,S) uint8; cost_sell / block / carry may be None.
 * Validates only hard constraints that numpy cannot reinterpret directly (dtype, matching shapes) — the kernel's
 * out-of-bounds read boundary.
 */
#include "npy_glue.h"

#include "kernel.hpp"

enum { W, OPEN_RAW, PREV_CLOSE, OPEN_FF0, CLOSE_FF0, COST_BUY, COST_SELL, LIVE, BLOCK, CARRY, N_ARG };

#define F64(i) (arr[i] ? (const double*)PyArray_DATA(arr[i]) : nullptr)
#define U8(i)  (arr[i] ? (const uint8_t*)PyArray_DATA(arr[i]) : nullptr)

static PyObject* py_simulate(PyObject* self, PyObject* args) {
    PyObject* o[N_ARG];
    if (!PyArg_ParseTuple(args, "OOOOOOOOOO", &o[0], &o[1], &o[2], &o[3], &o[4],
                          &o[5], &o[6], &o[7], &o[8], &o[9])) return NULL;
    PyArrayObject* arr[N_ARG] = {};
    bool bad = false;
    for (int i = 0; i < N_ARG && !bad; i++) {
        if (o[i] == Py_None && (i == COST_SELL || i == BLOCK || i == CARRY)) continue;
        arr[i] = (i == LIVE || i == BLOCK)
            ? (PyArrayObject*)PyArray_FROMANY(o[i], NPY_UINT8, 2, 2, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED)
            : bt_as_2d(o[i]);
        bad = arr[i] == NULL;
    }
    npy_intp T = 0, S = 0;
    if (!bad) {
        T = PyArray_DIM(arr[W], 0); S = PyArray_DIM(arr[W], 1);
        for (int i = 1; i < N_ARG && !bad; i++)
            bad = arr[i] && (PyArray_DIM(arr[i], 0) != T || PyArray_DIM(arr[i], 1) != S);
        if (bad) PyErr_SetString(PyExc_ValueError, "simulate: every panel must be (T,S) like w");
    }
    PyArrayObject* rr = bad ? NULL : bt_alloc_1d(T, NPY_DOUBLE);
    if (rr) {
        Py_BEGIN_ALLOW_THREADS
        bt::simulate(F64(W), F64(OPEN_RAW), F64(PREV_CLOSE), F64(OPEN_FF0), F64(CLOSE_FF0),
                     F64(COST_BUY), F64(COST_SELL), U8(LIVE), U8(BLOCK), F64(CARRY),
                     (int64_t)T, (int64_t)S, (double*)PyArray_DATA(rr));
        Py_END_ALLOW_THREADS
    }
    for (int i = 0; i < N_ARG; i++) Py_XDECREF(arr[i]);
    return (PyObject*)rr;
}

static PyMethodDef methods[] = {
    {"simulate", py_simulate, METH_VARARGS,
     "simulate(w, open_raw, prev_close, open_ff0, close_ff0, cost_buy, cost_sell, live, block, carry) -> rr (T,)\n"
     "Daily execution recursion (qnt calc_relative_return generalised); cost_sell/block/carry may be None."},
    {NULL, NULL, 0, NULL}
};

BT_MODULE(_bt, methods, "Market-agnostic daily execution kernel (CPU reference).")
