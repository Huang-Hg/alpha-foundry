/* foundry/backtest/src/module.cpp — _bt extension: cash and isolated perpetual execution.
 *
 * simulate(w, open_raw, prev_close, open_ff0, close_ff0, cost_buy, cost_sell, live, block, carry) -> rr (T,)
 *   Price/cost/weight panels are (T,S) float64; live and block are (T,S) uint8; cost_sell / block / carry may be None.
 * Validates only hard constraints that numpy cannot reinterpret directly (dtype, matching shapes) — the kernel's
 * out-of-bounds read boundary.
 */
#include "npy_glue.h"

#include "kernel.hpp"
#include "perpetual.hpp"

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

static PyObject* py_perpetual(PyObject* self, PyObject* args) {
    PyObject *weights, *inputs;
    double leverage;
    if (!PyArg_ParseTuple(args, "OOd", &weights, &inputs, &leverage)) return NULL;
    if (!PyTuple_Check(inputs) || PyTuple_GET_SIZE(inputs) != 13) {
        PyErr_SetString(PyExc_ValueError, "perpetual: input spec must contain 13 arrays");
        return NULL;
    }
    PyArrayObject* w = (PyArrayObject*)PyArray_FROMANY(
        weights, NPY_DOUBLE, 3, 3, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!w) return NULL;
    PyArrayObject* input[13] = {};
    PyArrayObject* output[11] = {};
    bool bad = false;
    for (int i = 0; i < 13 && !bad; ++i) {
        const int dtype = i < 2 ? NPY_INT64 : i == 2 ? NPY_UINT8 : NPY_DOUBLE;
        const int dims = i < 3 ? 1 : 2;
        input[i] = (PyArrayObject*)PyArray_FROMANY(PyTuple_GET_ITEM(inputs, i), dtype, dims, dims,
                                                 NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
        bad = input[i] == NULL;
    }
    const npy_intp B = PyArray_DIM(w, 0), T = PyArray_DIM(w, 1), S = PyArray_DIM(w, 2);
    npy_intp E = 0;
    if (!bad) {
        E = PyArray_DIM(input[0], 0);
        bad = B <= 0 || T <= 0 || S <= 0 || E <= 0;
        for (int i = 1; i < 13 && !bad; ++i)
            bad = PyArray_DIM(input[i], 0) != E || (i >= 3 && PyArray_DIM(input[i], 1) != S);
        if (bad) PyErr_SetString(PyExc_ValueError, "perpetual: expected nonempty (B,T,S), (E,), (E,S) arrays");
    }
    if (!bad) {
        const long long* day = (const long long*)PyArray_DATA(input[1]);
        for (npy_intp e = 0; e < E && !bad; ++e) bad = day[e] < 0 || day[e] >= T;
        if (bad) PyErr_SetString(PyExc_ValueError, "perpetual: day_index lies outside the target grid");
    }
    npy_intp daily[2] = {B, T}, events[3] = {B, E, S}, scalar[1] = {B};
    for (int i = 0; i < 11 && !bad; ++i) {
        const int dims = i <= 5 ? 2 : i == 6 ? 3 : 1;
        npy_intp* shape = i <= 5 ? daily : i == 6 ? events : scalar;
        const int dtype = i == 5 || i == 7 ? NPY_INT64 : i == 6 ? NPY_UINT8 : i == 8 ? NPY_BOOL : NPY_DOUBLE;
        output[i] = (PyArrayObject*)PyArray_ZEROS(dims, shape, dtype, 0);
        bad = output[i] == NULL;
    }
    if (!bad) {
        bt::PerpetualInput in {
            (const long long*)PyArray_DATA(input[0]), (const long long*)PyArray_DATA(input[1]),
            (const unsigned char*)PyArray_DATA(input[2]),
            (const double*)PyArray_DATA(input[3]), (const double*)PyArray_DATA(input[4]),
            (const double*)PyArray_DATA(input[5]), (const double*)PyArray_DATA(input[6]),
            (const double*)PyArray_DATA(input[7]), (const double*)PyArray_DATA(input[8]),
            (const double*)PyArray_DATA(input[9]), (const double*)PyArray_DATA(input[10]),
            (const double*)PyArray_DATA(input[11]), (const double*)PyArray_DATA(input[12]),
            (long long)E, (long long)T, (long long)S, leverage
        };
        bt::PerpetualOutput out {
            (double*)PyArray_DATA(output[0]), (double*)PyArray_DATA(output[1]),
            (double*)PyArray_DATA(output[2]), (double*)PyArray_DATA(output[3]),
            (double*)PyArray_DATA(output[4]), (long long*)PyArray_DATA(output[5]),
            (unsigned char*)PyArray_DATA(output[6]), (long long*)PyArray_DATA(output[7]),
            (unsigned char*)PyArray_DATA(output[8]), (double*)PyArray_DATA(output[9]),
            (double*)PyArray_DATA(output[10])
        };
        Py_BEGIN_ALLOW_THREADS
        bt::simulate_perpetual(in, (const double*)PyArray_DATA(w), (long long)B, out);
        Py_END_ALLOW_THREADS
    }
    Py_DECREF(w);
    for (int i = 0; i < 13; ++i) Py_XDECREF(input[i]);
    if (bad) {
        for (int i = 0; i < 11; ++i) Py_XDECREF(output[i]);
        return NULL;
    }
    PyObject* result = PyTuple_New(11);
    if (!result) {
        for (int i = 0; i < 11; ++i) Py_DECREF(output[i]);
        return NULL;
    }
    for (int i = 0; i < 11; ++i) PyTuple_SET_ITEM(result, i, (PyObject*)output[i]);
    return result;
}

static PyMethodDef methods[] = {
    {"simulate", py_simulate, METH_VARARGS,
     "simulate(w, open_raw, prev_close, open_ff0, close_ff0, cost_buy, cost_sell, live, block, carry) -> rr (T,)\n"
     "Daily execution recursion; cost_sell/block/carry may be None."},
    {"perpetual", py_perpetual, METH_VARARGS,
     "perpetual(weights, spec_arrays, leverage) -> returns and isolated-margin risk arrays"},
    {NULL, NULL, 0, NULL}
};

BT_MODULE(_bt, methods, "Market-agnostic daily execution kernel (CPU reference).")
