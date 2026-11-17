/* foundry/backtest/src/module.cpp — _bt extension: cash and isolated perpetual execution.
 *
 * Targets use (D,S), shared panels use (E,S), and the explicit schedule maps
 * events to target updates and V measurement buckets. Optional sell/block/carry
 * panels may be None. No target expansion to the event axis is required.
 * Validates only hard constraints that numpy cannot reinterpret directly (dtype, matching shapes) — the kernel's
 * out-of-bounds read boundary.
 */
#include "npy_glue.h"

#include "kernel.hpp"
#include "perpetual.hpp"

enum { W, OPEN_RAW, PREV_CLOSE, OPEN_FF0, CLOSE_FF0, COST_BUY, COST_SELL, LIVE, BLOCK, CARRY,
       DECISION, MEASUREMENT, N_ARG };

#define F64(i) (arr[i] ? (const double*)PyArray_DATA(arr[i]) : nullptr)
#define U8(i)  (arr[i] ? (const uint8_t*)PyArray_DATA(arr[i]) : nullptr)

static PyObject* py_simulate(PyObject* self, PyObject* args) {
    PyObject* o[N_ARG];
    long long measurements;
    if (!PyArg_ParseTuple(args, "OOOOOOOOOOOOL", &o[0], &o[1], &o[2], &o[3], &o[4],
                          &o[5], &o[6], &o[7], &o[8], &o[9], &o[10], &o[11],
                          &measurements)) return NULL;
    PyArrayObject* arr[N_ARG] = {};
    bool bad = false;
    for (int i = 0; i < N_ARG && !bad; i++) {
        if (o[i] == Py_None && (i == COST_SELL || i == BLOCK || i == CARRY)) continue;
        arr[i] = i >= DECISION
            ? (PyArrayObject*)PyArray_FROMANY(o[i], NPY_INT64, 1, 1, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED)
            : (i == LIVE || i == BLOCK)
            ? (PyArrayObject*)PyArray_FROMANY(o[i], NPY_UINT8, 2, 2, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED)
            : bt_as_2d(o[i]);
        bad = arr[i] == NULL;
    }
    npy_intp D = 0, E = 0, S = 0;
    if (!bad) {
        D = PyArray_DIM(arr[W], 0); S = PyArray_DIM(arr[W], 1);
        E = PyArray_DIM(arr[OPEN_RAW], 0);
        bad = D <= 0 || E <= 0 || S <= 0 || measurements <= 0;
        for (int i = 1; i < DECISION && !bad; i++)
            bad = arr[i] && (PyArray_DIM(arr[i], 0) != E || PyArray_DIM(arr[i], 1) != S);
        for (int i = DECISION; i < N_ARG && !bad; i++) bad = PyArray_DIM(arr[i], 0) != E;
        if (bad) PyErr_SetString(PyExc_ValueError, "simulate: expected (D,S) targets, (E,S) panels and (E,) schedule");
    }
    if (!bad) {
        const int64_t* decision = (const int64_t*)PyArray_DATA(arr[DECISION]);
        const int64_t* measurement = (const int64_t*)PyArray_DATA(arr[MEASUREMENT]);
        for (npy_intp e = 0; e < E && !bad; e++)
            bad = decision[e] < -1 || decision[e] >= D || measurement[e] < 0 || measurement[e] >= measurements
                || (e == 0 ? measurement[e] != 0 : measurement[e] < measurement[e - 1]
                    || measurement[e] > measurement[e - 1] + 1);
        if (!bad) bad = measurement[E - 1] != measurements - 1;
        if (bad) PyErr_SetString(PyExc_ValueError, "simulate: schedule indices lie outside target or measurement bounds");
    }
    PyArrayObject* rr = bad ? NULL : bt_alloc_1d(measurements, NPY_DOUBLE);
    if (rr) {
        Py_BEGIN_ALLOW_THREADS
        bt::simulate(F64(W), F64(OPEN_RAW), F64(PREV_CLOSE), F64(OPEN_FF0), F64(CLOSE_FF0),
                     F64(COST_BUY), F64(COST_SELL), U8(LIVE), U8(BLOCK), F64(CARRY),
                     (const int64_t*)PyArray_DATA(arr[DECISION]),
                     (const int64_t*)PyArray_DATA(arr[MEASUREMENT]),
                     (int64_t)E, (int64_t)S, (double*)PyArray_DATA(rr));
        Py_END_ALLOW_THREADS
    }
    for (int i = 0; i < N_ARG; i++) Py_XDECREF(arr[i]);
    return (PyObject*)rr;
}

static PyObject* py_perpetual(PyObject* self, PyObject* args) {
    PyObject *weights, *inputs;
    double leverage;
    long long measurements;
    if (!PyArg_ParseTuple(args, "OOdL", &weights, &inputs, &leverage, &measurements)) return NULL;
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
        const int dtype = i < 3 ? NPY_INT64 : NPY_DOUBLE;
        const int dims = i < 3 ? 1 : 2;
        input[i] = (PyArrayObject*)PyArray_FROMANY(PyTuple_GET_ITEM(inputs, i), dtype, dims, dims,
                                                 NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
        bad = input[i] == NULL;
    }
    const npy_intp B = PyArray_DIM(w, 0), T = PyArray_DIM(w, 1), S = PyArray_DIM(w, 2);
    npy_intp E = 0;
    if (!bad) {
        E = PyArray_DIM(input[0], 0);
        bad = B <= 0 || T <= 0 || S <= 0 || E <= 0 || measurements <= 0;
        for (int i = 1; i < 13 && !bad; ++i)
            bad = PyArray_DIM(input[i], 0) != E || (i >= 3 && PyArray_DIM(input[i], 1) != S);
        if (bad) PyErr_SetString(PyExc_ValueError, "perpetual: expected nonempty (B,D,S), (E,), (E,S) arrays");
    }
    if (!bad) {
        const long long* measurement = (const long long*)PyArray_DATA(input[1]);
        const long long* decision = (const long long*)PyArray_DATA(input[2]);
        for (npy_intp e = 0; e < E && !bad; ++e)
            bad = measurement[e] < 0 || measurement[e] >= measurements || decision[e] < -1 || decision[e] >= T
                || (e == 0 ? measurement[e] != 0 : measurement[e] < measurement[e - 1]
                    || measurement[e] > measurement[e - 1] + 1);
        if (!bad) bad = measurement[E - 1] != measurements - 1;
        if (bad) PyErr_SetString(PyExc_ValueError, "perpetual: schedule indices lie outside target or measurement bounds");
    }
    npy_intp daily[2] = {B, measurements}, events[3] = {B, E, S}, scalar[1] = {B};
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
            (const long long*)PyArray_DATA(input[2]),
            (const double*)PyArray_DATA(input[3]), (const double*)PyArray_DATA(input[4]),
            (const double*)PyArray_DATA(input[5]), (const double*)PyArray_DATA(input[6]),
            (const double*)PyArray_DATA(input[7]), (const double*)PyArray_DATA(input[8]),
            (const double*)PyArray_DATA(input[9]), (const double*)PyArray_DATA(input[10]),
            (const double*)PyArray_DATA(input[11]), (const double*)PyArray_DATA(input[12]),
            (long long)E, (long long)T, measurements, (long long)S, leverage
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
     "simulate(w, open_raw, prev_close, open_ff0, close_ff0, cost_buy, cost_sell, live, block, carry, decision_index, measurement_index, measurement_count) -> rr (V,)\n"
     "Scheduled cash execution; cost_sell/block/carry may be None."},
    {"perpetual", py_perpetual, METH_VARARGS,
     "perpetual(weights, spec_arrays, leverage, measurement_count) -> returns and isolated-margin risk arrays"},
    {NULL, NULL, 0, NULL}
};

BT_MODULE(_bt, methods, "Scheduled cash and isolated-perpetual execution kernels (CPU reference).")
