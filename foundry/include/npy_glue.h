/* foundry/include/npy_glue.h
 *
 * Shared numpy <-> C glue + Python module boilerplate — extension bindings
 * (foundry/ops/src/module.cpp, foundry/backtest/src/module.cpp) share one copy of
 * as_2d / import_array / PyModuleDef.
 *
 * Include ONLY in the binding file that calls import_array() (numpy's PyArray_API is a
 * per-TU pointer table; each extension module has exactly one numpy TU = its binding file).
 * Pure operator/kernel .cpp files (touching double* only) do not include this header.
 * All helpers are static inline (internal linkage, no ODR collisions).
 *
 * Convention (no defensive programming): kernels assume well-formed inputs; bindings validate
 * only hard constraints that numpy cannot reinterpret directly (dtype / ndim / shape /
 * strides / writeability) — i.e., the segfault boundary.
 */
#ifndef BT_COMMON_NPY_GLUE_H
#define BT_COMMON_NPY_GLUE_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <stdint.h>


/* ---------- (T,S) C-contiguous float64 (for cross-sectional ops / backtest panels) ---------- */
static inline PyArrayObject* bt_as_2d(PyObject* obj) {
    return (PyArrayObject*)PyArray_FROMANY(
        obj, NPY_DOUBLE, 2, 2, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
}

static inline PyArrayObject* bt_as_2d_or_none(PyObject* obj) {
    if (obj == Py_None) return NULL;
    return bt_as_2d(obj);
}

/* ---------- (k,) 1D C-contiguous float64 / int64 ---------- */
static inline PyArrayObject* bt_as_1d_f64(PyObject* obj) {
    return (PyArrayObject*)PyArray_FROMANY(
        obj, NPY_DOUBLE, 1, 1, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
}

static inline PyArrayObject* bt_as_1d_i64(PyObject* obj) {
    return (PyArrayObject*)PyArray_FROMANY(
        obj, NPY_INT64, 1, 1, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
}

/* ---------- (T,S) float64 row-strided view (zero-copy) ----------
 * f64 + 2D + col stride == 8B + row stride >= S*8B (positive, non-overlapping rows)
 * -> borrow reference, *ld = row stride in elements; otherwise FROMANY materialises a
 * C-contiguous copy (ld = S). Column-block slices pass straight through without copying. */
static inline PyArrayObject* bt_as_panel_ld(PyObject* obj, int64_t* ld_out) {
    if (PyArray_Check(obj)) {
        PyArrayObject* a = (PyArrayObject*)obj;
        if (PyArray_TYPE(a) == NPY_DOUBLE && PyArray_NDIM(a) == 2 &&
            PyArray_STRIDE(a, 1) == (npy_intp)sizeof(double) &&
            PyArray_STRIDE(a, 0) >= (npy_intp)(PyArray_DIM(a, 1) * sizeof(double)) &&
            PyArray_ISALIGNED(a)) {
            Py_INCREF(a);
            *ld_out = (int64_t)(PyArray_STRIDE(a, 0) / (npy_intp)sizeof(double));
            return a;
        }
    }
    PyArrayObject* arr = bt_as_2d(obj);
    if (arr) *ld_out = (int64_t)PyArray_DIM(arr, 1);
    return arr;
}

/* out= target: writable float64 (T,S) view with contiguous columns. shape/dtype/strides are the segfault boundary and must be checked. */
static inline PyArrayObject* bt_as_out_ld(PyObject* obj, npy_intp T, npy_intp S, int64_t* ld_out) {
    PyArrayObject* a = (PyArrayObject*)obj;
    if (!PyArray_Check(obj) || PyArray_TYPE(a) != NPY_DOUBLE || PyArray_NDIM(a) != 2 ||
        PyArray_DIM(a, 0) != T || PyArray_DIM(a, 1) != S ||
        PyArray_STRIDE(a, 1) != (npy_intp)sizeof(double) ||
        PyArray_STRIDE(a, 0) < (npy_intp)(S * (npy_intp)sizeof(double)) ||
        !PyArray_ISWRITEABLE(a) || !PyArray_ISALIGNED(a)) {
        PyErr_SetString(PyExc_ValueError,
                        "out= must be a writable float64 (T,S) view with contiguous columns");
        return NULL;
    }
    Py_INCREF(a);
    *ld_out = (int64_t)(PyArray_STRIDE(a, 0) / (npy_intp)sizeof(double));
    return a;
}

static inline PyArrayObject* bt_alloc_panel(npy_intp T, npy_intp S, int64_t* ld_out) {
    npy_intp dims[2] = { T, S };
    *ld_out = (int64_t)S;
    return (PyArrayObject*)PyArray_EMPTY(2, dims, NPY_DOUBLE, 0);
}

static inline PyArrayObject* bt_alloc_1d(npy_intp n, int typenum) {
    npy_intp dims[1] = { n };
    return (PyArrayObject*)PyArray_EMPTY(1, dims, typenum, 0);
}


/* ---------- Module boilerplate: PyModuleDef + PyInit (with import_array) ----------
 * Usage (at the end of a binding file):  BT_MODULE(_bt, methods, "Backtest C/C++ kernel.")
 * C++ bindings automatically get extern "C" linkage (PyInit requires C ABI). */
#ifdef __cplusplus
#define BT_MODULE_EXTERN_C extern "C"
#else
#define BT_MODULE_EXTERN_C
#endif

#define BT_MODULE(modname, methods_arr, docstr)                                    \
    static struct PyModuleDef _bt_moduledef_##modname = {                          \
        PyModuleDef_HEAD_INIT, #modname, docstr, -1, methods_arr,                  \
        NULL, NULL, NULL, NULL };                                                  \
    BT_MODULE_EXTERN_C PyMODINIT_FUNC PyInit_##modname(void) {                     \
        import_array();                                                            \
        return PyModule_Create(&_bt_moduledef_##modname);                          \
    }

#endif /* BT_COMMON_NPY_GLUE_H */
