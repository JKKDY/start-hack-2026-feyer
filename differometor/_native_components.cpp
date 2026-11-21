#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <complex>
#include <cstring>

using C = std::complex<double>;
using Kernel = void (*)(const C*, const C*, C*);

static C ipow(C value, int exponent) {
    C result(1);
    const int magnitude = exponent < 0 ? -exponent : exponent;
    for (int i = 0; i < magnitude; ++i) result *= value;
    return exponent < 0 ? C(1) / result : result;
}

#include "_native_components.inc"

static PyObject* evaluate(PyObject*, PyObject* args) {
    int kind;
    PyObject *input, *seed, *result;
    if (!PyArg_ParseTuple(args, "iOOO", &kind, &input, &seed, &result)) return nullptr;
    if (kind < 0 || kind >= sizeof(components) / sizeof(Kernel)) {
        PyErr_SetString(PyExc_ValueError, "Unknown component");
        return nullptr;
    }
    const bool reverse = seed != Py_None;
    Py_buffer x{}, g{}, y{};
    if (PyObject_GetBuffer(input, &x, PyBUF_SIMPLE) ||
        (reverse && PyObject_GetBuffer(seed, &g, PyBUF_SIMPLE)) ||
        PyObject_GetBuffer(result, &y, PyBUF_WRITABLE)) {
        if (y.obj) PyBuffer_Release(&y);
        if (g.obj) PyBuffer_Release(&g);
        if (x.obj) PyBuffer_Release(&x);
        return nullptr;
    }
    const auto input_stride = 7 * sizeof(C);
    const auto output_stride = (reverse ? 7 : 4) * sizeof(C);
    const auto rows = x.len / input_stride;
    const bool valid = x.len % input_stride == 0 && y.len == rows * output_stride &&
                       (!reverse || g.len == rows * 4 * sizeof(C));
    if (valid) {
        Kernel kernel = reverse ? pullbacks[kind] : components[kind];
        Py_BEGIN_ALLOW_THREADS
        for (size_t i = 0; i < rows; ++i) {
            C values[7], seeds[4], output[7];
            std::memcpy(values, static_cast<const char*>(x.buf) + i * input_stride, input_stride);
            if (reverse) std::memcpy(seeds, static_cast<const char*>(g.buf) + i * sizeof(seeds), sizeof(seeds));
            kernel(values, seeds, output);
            std::memcpy(static_cast<char*>(y.buf) + i * output_stride, output, output_stride);
        }
        Py_END_ALLOW_THREADS
    }
    PyBuffer_Release(&y);
    if (reverse) PyBuffer_Release(&g);
    PyBuffer_Release(&x);
    if (!valid) {
        PyErr_SetString(PyExc_ValueError, "Invalid native component buffer size");
        return nullptr;
    }
    Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
    {"evaluate", evaluate, METH_VARARGS, "Evaluate a component or its pullback on contiguous buffers."},
    {nullptr, nullptr, 0, nullptr},
};
static PyModuleDef module = {PyModuleDef_HEAD_INIT, "_native_components", nullptr, -1, methods};
PyMODINIT_FUNC PyInit__native_components() { return PyModule_Create(&module); }
