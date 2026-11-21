# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False
from libc.stdint cimport int64_t, uint8_t

cimport cython
import numpy as np


def forward(const double[::1] template, const int64_t[::1] writes,
            const int64_t[::1] couplings, const uint8_t[::1] conjugate,
            const double[:, ::1] updates, const double[:, ::1] carrier):
    cdef Py_ssize_t f, s, j, p, c
    cdef double ar, ai, br, bi
    cdef Py_ssize_t f_count = updates.shape[0], slots = conjugate.shape[0]
    cdef Py_ssize_t k = couplings.shape[0]
    output = np.empty((f_count, slots), dtype=np.complex128)
    before = np.empty((f_count, k), dtype=np.complex128)
    cdef double[:, ::1] out = output.view(np.float64)
    cdef double[:, ::1] pre = before.view(np.float64)
    with nogil:
        for f in range(f_count):
            c = 0 if carrier.shape[0] == 1 else f
            for s in range(2 * slots):
                out[f,s] = template[s]
            for j in range(writes.shape[0]):
                p = 2*writes[j]
                out[f,p] = updates[f,2*j]
                out[f,p+1] = updates[f,2*j+1]
            for j in range(k):
                p = 2*couplings[j]
                pre[f,2*j] = out[f,p]
                pre[f,2*j+1] = out[f,p+1]
            for j in range(k):
                p = 2*couplings[j]
                ar = -pre[f,2*j]
                ai = -pre[f,2*j+1]
                br = carrier[c,2*j]
                bi = carrier[c,2*j+1]
                out[f,p] = ar*br - ai*bi
                out[f,p+1] = ar*bi + ai*br
            for s in range(slots):
                if conjugate[s]:
                    out[f,2*s+1] = -out[f,2*s+1]
    return output, before


def backward(const int64_t[::1] last_write, const int64_t[::1] last_coupling,
             const uint8_t[::1] conjugate, const double[:, ::1] before,
             const double[:, ::1] carrier, const double[:, ::1] cotangent,
             Py_ssize_t update_count):
    cdef Py_ssize_t f, c, s, k, update, p
    cdef Py_ssize_t slots = conjugate.shape[0], f_count = cotangent.shape[0]
    cdef double gr, gi, ar, ai, br, bi, cr, ci
    template_bar = np.zeros(slots, dtype=np.complex128)
    updates_bar = np.zeros((f_count, update_count), dtype=np.complex128)
    carrier_bar = np.zeros((carrier.shape[0], carrier.shape[1]//2), dtype=np.complex128)
    cdef double[::1] dt = template_bar.view(np.float64)
    cdef double[:, ::1] du = updates_bar.view(np.float64)
    cdef double[:, ::1] dc = carrier_bar.view(np.float64)
    with nogil:
        for f in range(f_count):
            c = 0 if carrier.shape[0] == 1 else f
            for s in range(slots):
                p = 2*s
                gr = cotangent[f,p]
                gi = -cotangent[f,p+1] if conjugate[s] else cotangent[f,p+1]
                k = last_coupling[s]
                if k >= 0:
                    ar = -before[f,2*k]
                    ai = -before[f,2*k+1]
                    dc[c,2*k] += ar*gr - ai*gi
                    dc[c,2*k+1] += ar*gi + ai*gr
                    br = -carrier[c,2*k]
                    bi = -carrier[c,2*k+1]
                    cr = br*gr - bi*gi
                    ci = br*gi + bi*gr
                    gr = cr
                    gi = ci
                update = last_write[s]
                if update >= 0:
                    du[f,2*update] = gr
                    du[f,2*update+1] = gi
                else:
                    dt[p] += gr
                    dt[p+1] += gi
    return template_bar, updates_bar, carrier_bar


def noise_forward(const double[::1] template, const int64_t[::1] writes,
                  const int64_t[::1] rows, const int64_t[::1] columns,
                  const double[:, ::1] updates, const double[:, :, ::1] weights):
    """Evaluate real(w^H Q w) using only stored covariance coefficients."""
    cdef Py_ssize_t f, d, j, p, r, c
    cdef Py_ssize_t frequencies = updates.shape[0], slots = rows.shape[0]
    cdef double ar, ai, br, bi, total
    coefficients = np.empty((frequencies, slots), dtype=np.complex128)
    variance = np.empty((weights.shape[0], frequencies), dtype=np.float64)
    cdef double[:, ::1] values = coefficients.view(np.float64)
    cdef double[:, ::1] result = variance
    with nogil:
        for f in range(frequencies):
            for p in range(2 * slots):
                values[f, p] = template[p]
            for j in range(writes.shape[0]):
                p = 2 * writes[j]
                values[f, p] = updates[f, 2*j]
                values[f, p+1] = updates[f, 2*j+1]
            for d in range(weights.shape[0]):
                total = 0
                for j in range(slots):
                    r = 2 * rows[j]
                    c = 2 * columns[j]
                    ar = values[f, 2*j]
                    ai = values[f, 2*j+1]
                    br = weights[d, f, c]
                    bi = weights[d, f, c+1]
                    total += weights[d, f, r] * (ar*br - ai*bi) + weights[d, f, r+1] * (ar*bi + ai*br)
                result[d, f] = total
    return variance, coefficients


def noise_backward(const int64_t[::1] last_write, const int64_t[::1] rows,
                   const int64_t[::1] columns, const double[:, ::1] values,
                   const double[:, :, ::1] weights, const double[:, ::1] cotangent,
                   Py_ssize_t update_count):
    cdef Py_ssize_t f, d, j, r, c, writer
    cdef Py_ssize_t slots = rows.shape[0]
    cdef double g, ar, ai, br, bi, qr, qi, vr, vi
    template_bar = np.zeros(slots, dtype=np.complex128)
    updates_bar = np.zeros((values.shape[0], update_count), dtype=np.complex128)
    weights_bar = np.zeros((weights.shape[0], weights.shape[1], weights.shape[2]//2), dtype=np.complex128)
    cdef double[::1] dt = template_bar.view(np.float64)
    cdef double[:, ::1] du = updates_bar.view(np.float64)
    cdef double[:, :, ::1] dw = weights_bar.view(np.float64)
    with nogil:
        for f in range(values.shape[0]):
            for j in range(slots):
                r = 2 * rows[j]
                c = 2 * columns[j]
                qr = values[f, 2*j]
                qi = values[f, 2*j+1]
                vr = 0
                vi = 0
                for d in range(weights.shape[0]):
                    g = cotangent[d, f]
                    ar = weights[d, f, r]
                    ai = weights[d, f, r+1]
                    br = weights[d, f, c]
                    bi = weights[d, f, c+1]
                    # JAX's complex VJP uses the algebraic transpose.
                    vr += g * (ar*br + ai*bi)
                    vi += g * (ar*bi - ai*br)
                    dw[d, f, r] += g * (qr*br - qi*bi)
                    dw[d, f, r+1] -= g * (qr*bi + qi*br)
                    dw[d, f, c] += g * (ar*qr + ai*qi)
                    dw[d, f, c+1] += g * (ar*qi - ai*qr)
                writer = last_write[j]
                if writer >= 0:
                    du[f, 2*writer] = vr
                    du[f, 2*writer+1] = vi
                else:
                    dt[2*j] += vr
                    dt[2*j+1] += vi
    return template_bar, updates_bar, weights_bar
