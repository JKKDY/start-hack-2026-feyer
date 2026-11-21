"""Isolated host-call experiment. The production simulator is unchanged."""
from functools import partial
import jax
import numpy as np
from differometor import _aot_assembly_kernel as kernel


def _real_view(value):
    return np.ascontiguousarray(value, dtype=np.complex128).view(np.float64)


def _host_forward(template, writes, couplings, conjugate, updates, carrier):
    return kernel.forward(
        _real_view(template), np.ascontiguousarray(writes, dtype=np.int64),
        np.ascontiguousarray(couplings, dtype=np.int64),
        np.ascontiguousarray(conjugate, dtype=np.uint8),
        _real_view(updates), _real_view(carrier),
    )


def _host_backward(last_write, last_coupling, conjugate, before, carrier, cotangent, *, update_count):
    return kernel.backward(
        np.ascontiguousarray(last_write, dtype=np.int64),
        np.ascontiguousarray(last_coupling, dtype=np.int64),
        np.ascontiguousarray(conjugate, dtype=np.uint8),
        _real_view(before), _real_view(carrier), _real_view(cotangent), update_count,
    )


def _call_forward(template, writes, couplings, conjugate, updates, carrier):
    return jax.pure_callback(
        _host_forward,
        (jax.ShapeDtypeStruct((updates.shape[0], template.size), template.dtype),
         jax.ShapeDtypeStruct((updates.shape[0], couplings.size), template.dtype)),
        template, writes, couplings, conjugate, updates, carrier, vmap_method="sequential",
    )


@jax.custom_vjp
def _assemble(template, writes, couplings, conjugate, last_write, last_coupling, updates, carrier):
    return _call_forward(template, writes, couplings, conjugate, updates, carrier)[0]


def _assemble_forward(template, writes, couplings, conjugate, last_write, last_coupling, updates, carrier):
    result, before = _call_forward(template, writes, couplings, conjugate, updates, carrier)
    return result, (last_write, last_coupling, conjugate, before, carrier, jax.numpy.empty((0, updates.shape[1]), dtype=updates.dtype))


def _assemble_backward(residual, cotangent):
    last_write, last_coupling, conjugate, before, carrier, update_shape = residual
    # A static shape is needed for the output even when this pullback is JIT compiled.
    update_count = update_shape.shape[1]
    dt, du, dc = jax.pure_callback(
        partial(_host_backward, update_count=update_count),
        (jax.ShapeDtypeStruct((cotangent.shape[1],), cotangent.dtype),
         jax.ShapeDtypeStruct((cotangent.shape[0], update_count), cotangent.dtype),
         jax.ShapeDtypeStruct(carrier.shape, carrier.dtype)),
        last_write, last_coupling, conjugate, before, carrier, cotangent, vmap_method="sequential",
    )
    return dt, None, None, None, None, None, du, dc


_assemble.defvjp(_assemble_forward, _assemble_backward)


def assemble(packing, updates, carrier):
    writes = np.asarray(packing.update_slots, dtype=np.int64)
    couplings = np.asarray(packing.coupling_slots, dtype=np.int64)
    last_write = np.full(packing.template.size, -1, dtype=np.int64)
    last_coupling = np.full(packing.template.size, -1, dtype=np.int64)
    for i, slot in enumerate(writes):
        last_write[slot] = i
    for i, slot in enumerate(couplings):
        last_coupling[slot] = i
    return _assemble(packing.template, writes, couplings, packing.conjugate,
                     last_write, last_coupling, updates, carrier)