"""CPU coefficient kernel and transformation rules for the pinned JAX release."""
import ctypes
from .rust_kernel import kernel_path

import jax
import jax.numpy as jnp
import numpy as np
from jax.extend import core
from jax.interpreters import ad, batching, mlir

_library = ctypes.CDLL(str(kernel_path()))
for _symbol in ("feyer_coeff_f", "feyer_coeff_b"):
    jax.ffi.register_ffi_target(
        _symbol, jax.ffi.pycapsule(getattr(_library, _symbol)), platform="cpu"
    )


def _plan(groups, order, varying, width):
    count = sum(len(group[1]) for group in groups if group is not None)
    destination = np.arange(count)
    if order is not None:
        if not np.array_equal(np.sort(order), destination):
            raise ValueError("Expected a permutation of coefficient outputs")
        destination = np.argsort(order)
    plan, writes = [], []
    offset = 0
    for kind, group in enumerate(groups):
        if group is None:
            continue
        inputs, rows, columns = group
        if inputs.shape[1] != 7 or np.any((inputs < 0) | (inputs >= width)):
            raise ValueError("Invalid component input indices")
        if np.any((columns < 0) | (columns >= 4)) or np.any(
            (rows < 0) | (rows >= len(inputs))
        ):
            raise ValueError("Invalid component output selection")
        masks = np.isin(inputs, varying) @ (1 << np.arange(7))
        # Group the writes of each instruction in one pass; retain output order.
        pos = np.argsort(rows, kind="stable")
        ends = np.cumsum(np.bincount(rows, minlength=len(inputs))) + offset
        starts = np.r_[offset, ends[:-1]]
        plan.append(
            np.column_stack((np.full(len(inputs), kind), masks, starts, ends, inputs))
        )
        writes.append(np.column_stack((columns[pos], destination[offset + pos])))
        offset += len(rows)
    return tuple(
        jnp.asarray(np.concatenate(parts), dtype=jnp.int32) for parts in (plan, writes)
    )


def _ffi(name, shape, *args):
    return jax.ffi.ffi_call(
        name, jax.ShapeDtypeStruct(shape, args[0].dtype), vmap_method="sequential"
    )(*args)


# The linearized operation uses the original JAX expressions for forward-mode
# and higher-order derivatives. Its first transpose uses the Rust reverse kernel.
_linear = core.Primitive("differometor_coefficient_tangent")


def _tangent(x, dx, plan, writes, *, reference):
    return jax.jvp(reference, (x,), (dx,))[1]


_linear.def_impl(_tangent)
_linear.def_abstract_eval(
    lambda x, dx, plan, writes, **_: jax.core.ShapedArray(
        (x.shape[0], writes.shape[0]), x.dtype
    )
)
mlir.register_lowering(_linear, mlir.lower_fun(_tangent, multiple_results=False))


def _transpose(g, x, dx, plan, writes, *, reference):
    if isinstance(g, ad.Zero):
        return None, ad.Zero(dx.aval), None, None

    @jax.custom_jvp
    def reverse(parameters, cotangent):
        return _ffi(
            "feyer_coeff_b", parameters.shape, parameters, plan, writes, cotangent
        )

    @reverse.defjvp
    def reverse_jvp(primals, tangents):
        def reference_reverse(parameters, cotangent):
            return jax.vjp(reference, parameters)[1](cotangent)[0]

        return reverse(*primals), jax.jvp(reference_reverse, primals, tangents)[1]

    return None, reverse(x, g), None, None


def _tangent_jvp(primals, tangents, *, reference):
    x, dx, plan, writes = primals
    tx, tdx = (ad.instantiate_zeros(t) for t in tangents[:2])
    return jax.jvp(
        lambda a, b: _tangent(a, b, plan, writes, reference=reference),
        (x, dx),
        (tx, tdx),
    )


def _batched_tangent(args, axes, *, reference):
    return (
        jax.vmap(
            lambda x, dx, p, w: _tangent(x, dx, p, w, reference=reference), in_axes=axes
        )(*args),
        0,
    )


ad.primitive_transposes[_linear] = _transpose
ad.primitive_jvps[_linear] = _tangent_jvp
batching.primitive_batchers[_linear] = _batched_tangent


def evaluate(parameters, groups, order, varying, reference):
    plan, writes = _plan(groups, order, varying, parameters.shape[1])

    @jax.custom_jvp
    def coefficients(x, p, w):
        return _ffi("feyer_coeff_f", (x.shape[0], w.shape[0]), x, p, w)

    @coefficients.defjvp
    def coefficients_jvp(primals, tangents):
        x, p, w = primals
        return coefficients(x, p, w), _linear.bind(
            x, tangents[0], p, w, reference=reference
        )

    return coefficients(parameters, plan, writes)
