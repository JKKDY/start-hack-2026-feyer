"""AD adapter for CPU optics. Only a design-only pullback uses the native adjoint."""
import jax
import jax.numpy as jnp
from jax.extend import core
from jax.interpreters import ad, batching, mlir
from jax.custom_derivatives import SymbolicZero


def _materialize(tangent):
    if isinstance(tangent, SymbolicZero):
        return jnp.zeros(tangent.aval.shape, tangent.aval.dtype)
    return ad.instantiate_zeros(tangent)


def _tangent(*operands, reference, count, **_):
    design, *inputs = operands[:count]
    return jax.jvp(lambda p: reference(p, *inputs), (design,), (operands[-1],))[1]


_linear = core.Primitive("optical_design_tangent")
_linear.multiple_results = True
_linear.def_impl(_tangent)
_linear.def_abstract_eval(lambda *_, outputs, **kw: outputs)
mlir.register_lowering(_linear, mlir.lower_fun(_tangent, multiple_results=True))


def _transpose(cotangents, *ops, reference, count, state, **_):
    seeds = tuple(map(ad.instantiate_zeros, cotangents))
    @jax.custom_jvp
    def reverse(*args):
        pullback = jax.tree.unflatten(state, args[count:-len(seeds)])
        return pullback(tuple(args[-len(seeds):]))[0]

    @reverse.defjvp
    def reverse_jvp(primals, tangents):
        def ref(*args):
            p, *other = args[:count]
            pullback = jax.vjp(lambda design: reference(design, *other), p)[1]
            return pullback(tuple(args[-len(seeds):]))[0]
        return reverse(*primals), jax.jvp(ref, primals, tangents)[1]

    return (*([None] * (len(ops)-1)), reverse(*ops[:-1], *seeds))


def _jvp(primals, tangents, **kw):
    return jax.jvp(lambda *x: _tangent(*x, **kw), primals, tuple(map(_materialize, tangents)))


def _batch(args, axes, **kw):
    outputs = jax.vmap(lambda *x: _tangent(*x, **kw), in_axes=axes)(*args)
    return outputs, (0,) * len(outputs)


ad.primitive_transposes[_linear] = _transpose
ad.primitive_jvps[_linear] = _jvp
batching.primitive_batchers[_linear] = _batch


def evaluate(design, inputs, native, reference):
    """Use a general JAX linearization for other active inputs and higher derivatives."""
    leaves, tree = jax.tree.flatten(inputs)
    def ref(p, *args):
        return reference(p, jax.tree.unflatten(tree, args))

    @jax.custom_jvp
    def forward_state(p, *args):
        inputs = jax.tree.unflatten(tree, args)
        return jax.vjp(lambda value: native(value, inputs), p)

    @forward_state.defjvp
    def state_jvp(primals, tangents):
        outputs, saved = forward_state(*primals)
        dy = jax.jvp(ref, primals, tangents)[1]
        # Private residuals are consumed only by reverse(), whose derivative
        # explicitly accounts for their dependence on the original primals.
        zeros = jax.tree.map(lambda x: jnp.zeros(x.shape, jax.typeof(x).to_tangent_aval().dtype), saved)
        return (outputs, saved), (dy, zeros)

    @jax.custom_jvp
    def simulate(p, *args):
        return native(p, jax.tree.unflatten(tree, args))

    def linearize(primals, tangents):
        if isinstance(tangents[0], SymbolicZero) or any(
            not isinstance(t, SymbolicZero) for t in tangents[1:]
        ):
            return simulate(*primals), jax.jvp(ref, primals, tuple(map(_materialize, tangents)))[1]
        outputs, pullback = forward_state(*primals)
        saved, structure = jax.tree.flatten(pullback)
        avals = tuple(jax.core.ShapedArray(x.shape, x.dtype) for x in outputs)
        tangent = _linear.bind(*primals, *saved, tangents[0], count=len(primals),
                               state=structure, reference=ref, outputs=avals)
        return outputs, tuple(tangent)

    simulate.defjvp(linearize, symbolic_zeros=True)
    return simulate(design, *leaves)
