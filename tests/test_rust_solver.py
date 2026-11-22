"""Complex JAX transformation parity for the CPU optics adapter and the Rust solve."""
import unittest
import jax
import jax.numpy as jnp
import numpy as np

from differometor.sparse_signal import _sweep_kernel, _factor_and_solve
from differometor._native_ad import evaluate


class OpticsADTest(unittest.TestCase):
    def test_design_and_other_inputs(self):
        z = jnp.array([1.2 + .4j, -.7 + .3j])
        a = jnp.array([.2 + .1j, .5 - .2j])
        def reference(z, a):
            return z * a + z.conj() ** 2, (z * a).real
        @jax.custom_vjp
        def native(z, a):
            return reference(z, a)
        def fw(z, a):
            return reference(z, a), (z, a)
        def bw(state, g):
            z, a = state
            return jax.vjp(lambda x: reference(x, a), z)[1](g)[0], None
        native.defvjp(fw, bw)
        f = lambda z, a: evaluate(z, a, native, reference)
        scalar = lambda op, z, a: jnp.sum(jnp.abs(op(z, a)[0]) ** 2)
        for op in (
            lambda f: jax.jit(jax.grad(lambda z: scalar(f, z, a)))(z),
            lambda f: jax.grad(lambda a: scalar(f, z, a))(a),
            lambda f: jax.jvp(lambda z: f(z, a)[0], (z,), (a,))[1],
            lambda f: jax.jvp(lambda x: jax.grad(lambda y: scalar(f, y, a))(x), (z,), (a,))[1],
            lambda f: jax.grad(lambda a: jnp.sum(jax.grad(lambda y: scalar(f, y, a))(z)).real)(a),
            lambda f: jax.vmap(jax.grad(lambda x: scalar(f, x, a)))(jnp.stack([z,z*2])),
        ):
            np.testing.assert_allclose(op(f), op(reference), rtol=1e-11, atol=1e-11)


@unittest.skipUnless(_sweep_kernel() is not None, "optional Rust/SuperLU library is absent")
class RustSparseTest(unittest.TestCase):
    def test_complex_sweep_with_distinct_pivots_and_detector_counts(self):
        mats=np.array([[[0,1j],[2,3]], [[4,1-2j],[1j,3]]],dtype=complex)
        rows=np.array([0,1,0,1],dtype=np.int32)
        ptr=np.array([0,2,4],dtype=np.int32)
        vals=mats.transpose(0,2,1).reshape(2,4)
        b=np.array([[1+2j,3],[2,1-2j]])
        for transpose in (False,True):
            for count in (0,1,2):
                d=np.stack([b.conj(),1j*b],axis=-1)[:,:,:count]
                x,y=_factor_and_solve(vals, rows, ptr,b,d,transpose=transpose)
                op=mats.transpose(0,2,1) if transpose else mats
                np.testing.assert_allclose(x,np.linalg.solve(op,b[...,None])[...,0],rtol=1e-12,atol=1e-12)
                np.testing.assert_allclose(y,np.linalg.solve(op.conj().transpose(0,2,1),d),rtol=1e-12,atol=1e-12)

    def test_public_cavity_derivatives(self):
        import differometor as df
        from differometor.simulate import _auto_cpu, _simulate_impl
        from cases import CASES
        if not _auto_cpu():
            self.skipTest("requires the CPU native adapter")
        case = CASES["cavity_tuning"]()
        arrays, *_ = df.run_build_step(case["setup"], case["changing_pairs"],
                                       jnp.asarray([-15.,7.,19.]), case["pairs"])
        p = jnp.asarray(case["initial"])
        direction = jnp.linspace(.01,.02,len(p))
        def field(fun, params, sweep):
            return fun(**dict(arrays, optimized_parameters=params,
                              carrier_changing_values=sweep))[0]
        def energy(fun, params, sweep):
            return jnp.sum(jnp.abs(field(fun,params,sweep))**2)
        sweep = arrays["carrier_changing_values"]
        for transform in (
            lambda fun: jax.jvp(lambda q: field(fun,q,sweep), (p,), (direction,))[1],
            lambda fun: jax.grad(lambda s: energy(fun,p,s))(sweep),
            lambda fun: jax.jvp(jax.grad(lambda q: energy(fun,q,sweep)), (p,), (direction,))[1],
            lambda fun: jax.vmap(jax.grad(lambda q: energy(fun,q,sweep)))(jnp.stack([p,p])),
        ):
            np.testing.assert_allclose(jax.jit(lambda: transform(df.simulate))(),
                                       jax.jit(lambda: transform(_simulate_impl))(),
                                       rtol=2e-8, atol=1e-8)
