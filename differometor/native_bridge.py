"""JAX bridge to the native C++ Differometor engine (native/, `differometor_native`).

With this bridge the public `differometor` API (`run_build_step`, `simulate`,
`run_simulation_step`, `run`, `run_setups`) runs on the native engine while
staying usable under `jax.jit`, `jax.vjp` and `jax.grad`, so unmodified
callers such as benchmark.py and tests/check_general.py work as before.

How it works
------------
- `run_build_step` builds an immutable native plan (C++ runtime planner) and
  returns `simulation_arrays = {"optimized_parameters": ..., "native_plan":
  NativePlan}`. `NativePlan` is a pytree node with no children, so `jax.jit`
  treats it as static structure (like differometor/static.py); only
  `optimized_parameters` is traced.
- `simulate` is a `jax.custom_vjp` function whose primal, forward and
  backward passes call the native engine through `jax.pure_callback`:
    primal:   Simulator.forward(x)                (passive, no tape)
    forward:  Simulator.forward_with_state(x)     (CoDiPack tape recorded)
    backward: Simulator.backward(state, cotangents)
  All numerical work and differentiation run in C++. JAX only moves arrays.
- JAX's complex cotangents are the complex conjugate of the native
  convention (dL/dRe z + i dL/dIm z); the bridge converts.

Backend selection
-----------------
The native backend is the default when `differometor_native` is importable.
Set DIFFEROMETOR_BACKEND=jax to use the original JAX implementation.

Limitations
-----------
- `jax.vmap` over `simulate` evaluates sequentially (one native call per
  batch element) and cannot be differentiated in reverse mode inside vmap.
- One recorded forward pass is live at a time (CoDiPack's tape is
  process-wide): call the pullback of a forward pass before recording the
  next one, as benchmark.py and the harness do. A stale pullback raises.
- The returned `simulation_arrays` contain only what the native engine
  needs, not upstream's internal arrays. The native package
  (`differometor_native.simulate.run_build_step`) exports those.
"""

from __future__ import annotations

import functools
import itertools
import os
import threading
import warnings
from time import time

import jax
import jax.numpy as jnp
import numpy as np

BACKEND = os.environ.get("DIFFEROMETOR_BACKEND", "native").strip().lower()

ENABLED = False
_IMPORT_ERROR = None
if BACKEND == "native":
    try:
        from differometor_native import native as _native

        ENABLED = True
    except ImportError as exc:  # pragma: no cover - depends on the environment
        _IMPORT_ERROR = exc
        warnings.warn(
            f"differometor: native backend unavailable ({exc}); falling back to the JAX implementation.",
            RuntimeWarning,
            stacklevel=2,
        )
elif BACKEND != "jax":
    raise ValueError("DIFFEROMETOR_BACKEND must be 'native' or 'jax'.")


@jax.tree_util.register_pytree_node_class
class NativePlan:
    """Static (traced-as-structure) handle to an immutable native plan."""

    __slots__ = ("plan", "simulator", "_function", "__weakref__")

    def __init__(self, plan):
        self.plan = plan
        self.simulator = _native.Simulator(plan)
        self._function = None

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    def __repr__(self):
        d = self.plan.dimensions
        return (f"NativePlan(carrier_ports={d['carrier_ports']}, signal_variables={d['signal_variables']}, "
                f"signal_values={d['signal_values']}, optimized_values={d['optimized_values']})")

    def tree_flatten(self):
        return (), self

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return aux_data

    def function(self):
        if self._function is None:
            self._function = _make_function(self)
        return self._function


# The newest recorded state (CoDiPack's tape holds exactly one recording).
_STATE_LOCK = threading.Lock()
_STATES: dict[int, tuple] = {}
_TOKENS = itertools.count(1)


def _make_function(handle: NativePlan):
    d = handle.plan.dimensions
    s, cv = d["carrier_ports"], d["carrier_values"]
    n, b, q = d["signal_variables"], d["signal_values"], d["noise_detectors"]
    noise_shape = np.zeros((q, b)).squeeze().shape  # upstream squeezes the noise
    out_specs = (
        jax.ShapeDtypeStruct((s, cv), jnp.complex128),
        jax.ShapeDtypeStruct((n, b), jnp.complex128),
        jax.ShapeDtypeStruct(noise_shape, jnp.float64),
    )
    simulator = handle.simulator

    def as_vector(x):
        return np.ascontiguousarray(np.asarray(x, dtype=np.float64).reshape(-1))

    def primal_cb(x):
        carrier, signal, noise = simulator.forward(as_vector(x))
        return carrier, signal, np.asarray(noise).reshape(noise_shape)

    def record_cb(x):
        (carrier, signal, noise), state = simulator.forward_with_state(as_vector(x))
        token = next(_TOKENS)
        with _STATE_LOCK:
            _STATES.clear()  # older recordings are invalid anyway
            _STATES[token] = state
        return np.int32(token), carrier, signal, np.asarray(noise).reshape(noise_shape)

    def backward_cb(token, carrier_ct, signal_ct, noise_ct, *, x_shape):
        with _STATE_LOCK:
            state = _STATES.get(int(token))
        if state is None:
            raise RuntimeError(
                "differometor: this pullback's native forward recording was replaced by a newer forward pass. "
                "Call each pullback before recording the next forward pass.")
        grad = simulator.backward(
            state,
            carrier=np.ascontiguousarray(np.conj(np.asarray(carrier_ct, dtype=np.complex128))),
            signal=np.ascontiguousarray(np.conj(np.asarray(signal_ct, dtype=np.complex128))),
            noise=np.ascontiguousarray(np.asarray(noise_ct, dtype=np.float64).reshape(q, b)),
        )
        return np.asarray(grad, dtype=np.float64).reshape(x_shape)

    @jax.custom_vjp
    def native_simulate(x):
        return jax.pure_callback(primal_cb, out_specs, x, vmap_method="sequential")

    def native_simulate_fwd(x):
        token, carrier, signal, noise = jax.pure_callback(
            record_cb, (jax.ShapeDtypeStruct((), jnp.int32),) + out_specs, x, vmap_method="sequential")
        return (carrier, signal, noise), (token, x)

    def native_simulate_bwd(residuals, cotangents):
        token, x = residuals
        carrier_ct, signal_ct, noise_ct = cotangents
        grad = jax.pure_callback(
            functools.partial(backward_cb, x_shape=x.shape),
            jax.ShapeDtypeStruct(x.shape, jnp.float64),
            token, carrier_ct, signal_ct, noise_ct, vmap_method="sequential")
        return (grad,)

    native_simulate.defvjp(native_simulate_fwd, native_simulate_bwd)
    return native_simulate


# ------------------------------------------------------------- public API
def run_build_step(
    setup,
    changing_pairs: list | None = None,
    changing_values=None,
    optimization_pairs: list | None = None,
    timeit: bool = False,
    *,
    signal_solver: str = "packed",
    pack_noise: bool | None = None,
) -> tuple:
    """Build a setup with the native runtime planner (same signature and returns as upstream).

    Returns ``simulation_arrays, detector_indices, mirror_indices,
    beamsplitter_indices, isolator_indices, port_to_index``, or
    ``(seconds, (simulation_arrays, metadata))`` with ``timeit=True``.
    """
    start = time()
    values = None if changing_values is None else np.asarray(changing_values, dtype=np.float64)
    plan = _native.build_plan(setup, changing_pairs, values, optimization_pairs,
                              signal_solver=signal_solver, pack_noise=pack_noise)
    handle = NativePlan(plan)
    optimized = None
    if not optimization_pairs:
        # upstream default: one optimized value mapped onto parameter 0
        optimized = jnp.asarray(plan.parameters[:1])
    simulation_arrays = {"optimized_parameters": optimized, "native_plan": handle}
    det, mirrors, beamsplitters, isolators, port_to_index = plan.metadata
    metadata = (jnp.asarray(det), jnp.asarray(mirrors), jnp.asarray(beamsplitters), jnp.asarray(isolators),
                port_to_index)
    if timeit:
        return time() - start, (simulation_arrays, metadata)
    return simulation_arrays, *metadata


def simulate(optimized_parameters=None, native_plan=None, **unused):
    """Run one native simulation: returns (carrier, signal, noise) as upstream.

    Works eagerly and under jax.jit / jax.vjp / jax.grad (custom VJP backed by
    the native reverse pass). Extra keyword arguments (upstream array names)
    are accepted and ignored.
    """
    if not isinstance(native_plan, NativePlan):
        raise TypeError("differometor.simulate (native backend) needs the arrays returned by run_build_step.")
    if optimized_parameters is None:
        raise TypeError("simulation_arrays['optimized_parameters'] is None: set the optimized values first.")
    return native_plan.function()(jnp.asarray(optimized_parameters, dtype=jnp.float64))


def run_simulation_step(simulation_arrays: dict, jit_simulation: bool = False, timeit: bool = False):
    """Run `simulate` on pre-built arrays (upstream signature)."""
    simulation_function = simulate
    if jit_simulation:
        simulation_function = jax.jit(simulate)
        _ = simulation_function(**simulation_arrays)  # warmup compilation
    start = time()
    results = simulation_function(**simulation_arrays)
    if timeit:
        for result in results:
            jax.block_until_ready(result)
        return time() - start, results
    return results


def run(setup, changing_pairs: list | None = None, changing_values=None, optimization_pairs: list | None = None):
    """Build and simulate (upstream signature and return values)."""
    simulation_arrays, *metadata = run_build_step(setup, changing_pairs, changing_values, optimization_pairs)
    carrier, signal, noise = run_simulation_step(simulation_arrays)
    return carrier, signal, noise, *metadata


def run_setups(setups: list, frequencies):
    """Run each setup over the shared frequency sweep ("f", "frequency")."""
    return [run(setup, [("f", "frequency")], frequencies) for setup in setups]
