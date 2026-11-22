"""
Simulation entry points (native Differometor, API-compatible with upstream).

Public functions keep upstream names, signatures, defaults and return
ordering:

    run(setup, changing_pairs=None, changing_values=None, optimization_pairs=None)
        -> carrier, signal, noise, detector_indices, mirror_indices,
           beamsplitter_indices, isolator_indices, port_to_index
    run_build_step(setup, changing_pairs=None, changing_values=None,
                   optimization_pairs=None, timeit=False,
                   *, signal_solver="packed", pack_noise=None)
        -> simulation_arrays, detector_indices, mirror_indices,
           beamsplitter_indices, isolator_indices, port_to_index
        (timeit=True: elapsed, (simulation_arrays, metadata))
    run_simulation_step(simulation_arrays, jit_simulation=False, timeit=False)
        -> carrier, signal, noise   (timeit=True: elapsed, (carrier, signal, noise))
    simulate(<upstream array arguments>) -> carrier, signal, noise
    run_setups(setups, frequencies) -> list of run(...) results

Execution happens in the native engine:
    Setup --setup_spec--> SetupSpec --build_plan--> SimulationPlan --> Simulator

`simulation_arrays` is a `SimulationArrays` dict holding the upstream array
contract plus a handle to the native plan. `simulate(**simulation_arrays)`
finds that plan again, so the upstream pattern of replacing
`simulation_arrays["optimized_parameters"]` and calling `simulate` still works.
See docs/api-compatibility.md for the differences from upstream.
"""

from __future__ import annotations

import weakref
from time import time
from typing import Any

import numpy as np

from differometor_native import _native
from differometor_native._convert import (
    normalize_changing_values,
    normalize_optimization,
    normalize_pairs,
    setup_spec,
)
from differometor_native.setups import Setup

# Native backend. Tests may replace it with a fake to check the adapter.
_backend = _native.core

#: Argument names of upstream `simulate`, in upstream order.
SIMULATE_ARGUMENTS = (
    "optimized_parameters",
    "optimized_parameter_indices",
    "optimized_value_indices",
    "carrier_changing_parameter_indices",
    "carrier_changing_values",
    "signal_changing_parameter_indices",
    "signal_changing_values",
    "parameters",
    "carrier_matrix",
    "signal_matrix",
    "noise_matrix",
    "noise_selection_vectors",
    "qhd_parameter_indices",
    "qhd_placing_indices",
    "linked_indices",
    "linking_function_indices",
    "indices_to_link",
    "carrier_function_input_indices",
    "carrier_function_indices",
    "carrier_output_indices",
    "carrier_matrix_indices",
    "signal_function_input_indices",
    "signal_function_indices",
    "signal_output_indices",
    "signal_matrix_indices",
    "signal_carrier_indices",
    "signal_carrier_placing_indices",
    "noise_function_input_indices",
    "noise_function_indices",
    "noise_output_indices",
    "noise_system_matrix_indices",
)


class SimulationArrays(dict):
    """Upstream `simulation_arrays` dict plus the native plan that produced it.

    Keys are `simulate` argument names. The `plan` attribute is the native
    `SimulationPlan`; it is not a key, so `simulate(**arrays)` is unchanged.
    """

    plan: Any = None


# ------------------------------------------------------- plan registry
# simulate() receives arrays only. Plans are found again through the
# identity of the `parameters` array, which run_build_step creates and which
# callers do not normally replace. Entries disappear with that array.
_REGISTRY: dict[int, tuple[weakref.ref, Any, list]] = {}


def _register(arrays: SimulationArrays) -> None:
    key_array = arrays.get("parameters")
    if key_array is None:
        return
    key = id(key_array)

    def _drop(_ref: weakref.ref, key: int = key) -> None:
        _REGISTRY.pop(key, None)

    # [simulator] is created lazily and cached per plan.
    _REGISTRY[key] = (weakref.ref(key_array, _drop), arrays.plan, [])


def _lookup(parameters: Any):
    entry = _REGISTRY.get(id(parameters))
    if entry is None or entry[0]() is not parameters:
        return None, None
    _, plan, simulator_box = entry
    if not simulator_box:
        simulator_box.append(_backend.Simulator(plan))
    return plan, simulator_box[0]


# ----------------------------------------------------------- public API
def run_build_step(
    setup: Setup,
    changing_pairs: list | None = None,
    changing_values: Any = None,
    optimization_pairs: list | None = None,
    timeit: bool = False,
    *,
    signal_solver: str = "packed",
    pack_noise: bool | None = None,
) -> tuple:
    """Build a `Setup` into a native plan and its upstream-compatible arrays.

    Returns ``simulation_arrays, detector_indices, mirror_indices,
    beamsplitter_indices, isolator_indices, port_to_index`` (upstream's actual
    behaviour; its docstring lists fewer values), or
    ``(elapsed_seconds, (simulation_arrays, metadata))`` when ``timeit`` is True.
    """
    start = time()
    pairs = normalize_pairs(changing_pairs)
    plan = _backend.build_plan(
        setup_spec(setup),
        pairs,
        normalize_changing_values(changing_values, len(pairs)),
        normalize_optimization(optimization_pairs),
        signal_solver,
        pack_noise,
    )
    simulation_arrays = SimulationArrays(plan.export_simulation_arrays())
    simulation_arrays.plan = plan
    _register(simulation_arrays)
    metadata = tuple(plan.metadata)

    if timeit:
        return time() - start, (simulation_arrays, metadata)
    return simulation_arrays, *metadata


def run_simulation_step(
    simulation_arrays: dict,
    jit_simulation: bool = False,
    timeit: bool = False,
) -> tuple:
    """Run `simulate` on pre-built arrays.

    `jit_simulation` is accepted for compatibility. The native engine is
    compiled ahead of time, so there is no JIT and no warm-up run.
    """
    del jit_simulation  # no JIT natively; see docs/api-compatibility.md
    start = time()
    results = simulate(**simulation_arrays)
    if timeit:
        return time() - start, results
    return results


def simulate(
    optimized_parameters,
    optimized_parameter_indices,
    optimized_value_indices,
    carrier_changing_parameter_indices,
    carrier_changing_values,
    signal_changing_parameter_indices,
    signal_changing_values,
    parameters,
    carrier_matrix,
    signal_matrix,
    noise_matrix,
    noise_selection_vectors,
    qhd_parameter_indices,
    qhd_placing_indices,
    linked_indices,
    linking_function_indices,
    indices_to_link,
    carrier_function_input_indices,
    carrier_function_indices,
    carrier_output_indices,
    carrier_matrix_indices,
    signal_function_input_indices,
    signal_function_indices,
    signal_output_indices,
    signal_matrix_indices,
    signal_carrier_indices,
    signal_carrier_placing_indices,
    noise_function_input_indices,
    noise_function_indices,
    noise_output_indices,
    noise_system_matrix_indices,
    *,
    signal_sparsity=None,
    packed_signal=None,
    packed_noise=None,
):
    """Run one full simulation pass: carrier, signal and quantum noise.

    Returns ``(carrier, signal, noise)`` with upstream shapes:
    carrier (carrier_ports, carrier_values) complex128,
    signal (signal_variables, signal_values) complex128,
    noise squeezed float64 (upstream applies ``.squeeze()``).

    Only ``optimized_parameters`` is read per call. The structural arrays
    identify the native plan built by `run_build_step`; editing them in place
    does not change the plan.
    """
    del signal_sparsity, packed_signal, packed_noise  # planner owns the sparse layout
    plan, simulator = _lookup(parameters)
    if plan is None:
        raise NotImplementedError(
            "simulate() needs arrays produced by run_build_step in this process. Planning from "
            "raw upstream arrays is not supported by the native engine."
        )
    if optimized_parameters is None:
        raise TypeError(
            "simulation_arrays['optimized_parameters'] is None: with optimization_pairs, set the "
            "optimized values before calling simulate (upstream fails here as well)."
        )
    values = np.ascontiguousarray(optimized_parameters, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(
            "optimized_parameters must be 1-D (num_optimized_values,); batched evaluation is "
            f"not supported by simulate(). Got shape {values.shape}."
        )
    carrier, signal, noise = simulator.forward(values)
    return carrier, signal, np.squeeze(noise)


def simulate_vjp(simulation_arrays: dict, optimized_parameters: Any = None, *, cotangent_convention: str = "jax"):
    """Forward pass plus pullback, like
    ``jax.vjp(lambda p: simulate(**dict(simulation_arrays, optimized_parameters=p)), p)``.

    Returns ``(outputs, pullback)`` where ``outputs = (carrier, signal, noise)``
    has the same layout as `simulate`, and ``pullback((carrier_ct, signal_ct,
    noise_ct))`` returns a 1-tuple with the gradient with respect to the
    optimized parameters. Any cotangent may be None (zero).

    The gradient is computed natively (CoDiPack reverse mode, implicit
    adjoints through the SuperLU solves). `cotangent_convention="jax"`
    (default) accepts complex cotangents as produced by JAX, which are the
    complex conjugate of the native convention ``dL/dRe z + 1j dL/dIm z``;
    pass ``"native"`` to use the native convention directly.
    """
    if cotangent_convention not in ("jax", "native"):
        raise ValueError("cotangent_convention must be 'jax' or 'native'.")
    x = simulation_arrays["optimized_parameters"] if optimized_parameters is None else optimized_parameters
    if x is None:
        raise TypeError("optimized_parameters is None; pass it or set simulation_arrays['optimized_parameters'].")
    plan, simulator = _lookup(simulation_arrays["parameters"])
    if plan is None:
        raise NotImplementedError("simulate_vjp() needs arrays produced by run_build_step in this process.")
    x_shape = np.shape(x)
    values = np.ascontiguousarray(np.asarray(x, dtype=np.float64).reshape(-1))
    (carrier, signal, noise2d), state = simulator.forward_with_state(values)
    conjugate = cotangent_convention == "jax"

    def complex_cotangent(value, shape):
        if value is None:
            return None
        array = np.asarray(value, dtype=np.complex128).reshape(shape)
        return np.ascontiguousarray(np.conj(array) if conjugate else array)

    def pullback(cotangents):
        carrier_ct, signal_ct, noise_ct = cotangents
        noise_array = None
        if noise_ct is not None:
            noise_array = np.ascontiguousarray(np.real(np.asarray(noise_ct)).reshape(noise2d.shape), dtype=np.float64)
        grad = simulator.backward(
            state,
            carrier=complex_cotangent(carrier_ct, carrier.shape),
            signal=complex_cotangent(signal_ct, signal.shape),
            noise=noise_array,
        )
        return (grad.reshape(x_shape),)

    return (carrier, signal, np.squeeze(noise2d)), pullback


def run(
    setup: Setup,
    changing_pairs: list | None = None,
    changing_values: Any = None,
    optimization_pairs: list | None = None,
) -> tuple:
    """Build a setup and run the simulation.

    Returns upstream's 8 values: carrier, signal, noise and build()'s metadata
    (detector, mirror, beamsplitter, isolator indices, port_to_index).
    """
    simulation_arrays, *metadata = run_build_step(setup, changing_pairs, changing_values, optimization_pairs)
    carrier, signal, noise = run_simulation_step(simulation_arrays)
    return carrier, signal, noise, *metadata


def run_setups(setups: list[Setup], frequencies: Any):
    """Run each setup over the shared frequency sweep ("f", "frequency")."""
    results = []
    for setup in setups:
        results.append(run(setup, [("f", "frequency")], frequencies))
    return results
