"""
JAX-based optical simulation runner.

This module builds linear systems from a `differometor` `Setup` and solves:
- a carrier (steady-state) system,
- a signal (sideband) system,
- and a quantum noise propagation system.

The main public entry points are:
- `run`: build once and simulate once
- `run_setups`: run multiple setups over a shared frequency grid
- `run_with_parameter_sets`: run the same setup over many optimized-parameter sets

Conventions used throughout:
- Arrays are JAX arrays (`jax.numpy`).
- Shapes are documented explicitly for every function.
- All indices are integer arrays that index into parameter vectors and matrix entries.
"""

import jax
from time import time
import jax.numpy as jnp
import numpy as np
from jax import lax, vmap, jit

from differometor.setups import Setup
from differometor.build import build, pairs_to_arrays, prepare_arrays, LINKING_FUNCTIONS
from differometor.components import FUNCTIONS, UNIT_VACUUM, H_PLANCK, F0
from differometor.sparse_signal import build_signal_pattern, pack_noise_template, pack_signal_template, solve_signal_and_adjoint


jax.config.update("jax_enable_x64", True)


def solve(matrix: jnp.ndarray, right_hand_side: jnp.ndarray) -> jnp.ndarray:
    """
    Solve a dense linear system for each batch item.

    Parameters
    ----------
    matrix : jnp.ndarray
        System matrix with shape (..., system_size, system_size).
    right_hand_side : jnp.ndarray
        Right-hand side vector with shape (..., system_size).

    Returns
    -------
    solution : jnp.ndarray
        Solution vector with shape (..., system_size).

    Notes
    -----
    Both inputs are cast to complex128 to reduce numerical issues during solving.

    The function intentionally returns the raw complex solution rather than derived
    quantities (such as phase or amplitude), because some downstream transforms
    (notably `jnp.angle`) can produce invalid gradients in some configurations.
    """
    right_hand_side = right_hand_side.astype(jnp.complex128)
    matrix = matrix.astype(jnp.complex128)
    return jnp.linalg.solve(matrix, right_hand_side)


def component_coefficients(parameters, function_input_indices, output_indices, function_indices):
    """Evaluate and select the coefficients written by the component instructions."""
    if isinstance(function_indices, tuple):
        coefficients = []
        for function, group in zip(FUNCTIONS, function_indices, strict=True):
            if group is None:
                continue
            inputs, rows, columns = group
            values = vmap(function)(parameters[inputs])
            coefficients.append(values[rows, columns])
        return jnp.concatenate(coefficients)[output_indices]

    function_inputs = parameters[function_input_indices]
    output_matrix = vmap(lambda i, x: lax.switch(i, FUNCTIONS, x))(function_indices, function_inputs)
    return output_matrix[output_indices[0], output_indices[1]]


def update(
    parameters: jnp.ndarray,
    matrix: jnp.ndarray,
    function_input_indices: jnp.ndarray,
    output_indices: tuple[jnp.ndarray, jnp.ndarray],
    matrix_indices: tuple[jnp.ndarray, jnp.ndarray],
    function_indices: jnp.ndarray,
) -> jnp.ndarray:
    """
    Update selected entries of a system matrix by evaluating component functions.

    This function takes a parameter vector, selects the subset that serves as
    inputs to component transfer functions, evaluates those functions, selects
    certain outputs, and writes them into `matrix` at specified indices.

    Parameters
    ----------
    parameters : jnp.ndarray
        Parameter values for one simulation instance, shape (P,).
    matrix : jnp.ndarray
        Base matrix to update, shape (N, N + 1) for augmented systems, or (N, N).
        The function writes entries in-place (functionally) using `.at[...].set(...)`.
    function_input_indices : jnp.ndarray
        Indices into `parameters` selecting inputs to the component functions,
        shape (K,).
    output_indices : tuple[jnp.ndarray, jnp.ndarray]
        Pair of index arrays selecting entries from the evaluated output matrix.
        Each element has shape (M,). Together they define M output positions.
    matrix_indices : tuple[jnp.ndarray, jnp.ndarray]
        Pair of index arrays selecting where to write into `matrix`.
        Each element has shape (M,).
    function_indices : jnp.ndarray
        Indices selecting which function to use for each input, shape (K,).
        Values index into the global `FUNCTIONS` list.

    Returns
    -------
    updated_matrix : jnp.ndarray
        Matrix with updated entries, same shape as `matrix`.

    Notes
    -----
    `lax.switch` selects functions from the global `FUNCTIONS` list. In JAX, a
    Python list of callables referenced as a global is treated as static during
    tracing, which keeps the function jittable as long as the list is stable.
    """
    outputs = component_coefficients(parameters, function_input_indices, output_indices, function_indices)
    return matrix.at[matrix_indices[0], matrix_indices[1]].set(outputs)


def expand_parameters(array: jnp.ndarray, indices: jnp.ndarray, values: jnp.ndarray) -> jnp.ndarray:
    """
    Expand a single parameter row into multiple rows by replacing selected entries.

    Parameters
    ----------
    array : jnp.ndarray
        Base parameter array with shape (1, P).
    indices : jnp.ndarray
        Column indices to replace, shape (V,).
    values : jnp.ndarray
        Replacement values stacked row-wise for each index, shape (V, R),
        where R is the number of expanded rows.

        Example:
        - array shape (1, 5)
        - indices shape (2,)
        - values shape (2, 3)
        Output has shape (3, 5), where the two indexed columns are filled from
        `values[:, row]`.

    Returns
    -------
    expanded : jnp.ndarray
        Expanded parameter array with shape (R, P).

    Notes
    -----
    This implementation uses indexed updates that remain compatible with JAX
    transformations and compilation.
    """
    tiled_array = jnp.tile(array, (values.shape[1], 1))
    row_indices = jnp.repeat(jnp.arange(values.shape[1]), len(indices))
    column_indices = jnp.tile(indices, values.shape[1])
    flat_values = values.T.flatten()
    return tiled_array.at[row_indices, column_indices].set(flat_values)


def simulate(
    # prepared arrays
    optimized_parameters: jnp.ndarray,
    optimized_parameter_indices: jnp.ndarray,
    optimized_value_indices: jnp.ndarray,
    carrier_changing_parameter_indices: jnp.ndarray,
    carrier_changing_values: jnp.ndarray,
    signal_changing_parameter_indices: jnp.ndarray,
    signal_changing_values: jnp.ndarray,
    # matrices
    parameters: jnp.ndarray,
    carrier_matrix: jnp.ndarray,
    signal_matrix: jnp.ndarray,
    noise_matrix: jnp.ndarray,
    noise_selection_vectors: jnp.ndarray,
    qhd_parameter_indices: jnp.ndarray,
    qhd_placing_indices: jnp.ndarray,
    linked_indices: jnp.ndarray,
    linking_function_indices: jnp.ndarray,
    indices_to_link: jnp.ndarray,
    # carrier arrays
    carrier_function_input_indices: jnp.ndarray,
    carrier_function_indices: jnp.ndarray,
    carrier_output_indices: tuple[jnp.ndarray, jnp.ndarray],
    carrier_matrix_indices: tuple[jnp.ndarray, jnp.ndarray],
    # signal arrays
    signal_function_input_indices: jnp.ndarray,
    signal_function_indices: jnp.ndarray,
    signal_output_indices: tuple[jnp.ndarray, jnp.ndarray],
    signal_matrix_indices: tuple[jnp.ndarray, jnp.ndarray],
    signal_carrier_indices: jnp.ndarray,
    signal_carrier_placing_indices: tuple[jnp.ndarray, jnp.ndarray],
    # noise arrays
    noise_function_input_indices: jnp.ndarray,
    noise_function_indices: jnp.ndarray,
    noise_output_indices: tuple[jnp.ndarray, jnp.ndarray],
    noise_system_matrix_indices: tuple[jnp.ndarray, jnp.ndarray],
    *,
    signal_sparsity=None,
    packed_signal=None,
    packed_noise=None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Run one full simulation pass: carrier, signal, and quantum noise.

    Parameters
    ----------
    optimized_parameters : jnp.ndarray
        Optimized parameter values provided by an optimizer or sampling routine,
        shape (OPV,) or (num_ranges, OPV) in outer vectorization contexts.
    optimized_parameter_indices : jnp.ndarray
        Indices into `parameters` for parameters that are driven by optimized values,
        shape (OP,).
    optimized_value_indices : jnp.ndarray
        Indices into `optimized_parameters` mapping each optimized parameter to
        its source value, shape (OP,). This supports shared optimized values.
    carrier_changing_parameter_indices : jnp.ndarray
        Indices into `parameters` that change for the carrier sweep, shape (CCP,).
    carrier_changing_values : jnp.ndarray
        Values for those carrier-changing parameters, shape (CCP, CV).
    signal_changing_parameter_indices : jnp.ndarray
        Indices into `parameters` that change for the signal sweep, shape (SCP,).
    signal_changing_values : jnp.ndarray
        Values for those signal-changing parameters, shape (SCP, SV).

    parameters : jnp.ndarray
        Base parameter array, shape (1, P). It is expanded internally.
    carrier_matrix : jnp.ndarray
        Base carrier augmented matrix, shape (CN, CN + 1).
    signal_matrix : jnp.ndarray
        Base signal augmented matrix, shape (SN, SN + 1).
    noise_matrix : jnp.ndarray
        Noise source covariance matrix, shape (NN, NN) or broadcastable for vmap.
    signal_sparsity : tuple of arrays or None
        Optional column-compressed signal pattern for the experimental CPU solver.
    packed_signal : PackedSignal or None
        Optional coefficient template and scatter slots for sparse assembly.
    packed_noise : PackedNoise or None
        Optional occupied covariance entries for direct noise contraction.
    noise_selection_vectors : jnp.ndarray
        Detector selection vectors, shape (D, 1, setup_size).
    qhd_parameter_indices : jnp.ndarray
        Indices of quadrature-homodyne-detector phase parameters in `parameters`,
        shape (Q,).
    qhd_placing_indices : jnp.ndarray
        Indices describing where detector phases are applied into
        `noise_selection_vectors`, shape (Q, 3). Each row contains:
        (detector_index, row_index, column_index).
    linked_indices : jnp.ndarray
        Parameter indices used as inputs to linking functions, shape (L,).
    linking_function_indices : jnp.ndarray
        Function selectors for linking functions, shape (L,).
    indices_to_link : jnp.ndarray
        Parameter indices to receive outputs of linking functions, shape (L,).

    carrier_function_input_indices, carrier_function_indices, carrier_output_indices,
    carrier_matrix_indices :
        Indexing and function-selection arrays used by `update` to fill the carrier matrix.

    signal_function_input_indices, signal_function_indices, signal_output_indices,
    signal_matrix_indices :
        Indexing and function-selection arrays used by `update` to fill the signal matrix.

    signal_carrier_indices : jnp.ndarray
        Indices selecting the carrier solution entries used to scale signal connectors.
    signal_carrier_placing_indices : tuple[jnp.ndarray, jnp.ndarray]
        Matrix indices locating the signal connector entries that must be scaled by carrier.

    noise_function_input_indices, noise_function_indices, noise_output_indices,
    noise_system_matrix_indices :
        Indexing and function-selection arrays used by `update` to fill the noise matrix.

    Returns
    -------
    carrier : jnp.ndarray
        Carrier solution, transposed to shape (CN, CV).
    signal : jnp.ndarray
        Signal solution, transposed to shape (SN, SV).
    noise : jnp.ndarray
        Quantum noise amplitude spectral density, shape (D, SV) after squeeze.

    Notes
    -----
    - Linking functions are applied before parameter expansion so they operate on a single row.
    - The signal path is always executed (even if effectively empty) to avoid conditionals
      that would force compilation of branches with empty shapes.
    - Conjugation and block manipulation follow the conventions used for upper and lower
      sidebands in frequency-domain interferometer models.
    """
    # Insert optimized parameter values into the base parameter row.
    parameters = parameters.at[[0], optimized_parameter_indices].set(
        optimized_parameters[optimized_value_indices]
    )

    # Apply parameter linking (derived parameters).
    inputs_to_link = parameters[0, linked_indices]
    outputs_to_link = vmap(lambda i, x: lax.switch(i, LINKING_FUNCTIONS, x))(
        linking_function_indices, inputs_to_link
    )
    parameters = parameters.at[[0], indices_to_link].set(outputs_to_link)

    # Quadrature-homodyne-detector phase factors (unit magnitude complex phase).
    qhd_phase_values = jnp.exp(1j * jnp.radians(parameters[0, qhd_parameter_indices]))

    # --------------------
    # Carrier solve
    # --------------------
    parameters = expand_parameters(parameters, carrier_changing_parameter_indices, carrier_changing_values)
    carrier_matrix = vmap(update, in_axes=(0, None, None, None, None, None))(
        parameters,
        carrier_matrix,
        carrier_function_input_indices,
        carrier_output_indices,
        carrier_matrix_indices,
        carrier_function_indices,
    )
    carrier = vmap(solve)(carrier_matrix[:, :, :-1], carrier_matrix[:, :, -1])

    # --------------------
    # Signal solve
    # --------------------
    parameters = expand_parameters(parameters, signal_changing_parameter_indices, signal_changing_values)
    carrier_size = carrier.shape[1]
    carrier_coupling = carrier[:, signal_carrier_indices.flatten()]
    if packed_signal is None:
        signal_matrix = vmap(update, in_axes=(0, None, None, None, None, None))(
            parameters, signal_matrix, signal_function_input_indices, signal_output_indices,
            signal_matrix_indices, signal_function_indices,
        )
        coupling = (slice(None), *signal_carrier_placing_indices)
        signal_matrix = signal_matrix.at[coupling].set(-signal_matrix[coupling] * carrier_coupling)
        # Conjugate the lower sideband and its couplings to the extra degrees of freedom.
        lower = (slice(None), slice(carrier_size, 2 * carrier_size), slice(carrier_size, 2 * carrier_size))
        extra = (slice(None), slice(2 * carrier_size, None), slice(None, carrier_size))
        lower_coupling = (slice(None), slice(carrier_size, 2 * carrier_size), slice(2 * carrier_size, None))
        signal_matrix = signal_matrix.at[lower].set(signal_matrix[lower].conj())
        signal_matrix = signal_matrix.at[extra].set(signal_matrix[extra].conj())
        signal_matrix = signal_matrix.at[lower_coupling].set(signal_matrix[lower_coupling].conj())
        signal_rhs = signal_matrix[:, :, -1]
    else:
        updates = vmap(component_coefficients, in_axes=(0, None, None, None))(
            parameters, signal_function_input_indices, signal_output_indices, signal_function_indices,
        )
        coefficients = jnp.broadcast_to(packed_signal.template, (parameters.shape[0], packed_signal.template.size))
        coefficients = coefficients.at[:, packed_signal.update_slots].set(updates)
        coupling = (slice(None), packed_signal.coupling_slots)
        coefficients = coefficients.at[coupling].set(-coefficients[coupling] * carrier_coupling)
        coefficients = jnp.where(packed_signal.conjugate, coefficients.conj(), coefficients)
        nnz = signal_sparsity[0].size
        signal_data, signal_rhs = coefficients[:, :nnz], coefficients[:, nnz:]

    # --------------------
    # Quantum noise solve
    # --------------------
    # Apply quadrature-homodyne-detector phases into the selection vectors.
    # `lax.scatter_mul` is used (rather than `.at[...].set`) so empty index arrays
    # remain valid without conditionals.
    noise_selection_vectors = lax.scatter_mul(
        noise_selection_vectors,
        qhd_placing_indices.reshape(-1, 3),
        qhd_phase_values,
        lax.ScatterDimensionNumbers(
            update_window_dims=(),
            inserted_window_dims=(0, 1, 2),
            scatter_dims_to_operand_dims=(0, 1, 2),
        ),
        unique_indices=True,
    )

    # The carrier multiplication broadcasts the optical selections across the sweep.
    upper = noise_selection_vectors[..., :carrier_size] * carrier
    lower = jnp.conjugate(noise_selection_vectors[..., carrier_size : 2 * carrier_size] * carrier)
    extra = noise_selection_vectors[..., 2 * carrier_size :]
    extra = jnp.broadcast_to(extra, (*upper.shape[:-1], extra.shape[-1]))
    carrier_selections = jnp.concatenate((upper, lower, extra), axis=-1)

    # Back-propagate detector weights by solving the transposed conjugated signal system.
    carrier_selections = jnp.broadcast_to(
        carrier_selections,
        (carrier_selections.shape[0], signal_rhs.shape[0], carrier_selections.shape[2]),
    )
    if signal_sparsity is None:
        system = signal_matrix[:, :, :-1]
        signal = vmap(solve)(system, signal_rhs)
        adjoint = jnp.swapaxes(system.conj(), -1, -2)
        noise_weights = vmap(vmap(solve), in_axes=(None, 0))(adjoint, carrier_selections)
    else:
        if packed_signal is None:
            rows, columns, _ = signal_sparsity
            signal_data = signal_matrix[:, rows, columns]
        signal, weights = solve_signal_and_adjoint(
            signal_data.astype(jnp.complex128),
            signal_sparsity,
            signal_rhs.astype(jnp.complex128),
            jnp.transpose(carrier_selections, (1, 2, 0)).astype(jnp.complex128),
        )
        noise_weights = jnp.transpose(weights, (2, 0, 1))

    # Since A^H u = d, the detector variance d^H A^-1 Q u reduces to u^H Q u.
    if packed_noise is None:
        noise_matrix = vmap(update, in_axes=(0, None, None, None, None, None))(
            parameters, noise_matrix, noise_function_input_indices, noise_output_indices,
            noise_system_matrix_indices, noise_function_indices,
        )
        noise_sources = jnp.einsum("bij,dbj->dbi", noise_matrix, noise_weights)
        variance_terms = noise_weights.conj() * noise_sources
    else:
        updates = vmap(component_coefficients, in_axes=(0, None, None, None))(
            parameters, noise_function_input_indices, noise_output_indices, noise_function_indices,
        )
        values = jnp.broadcast_to(packed_noise.template, (updates.shape[0], packed_noise.template.size))
        values = values.at[:, packed_noise.update_slots].set(updates)
        variance_terms = noise_weights[..., packed_noise.rows].conj() * (
            values * noise_weights[..., packed_noise.columns]
        )
    noise_variance = jnp.real(jnp.sum(variance_terms, axis=-1))
    noise = jnp.sqrt(2 * noise_variance * UNIT_VACUUM * H_PLANCK * F0 * 0.25)

    return carrier.T, signal.T, noise.squeeze()


def group_component_instructions(arrays):
    """Group calls by component function and restore the original scatter order."""
    function_inputs, functions, (output_rows, output_columns), *placement = arrays
    inputs = np.asarray(function_inputs)
    functions = np.asarray(functions)
    output_rows = np.asarray(output_rows)
    output_columns = np.asarray(output_columns)
    output_functions = functions[output_rows]
    groups = [None] * len(FUNCTIONS)
    output_positions = []
    for index in range(len(FUNCTIONS)):
        positions = np.flatnonzero(output_functions == index)
        if not positions.size:
            continue
        calls = np.unique(output_rows[positions])
        groups[index] = (
            jnp.asarray(inputs[calls]),
            jnp.asarray(np.searchsorted(calls, output_rows[positions])),
            jnp.asarray(output_columns[positions]),
        )
        output_positions.append(positions)

    permutation = jnp.asarray(np.argsort(np.concatenate(output_positions)))
    return None, tuple(groups), permutation, *placement


def run_build_step(
    setup: Setup,
    changing_pairs: list | None = None,
    changing_values: jnp.ndarray | None = None,
    optimization_pairs: list | None = None,
    timeit: bool = False,
    *,
    signal_solver: str = "dense",
    pack_noise: bool = False,
) -> tuple:
    """
    Build a `Setup` into simulation-ready arrays and matrices.

    Parameters
    ----------
    setup : Setup
        The optical setup to build.
    changing_pairs : list | None
        Each element is a (component_name, parameter_name) tuple describing which
        parameters are swept during simulation. If None, no sweep is applied.
    changing_values : jnp.ndarray | None
        Values for the parameters described by `changing_pairs`.
        Expected shape is (num_changing_parameters, num_values).
    optimization_pairs : list | None
        Each element is a (component_name, parameter_name) tuple describing which
        parameters are treated as optimized inputs (external values injected into
        the parameter vector before solving). If None, no optimized parameters exist.
    timeit : bool
        If True, return an additional timing measurement (seconds) for the build.
    signal_solver : str
        "dense" uses the reference solver. "sparse" gathers entries from the dense
        assembly and factors a sparse system on the CPU. "packed" also assembles
        the frequency batch directly in sparse storage.
    pack_noise : bool
        Evaluate the noise quadratic form using only occupied covariance entries.

    Returns
    -------
    If `timeit` is False
        simulation_arrays : dict[str, jnp.ndarray]
            Mapping from `simulate` argument name to the array value.
        metadata : tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
            (detector_indices, mirror_indices, beamsplitter_indices, isolator_indices)

    If `timeit` is True
        build_seconds : float
            Time spent building and preparing arrays.
        (simulation_arrays, metadata) : tuple
            The same outputs as the non-timed case, packed into a tuple.

    Notes
    -----
    The returned dictionary keys are derived from the `simulate` function signature.
    """
    if signal_solver not in ("dense", "sparse", "packed"):
        raise ValueError(f"Unknown signal solver: {signal_solver}")
    start = time()

    instructions, matrices, metadata = build(setup)
    arrays_to_prepare, carrier_arrays, signal_arrays, noise_arrays = pairs_to_arrays(
        *instructions, optimization_pairs, changing_pairs
    )
    carrier_arrays = group_component_instructions(carrier_arrays)
    signal_arrays = group_component_instructions(signal_arrays)
    noise_arrays = group_component_instructions(noise_arrays)
    prepared_arrays = prepare_arrays(*arrays_to_prepare, changing_values, parameters=matrices[0])

    arg_count = simulate.__code__.co_argcount
    arg_names = simulate.__code__.co_varnames[:arg_count]

    all_arrays = (*prepared_arrays, *matrices, *carrier_arrays, *signal_arrays, *noise_arrays)
    if len(arg_names) != len(all_arrays):
        raise ValueError(
            "simulate signature does not match the number of prepared arrays and matrices. "
            f"simulate expects {len(arg_names)} arguments, but build produced {len(all_arrays)} arrays."
        )

    simulation_arrays = {arg_names[i]: array for i, array in enumerate(all_arrays)}
    if signal_solver != "dense":
        template, updates, coupling = matrices[2], signal_arrays[3], signal_arrays[5]
        pattern = build_signal_pattern(template, updates, coupling)
        simulation_arrays["signal_sparsity"] = pattern
        if signal_solver == "packed":
            simulation_arrays["packed_signal"] = pack_signal_template(
                template, pattern, updates, coupling, matrices[1].shape[0],
            )
    if pack_noise:
        simulation_arrays["packed_noise"] = pack_noise_template(matrices[3], noise_arrays[3])

    if timeit:
        for array in simulation_arrays.values():
            jax.block_until_ready(array)
        return time() - start, (simulation_arrays, metadata)

    return simulation_arrays, *metadata


def run_simulation_step(
    simulation_arrays: dict,
    jit_simulation: bool = False,
    timeit: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | tuple[float, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """
    Run `simulate` using a pre-built set of arrays.

    Parameters
    ----------
    simulation_arrays : dict[str, jnp.ndarray]
        Dictionary mapping argument names to arrays. This is the first return value
        of `run_build_step`.
    jit_simulation : bool
        If True, compile `simulate` using `jax.jit` and run one warmup call to
        trigger compilation before measuring runtime.
    timeit : bool
        If True, return the simulation runtime (seconds) alongside the results.

    Returns
    -------
    If `timeit` is False
        carrier : jnp.ndarray
        signal : jnp.ndarray
        noise : jnp.ndarray

    If `timeit` is True
        simulation_seconds : float
            Time spent executing the simulation (excluding warmup compilation).
        results : tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
            (carrier, signal, noise)

    Notes
    -----
    The warmup run uses the same inputs and forces compilation when `jit_simulation`
    is enabled.
    """
    simulation_function = simulate

    if jit_simulation:
        simulation_function = jit(simulate)
        _ = simulation_function(**simulation_arrays)  # warmup compilation

    start = time()
    results = simulation_function(**simulation_arrays)

    if timeit:
        for result in results:
            jax.block_until_ready(result)
        return time() - start, results

    return results


def run(
    setup: Setup,
    changing_pairs: list | None = None,
    changing_values: jnp.ndarray | None = None,
    optimization_pairs: list | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Convenience wrapper: build a setup and immediately run the simulation.

    Parameters
    ----------
    setup : Setup
        The setup to simulate.
    changing_pairs : list | None
        Sweep definition as (component_name, parameter_name) pairs.
    changing_values : jnp.ndarray | None
        Sweep values, shape (num_changing_parameters, num_values).
    optimization_pairs : list | None
        Optimized parameter definition as (component_name, parameter_name) pairs.

    Returns
    -------
    carrier : jnp.ndarray
        Carrier solution, shape (num_ports_carrier, num_carrier_values).
    signal : jnp.ndarray
        Signal solution, shape (num_ports_signal, num_signal_values).
    noise : jnp.ndarray
        Noise amplitude spectral density, shape (num_detectors, num_signal_values) after squeeze.
    detector_indices : jnp.ndarray
    mirror_indices : jnp.ndarray
    beamsplitter_indices : jnp.ndarray
    isolator_indices : jnp.ndarray
        Port indices in the order defined by the setup.
    """
    simulation_arrays, *metadata = run_build_step(setup, changing_pairs, changing_values, optimization_pairs)
    carrier, signal, noise = run_simulation_step(simulation_arrays)
    return carrier, signal, noise, *metadata


def run_setups(setups: list[Setup], frequencies: jnp.ndarray):
    """
    Run a list of setups over a shared frequency sweep.

    Parameters
    ----------
    setups : list[Setup]
        Setups to simulate.
    frequencies : jnp.ndarray
        Frequency values. This is passed as the sweep values for the pair ("f", "frequency").
        Expected shape is (1, num_frequencies) or broadcastable to the builder expectations.

    Returns
    -------
    results : list
        Each element is the return value of `run(setup, [("f", "frequency")], frequencies)`.
    """
    results = []
    for setup in setups:
        results.append(run(setup, [("f", "frequency")], frequencies))
    return results
