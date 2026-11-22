"""Packed CPU simulation and its explicit reverse pass."""
from functools import partial
from typing import NamedTuple

import jax
import numpy as np

from differometor._native_components import evaluate as evaluate_component
from differometor.components import F0, H_PLANCK, UNIT_VACUUM
from differometor.sparse_signal import _factor_and_solve

NOISE_SCALE = 2 * UNIT_VACUUM * H_PLANCK * F0 * 0.25


class State(NamedTuple):
    carrier: np.ndarray
    carrier_matrix: np.ndarray
    signal: np.ndarray
    coefficients: np.ndarray
    before_coupling: np.ndarray
    weights: np.ndarray
    covariance: np.ndarray
    noise: np.ndarray


def expand(parameters, indices, values):
    result = np.tile(parameters, (values.shape[1], 1))
    result[:, indices] = values.T
    return result


def unexpand(cotangent, indices, count):
    cotangent[:, indices] = 0
    return cotangent.reshape(-1, count, cotangent.shape[-1]).sum(axis=0)


def coefficient_call(parameters, indices, kind, seed=None):
    inputs = parameters[:, indices]
    if inputs.shape[-1] != 7:
        inputs = np.pad(inputs, [(0, 0), (0, 0), (0, 7 - inputs.shape[-1])])
    output = np.empty((*inputs.shape[:2], 4 if seed is None else 7), dtype=np.complex128)
    evaluate_component(kind, np.ascontiguousarray(inputs), seed, output)
    return output


def coefficients(parameters, instruction, cotangent=None, active_parameters=None):
    if callable(instruction):
        return instruction(parameters, cotangent)
    groups, permutation = instruction
    if cotangent is None:
        blocks = [
            coefficient_call(parameters, indices, kind)[:, rows, columns]
            for kind, group in enumerate(groups) if group is not None
            for indices, rows, columns in [group]
        ]
        return np.concatenate(blocks, axis=1)[:, permutation]

    result = np.zeros_like(parameters)
    ordered = np.empty_like(cotangent)
    ordered[:, permutation] = cotangent
    offset = 0
    for kind, group in enumerate(groups):
        if group is None:
            continue
        indices, rows, columns = group
        group_cotangent = ordered[:, offset:offset + len(rows)]
        offset += len(rows)
        active_calls = active_parameters[indices].any(axis=1)
        if not active_calls.any():
            continue
        keep = active_calls[rows]
        rows = (np.cumsum(active_calls) - 1)[rows[keep]]
        indices = indices[active_calls]
        seed = np.zeros((len(parameters), len(indices), 4), dtype=np.complex128)
        np.add.at(seed, (slice(None), rows, columns[keep]), group_cotangent[:, keep])
        gradient = coefficient_call(parameters, indices, kind, seed)
        np.add.at(result, (slice(None), indices), gradient[..., :indices.shape[1]])
    return result


def last_writes(indices):
    """Only the final write to a slot contributes to its input gradient."""
    _, reverse_indices = np.unique(indices[::-1], return_index=True)
    mask = np.zeros(len(indices), dtype=bool)
    mask[len(indices) - 1 - reverse_indices] = True
    return mask


class Plan(NamedTuple):
    active_parameters: np.ndarray
    carrier_parameters: np.ndarray
    signal_parameters: np.ndarray
    carrier_instruction: tuple
    signal_instruction: tuple
    noise_instruction: tuple
    carrier_slots: tuple
    signal_slots: np.ndarray
    coupling_slots: np.ndarray
    coupling_sources: np.ndarray
    conjugate: np.ndarray
    noise_slots: np.ndarray
    noise_rows: np.ndarray
    noise_columns: np.ndarray
    pattern: tuple
    phase_slots: np.ndarray


def detector_selection(plan, selection_template, phases, carrier):
    selection = selection_template.copy()
    slots = tuple(plan.phase_slots.reshape(-1, 3).T)
    selection[slots] *= phases
    n = carrier.shape[-1]
    upper = selection[..., :n] * carrier
    lower = (selection[..., n:2 * n] * carrier).conj()
    extra = np.broadcast_to(selection[..., 2 * n:], (*upper.shape[:-1], selection.shape[-1] - 2 * n))
    return selection, np.concatenate((upper, lower, extra), axis=-1)


def host_forward(plan, base, phases, carrier_sweep, signal_sweep,
                 carrier_template, signal_template, noise_template, selection_template):
    base, phases, carrier_sweep, signal_sweep, carrier_template, signal_template, noise_template, selection_template = map(
        np.asarray, (base, phases, carrier_sweep, signal_sweep, carrier_template, signal_template, noise_template, selection_template),
    )
    carrier_params = expand(base, plan.carrier_parameters, carrier_sweep)
    carrier_matrix = np.broadcast_to(carrier_template, (len(carrier_params), *carrier_template.shape)).copy()
    carrier_matrix[(slice(None), *plan.carrier_slots)] = coefficients(carrier_params, plan.carrier_instruction)
    carrier = np.linalg.solve(carrier_matrix[:, :, :-1], carrier_matrix[:, :, -1:])[..., 0]

    signal_params = expand(carrier_params, plan.signal_parameters, signal_sweep)
    packed = np.broadcast_to(signal_template, (len(signal_params), signal_template.size)).copy()
    packed[:, plan.signal_slots] = coefficients(signal_params, plan.signal_instruction)
    before_coupling = packed[:, plan.coupling_slots].copy()
    packed[:, plan.coupling_slots] = -before_coupling * carrier[:, plan.coupling_sources]
    packed[:, plan.conjugate] = packed[:, plan.conjugate].conj()
    _, selection = detector_selection(plan, selection_template, phases, carrier)
    d = np.broadcast_to(selection, (selection.shape[0], len(packed), selection.shape[-1])).transpose(1, 2, 0)
    rows, _, pointers = plan.pattern
    signal, weights = _factor_and_solve(packed[:, :len(rows)], rows, pointers, packed[:, len(rows):], d, transpose=False)

    covariance = np.broadcast_to(noise_template, (len(packed), noise_template.size)).copy()
    covariance[:, plan.noise_slots] = coefficients(signal_params, plan.noise_instruction)
    weights = weights.transpose(2, 0, 1)
    variance = np.sum(weights[..., plan.noise_rows].conj() * (covariance * weights[..., plan.noise_columns]), axis=-1).real
    noise = np.sqrt(NOISE_SCALE * variance)
    return State(carrier, carrier_matrix, signal, packed, before_coupling, weights, covariance, noise)


from functools import lru_cache

@lru_cache(maxsize=1)
def _covariance_kernel():
    import ctypes
    from .rust_kernel import installed_kernel
    path = installed_kernel()
    if path is None:
        return None
    try:
        kernel = ctypes.CDLL(str(path.with_name("sparse-" + path.name))).covariance_pullback
    except (OSError, AttributeError):
        return None
    kernel.argtypes = [ctypes.c_size_t] * 4 + [ctypes.c_void_p] * 7
    kernel.restype = None
    return kernel


def covariance_pullback(weights, covariance, variance_bar, rows, cols):
    nd, nf, width = weights.shape
    entries = len(rows)
    kernel = _covariance_kernel()
    if (kernel is not None and weights.dtype == covariance.dtype == np.complex128
            and variance_bar.dtype == np.float64 and variance_bar.shape == (nd, nf, 1)
            and covariance.shape == (nf, entries) and rows.shape == cols.shape == (entries,)
            and rows.dtype.kind in "iu" and cols.dtype.kind in "iu"
            and np.all((rows >= 0) & (rows < width)) and np.all((cols >= 0) & (cols < width))):
        inputs = (np.ascontiguousarray(rows, dtype=np.int64), np.ascontiguousarray(cols, dtype=np.int64),
                  np.ascontiguousarray(weights), np.ascontiguousarray(covariance),
                  np.ascontiguousarray(variance_bar))
        terms = np.empty((nd, nf, entries), dtype=np.complex128)
        weights_bar = np.zeros((nd, nf, width), dtype=np.complex128)
        kernel(nd, nf, width, entries, *(x.ctypes.data for x in (*inputs, terms, weights_bar)))
        return terms.sum(axis=0), weights_bar
    left, right = weights[..., rows], weights[..., cols]
    covariance_bar = np.sum(variance_bar * left.conj() * right, axis=0)
    weights_bar = np.zeros_like(weights)
    np.add.at(weights_bar, (slice(None), slice(None), rows), variance_bar * (covariance * right).conj())
    np.add.at(weights_bar, (slice(None), slice(None), cols), variance_bar * left.conj() * covariance)

    return covariance_bar, weights_bar


def host_reverse(plan, base, phases, carrier_sweep, signal_sweep, selection_template, state, seeds):
    base, phases, carrier_sweep, signal_sweep, selection_template = map(
        np.asarray, (base, phases, carrier_sweep, signal_sweep, selection_template),
    )
    carrier, carrier_matrix, signal, packed, before_coupling, weights, covariance, noise = map(np.asarray, state)
    carrier_bar, signal_bar, noise_bar = map(np.asarray, seeds)
    carrier_bar = carrier_bar.T.copy()
    variance_bar = (noise_bar.reshape(noise.shape) * (NOISE_SCALE / 2) / noise)[..., None]
    covariance_bar, weights_bar = covariance_pullback(
        weights, covariance, variance_bar, plan.noise_rows, plan.noise_columns)

    rows, cols, pointers = plan.pattern
    rhs_bar, d_bar = _factor_and_solve(
        packed[:, :len(rows)], rows, pointers, signal_bar.T, weights_bar.transpose(1, 2, 0), transpose=True,
    )
    packed_bar = np.concatenate((
        -rhs_bar[:, rows] * signal[:, cols]
        - np.sum(d_bar[:, cols, :] * weights.transpose(1, 2, 0)[:, rows, :], axis=2).conj(),
        rhs_bar,
    ), axis=1)
    packed_bar[:, plan.conjugate] = packed_bar[:, plan.conjugate].conj()
    coupling_bar = packed_bar[:, plan.coupling_slots]
    coupling_mask = last_writes(plan.coupling_slots)
    coupling_bar[:, ~coupling_mask] = 0
    carrier_coupling_bar = -coupling_bar * before_coupling
    np.add.at(carrier_bar, (slice(None), plan.coupling_sources),
              carrier_coupling_bar.reshape(len(packed) // len(carrier), len(carrier), -1).sum(axis=0))
    packed_bar[:, plan.coupling_slots[coupling_mask]] = (
        -coupling_bar[:, coupling_mask] * carrier[:, plan.coupling_sources[coupling_mask]]
    )

    selection, d = detector_selection(plan, selection_template, phases, carrier)
    d_bar = d_bar.transpose(2, 0, 1)
    if d.shape[1] == 1:
        d_bar = d_bar.sum(axis=1, keepdims=True)
    n = carrier.shape[-1]
    upper_bar, lower_bar = d_bar[..., :n], d_bar[..., n:2 * n].conj()
    carrier_bar += (upper_bar * selection[..., :n] + lower_bar * selection[..., n:2 * n]).sum(axis=0)
    selection_bar = np.concatenate((upper_bar * carrier, lower_bar * carrier, d_bar[..., 2 * n:]), axis=-1)
    if selection.shape[1] == 1:
        selection_bar = selection_bar.sum(axis=1, keepdims=True)
    slots = tuple(plan.phase_slots.reshape(-1, 3).T)
    phase_bar = selection_bar[slots] * selection_template[slots]

    carrier_params = expand(base, plan.carrier_parameters, carrier_sweep)
    signal_params = expand(carrier_params, plan.signal_parameters, signal_sweep)
    signal_seed = packed_bar[:, plan.signal_slots] * last_writes(plan.signal_slots)
    noise_seed = covariance_bar[:, plan.noise_slots] * last_writes(plan.noise_slots)
    parameter_bar = coefficients(signal_params, plan.signal_instruction, signal_seed, plan.active_parameters)
    parameter_bar += coefficients(signal_params, plan.noise_instruction, noise_seed, plan.active_parameters)
    carrier_parameter_bar = unexpand(parameter_bar, plan.signal_parameters, len(carrier))

    carrier_rhs_bar = np.linalg.solve(carrier_matrix[:, :, :-1].transpose(0, 2, 1), carrier_bar[..., None])[..., 0]
    r, c = plan.carrier_slots
    c = c % (n + 1)
    carrier_seed = carrier_rhs_bar[:, r].copy()
    columns = c < n
    carrier_seed[:, columns] *= -carrier[:, c[columns]]
    carrier_seed *= last_writes(r * (n + 1) + c)
    carrier_parameter_bar += coefficients(carrier_params, plan.carrier_instruction, carrier_seed, plan.active_parameters)
    return unexpand(carrier_parameter_bar, plan.carrier_parameters, 1), phase_bar


def run(plan, base, phases, carrier_sweep, signal_sweep, carrier_template,
        signal_template, noise_template, selection_template):
    from .rust_kernel import installed_kernel
    library = installed_kernel()
    if library:
        from ._rust_sweep import Sweep
        signal_varying = np.union1d(plan.carrier_parameters, plan.signal_parameters)
        if carrier_sweep.shape[1] > 1:
            # Separate cotangents belong to each carrier row before unexpand.
            signal_varying = np.arange(base.shape[-1])
        def prepare(instruction, varying):
            groups, _ = instruction
            if any(g is not None for g in groups) and all(
                g is None or g[0].shape[1] == 7 for g in groups
            ):
                return Sweep(instruction, varying, base.shape[-1], library)
            return instruction
        plan = plan._replace(
            carrier_instruction=prepare(plan.carrier_instruction, plan.carrier_parameters),
            signal_instruction=prepare(plan.signal_instruction, signal_varying),
            noise_instruction=prepare(plan.noise_instruction, signal_varying),
        )
    nc = carrier_sweep.shape[1]
    nf = nc * signal_sweep.shape[1]
    n = carrier_template.shape[0]
    nd = selection_template.shape[0]
    ns = selection_template.shape[-1]
    shapes = ((nc, n), (nc, n, n + 1), (nf, ns), (nf, signal_template.size),
              (nf, plan.coupling_slots.size), (nd, nf, ns), (nf, noise_template.size), (nd, nf))
    specs = State(*(jax.ShapeDtypeStruct(shape, np.float64 if i == 7 else np.complex128)
                    for i, shape in enumerate(shapes)))

    def evaluate(base, phases, cs, ss, cm, sm, nm, selections):
        return jax.pure_callback(
            partial(host_forward, plan), specs, base, phases, cs, ss, cm, sm, nm, selections,
            vmap_method="sequential",
        )

    def outputs(state):
        return state.carrier.T, state.signal.T, state.noise.squeeze()

    @jax.custom_vjp
    def simulate(base, phases, cs, ss, cm, sm, nm, selections):
        return outputs(evaluate(base, phases, cs, ss, cm, sm, nm, selections))

    def forward(base, phases, cs, ss, cm, sm, nm, selections):
        state = evaluate(base, phases, cs, ss, cm, sm, nm, selections)
        return outputs(state), (base, phases, cs, ss, selections, state)

    def reverse(residual, seeds):
        gradients = jax.pure_callback(
            partial(host_reverse, plan), (jax.ShapeDtypeStruct(base.shape, base.dtype),
                                         jax.ShapeDtypeStruct(phases.shape, phases.dtype)),
            *residual, seeds, vmap_method="sequential",
        )
        return (*gradients, None, None, None, None, None, None)

    simulate.defvjp(forward, reverse)
    return simulate(base, phases, carrier_sweep, signal_sweep, carrier_template,
                    signal_template, noise_template, selection_template)
