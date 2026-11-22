"""Native planner and matrix assembly against the original Python/JAX code.

Reference: verbatim upstream build.py / simulate.py / setups.py
(tests/reference/differometor_upstream). For each configuration the same
setup is built by upstream and by the native planner, then compared:

- metadata (detector / mirror / beamsplitter / isolator indices, port_to_index);
- parameter vector, names-derived mappings (optimised, swept, linked, qhd);
- instruction arrays (carrier and signal exactly; noise as a set, because
  upstream orders unused-port noise sources by Python set iteration);
- build-time template matrices;
- assembled carrier / signal / noise matrices for several frequencies and
  parameter values, including sparsity (identical non-zero positions);
- reverse-mode gradients of a random complex linear functional of the
  assembled matrices with respect to the optimised parameters and, for the
  signal matrix, the externally supplied carrier fields.

The same compiled extension handles every configuration.
"""

import copy
import json
import os
import pathlib

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from jax import lax, vmap  # noqa: E402

from differometor_native import _native as native_ext  # noqa: E402
from differometor_native import native  # noqa: E402
from differometor_native import setups as nat_setups  # noqa: E402
from differometor_upstream import build as up_build  # noqa: E402
from differometor_upstream import setups as up_setups  # noqa: E402
from differometor_upstream import simulate as up_simulate  # noqa: E402

PROPERTIES = ["reflectivity", "tuning", "db", "angle", "power", "mass", "length"]
FREQUENCIES = np.array([[150.0, 900.0, 2700.0]])


def _examples_dir():
    env = os.environ.get("DFM_EXAMPLES")
    if env:
        return pathlib.Path(env)
    return pathlib.Path(__file__).resolve().parents[3] / "examples" / "data"


# ------------------------------------------------------------- configurations
def cavity(module, signal=False):
    S = module.Setup()
    if signal:
        S.add("frequency", "f", frequency=1)
    S.add("laser", "l0", power=1)
    S.add("mirror", "m0", reflectivity=0.99, loss=0)
    S.add("mirror", "m1", reflectivity=0.991, loss=0)
    S.space("l0", "m0", length=1)
    S.space("m0", "m1", length=1)
    S.add("detector", "refl", target="m0", port="left", direction="out")
    S.add("detector", "circ", target="m1", port="left", direction="in")
    S.add("detector", "trns", target="m1", port="right", direction="out")
    if signal:
        S.add("free_mass", "fm1", target="m1", mass=10)
        S.add("signal", "sig", target="m0_m1")
        S.add("qnoised", "qn", target="m1", port="right", direction="out")
    return S


def uifo_fixture(module):
    path = _examples_dir() / "uifo_800_3000.json"
    if not path.exists():
        pytest.skip(f"UIFO fixture not found at {path} (set DFM_EXAMPLES)")
    data = json.loads(path.read_text())
    return module.Setup.from_data(data)


def first_values(setup, pairs):
    data = setup.to_data()
    values = []
    for entry in pairs:
        name, prop = (entry[0] if isinstance(entry[0], (list, tuple)) and not isinstance(entry[0], str) else entry)
        source = data["nodes"] if name in data["nodes"] else data["edges"]
        values.append(float(source[name]["properties"][prop]))
    return np.array(values)


CONFIGS = {
    "cavity_carrier_sweep": dict(
        make=lambda m: cavity(m),
        changing=([("m1", "tuning")], np.linspace(-20.0, 20.0, 3)),
        optimization=lambda s: [("m0", "tuning"), ("m1", "reflectivity"), ("l0", "power"), ("l0_m0", "length")],
    ),
    "cavity_signal": dict(
        make=lambda m: cavity(m, signal=True),
        changing=([("f", "frequency")], FREQUENCIES),
        optimization=lambda s: [("m0", "tuning"), ("m1", "reflectivity"), ("fm1", "mass"), ("m0_m1", "length")],
    ),
    "aligo": dict(
        make=lambda m: m.aligo()[0],
        changing=([("f", "frequency")], FREQUENCIES),
        optimization=lambda s: [p for p in s.parameters if p[1] in PROPERTIES][::3],
    ),
    "voyager_space": dict(
        make=lambda m: m.voyager("space_modulation")[0],
        changing=([("f", "frequency")], FREQUENCIES),
        optimization=lambda s: [p for p in s.parameters if p[1] in PROPERTIES][::2],
    ),
    "voyager_amplitude": dict(  # parameter links (sqrt of laser power)
        make=lambda m: m.voyager("amplitude_modulation")[0],
        changing=([("f", "frequency")], FREQUENCIES),
        optimization=lambda s: [p for p in s.parameters if p[1] in PROPERTIES][1::2],
    ),
    "voyager_frequency": dict(
        make=lambda m: m.voyager("frequency_modulation")[0],
        changing=([("f", "frequency")], FREQUENCIES),
        optimization=lambda s: [p for p in s.parameters if p[1] in PROPERTIES][::4],
    ),
    "uifo_2x2_generated": dict(
        make=lambda m: m.uifo(2, random_seed=3)[0],
        changing=([("f", "frequency")], FREQUENCIES),
        optimization=lambda s: m_constrain(s),
    ),
    "uifo_3x3_fixture": dict(
        make=uifo_fixture,
        changing=([("f", "frequency")], FREQUENCIES),
        optimization=lambda s: m_constrain(s),
    ),
}


def m_constrain(setup):
    return up_setups.constrain_inter_grid_cell_spaces(setup.parameters, PROPERTIES)


# ------------------------------------------------------------ reference code
def upstream_build(name):
    cfg = CONFIGS[name]
    setup = cfg["make"](up_setups)
    pairs, values = cfg["changing"]
    opt = cfg["optimization"](setup)
    arrays, *metadata = up_simulate.run_build_step(copy.deepcopy(setup), pairs, jnp.asarray(values), opt)
    return setup, arrays, metadata, opt


def native_plan(name):
    cfg = CONFIGS[name]
    setup = cfg["make"](nat_setups)
    pairs, values = cfg["changing"]
    opt = cfg["optimization"](setup)
    plan = native.build_plan(setup, pairs, values, opt)
    return setup, plan, opt


def upstream_assembly(arrays, x, carrier):
    """Matrix-building part of upstream `simulate`, with carrier fields given."""
    a = arrays
    parameters = a["parameters"].at[[0], a["optimized_parameter_indices"]].set(x[a["optimized_value_indices"]])
    inputs_to_link = parameters[0, a["linked_indices"]]
    outputs_to_link = vmap(lambda i, v: lax.switch(i, up_build.LINKING_FUNCTIONS, v))(
        a["linking_function_indices"], inputs_to_link)
    parameters = parameters.at[[0], a["indices_to_link"]].set(outputs_to_link)
    parameters = up_simulate.expand_parameters(parameters, a["carrier_changing_parameter_indices"],
                                               a["carrier_changing_values"])
    carrier_matrix = vmap(up_simulate.update, in_axes=(0, None, None, None, None, None))(
        parameters, a["carrier_matrix"], a["carrier_function_input_indices"], a["carrier_output_indices"],
        a["carrier_matrix_indices"], a["carrier_function_indices"])
    parameters = up_simulate.expand_parameters(parameters, a["signal_changing_parameter_indices"],
                                               a["signal_changing_values"])
    signal_matrix = vmap(up_simulate.update, in_axes=(0, None, None, None, None, None))(
        parameters, a["signal_matrix"], a["signal_function_input_indices"], a["signal_output_indices"],
        a["signal_matrix_indices"], a["signal_function_indices"])
    place = a["signal_carrier_placing_indices"]
    entries = -signal_matrix[:, place[0], place[1]] * carrier[:, a["signal_carrier_indices"].flatten()]
    signal_matrix = signal_matrix.at[:, place[0], place[1]].set(entries)
    s = carrier.shape[1]
    signal_matrix = signal_matrix.at[:, s:2 * s, s:2 * s].set(jnp.conjugate(signal_matrix[:, s:2 * s, s:2 * s]))
    signal_matrix = signal_matrix.at[:, 2 * s:, :s].set(jnp.conjugate(signal_matrix[:, 2 * s:, :s]))
    signal_matrix = signal_matrix.at[:, s:2 * s, 2 * s:].set(jnp.conjugate(signal_matrix[:, s:2 * s, 2 * s:]))
    noise_matrix = vmap(up_simulate.update, in_axes=(0, None, None, None, None, None))(
        parameters, a["noise_matrix"], a["noise_function_input_indices"], a["noise_output_indices"],
        a["noise_system_matrix_indices"], a["noise_function_indices"])
    return carrier_matrix, signal_matrix, noise_matrix


def dense(indptr, indices, values, rhs):
    """Native CSC batch -> dense (batch, n, n [+1])."""
    n = len(indptr) - 1
    cols = np.repeat(np.arange(n), np.diff(indptr))
    out = np.zeros((values.shape[0], n, n + (1 if rhs is not None else 0)), dtype=np.complex128)
    out[:, indices, cols] = values
    if rhs is not None:
        out[:, :, n] = rhs
    return out


# --------------------------------------------------------------------- tests
@pytest.fixture(scope="module", params=list(CONFIGS))
def pair(request):
    up_setup, arrays, metadata, up_opt = upstream_build(request.param)
    nat_setup, plan, nat_opt = native_plan(request.param)
    assert [tuple(map(tuple, p)) if not isinstance(p[0], str) else tuple(p) for p in up_opt] == \
           [tuple(map(tuple, p)) if not isinstance(p[0], str) else tuple(p) for p in nat_opt]
    x0 = first_values(nat_setup, nat_opt) if nat_opt else np.zeros(1)
    x = x0 * (1.0 + 0.01 * np.sin(np.arange(len(x0)) + 1.0))  # move away from the build values
    return request.param, arrays, metadata, plan, x


def test_metadata_matches(pair):
    name, arrays, metadata, plan, _ = pair
    det, mir, bs, iso, port_to_index = metadata
    nat = plan.metadata
    for up, mine in zip((det, mir, bs, iso), nat[:4]):
        np.testing.assert_array_equal(np.asarray(up), mine)
    assert dict(port_to_index) == nat[4]
    assert list(port_to_index) == list(nat[4])  # same order


def test_parameters_and_mappings_match(pair):
    name, up, _, plan, _ = pair
    mine = plan.export_simulation_arrays()
    np.testing.assert_array_equal(np.asarray(up["parameters"]).real, mine["parameters"].real)
    assert np.all(np.asarray(up["parameters"]).imag == 0)
    for key in ("optimized_parameter_indices", "optimized_value_indices", "carrier_changing_parameter_indices",
                "signal_changing_parameter_indices", "qhd_parameter_indices", "linked_indices", "indices_to_link"):
        np.testing.assert_array_equal(np.asarray(up[key]), mine[key], err_msg=key)
    for key in ("carrier_changing_values", "signal_changing_values"):
        np.testing.assert_allclose(np.asarray(up[key]).real, mine[key], err_msg=key)
    np.testing.assert_array_equal(np.asarray(up["qhd_placing_indices"]).reshape(-1, 3) if np.size(up["qhd_placing_indices"])
                                  else np.zeros((0, 3)), np.asarray(mine["qhd_placing_indices"]).reshape(-1, 3))
    np.testing.assert_allclose(np.asarray(up["noise_selection_vectors"]), mine["noise_selection_vectors"])
    # Link functions: upstream indexes a process-global callable list, the
    # native plan stores link-function ids. Compare what they compute.
    names = native_ext.link_functions()
    numpy_of = {"identity": lambda v: v, "sqrt": np.sqrt, "square": np.square, "exp": np.exp, "log": np.log,
                "sin": np.sin, "cos": np.cos, "negative": np.negative, "abs": np.abs}
    for up_index, native_index in zip(np.asarray(up["linking_function_indices"]), mine["linking_function_indices"]):
        upstream_fn = up_build.LINKING_FUNCTIONS[int(up_index)]
        native_fn = numpy_of[names[int(native_index)]]
        assert np.isclose(complex(upstream_fn(jnp.asarray(2.5))), native_fn(2.5))


def test_instruction_arrays_match(pair):
    name, up, _, plan, _ = pair
    mine = plan.export_simulation_arrays()
    for prefix in ("carrier", "signal"):
        for key in ("function_input_indices", "function_indices", "output_indices", "matrix_indices"):
            np.testing.assert_array_equal(np.asarray(up[f"{prefix}_{key}"]), mine[f"{prefix}_{key}"],
                                          err_msg=f"{prefix}_{key}")
    np.testing.assert_array_equal(np.asarray(up["signal_carrier_indices"]), mine["signal_carrier_indices"])
    np.testing.assert_array_equal(np.asarray(up["signal_carrier_placing_indices"]),
                                  mine["signal_carrier_placing_indices"])

    def noise_set(arrays):
        inputs = np.asarray(arrays["noise_function_input_indices"])
        functions = np.asarray(arrays["noise_function_indices"])
        out = np.asarray(arrays["noise_output_indices"])
        pos = np.asarray(arrays["noise_system_matrix_indices"])
        return sorted((int(functions[r]), tuple(inputs[r]), int(c), int(i), int(j))
                      for r, c, i, j in zip(out[0], out[1], pos[0], pos[1]))

    assert noise_set(up) == noise_set(mine)


def test_template_matrices_match(pair):
    name, up, _, plan, _ = pair
    mine = plan.export_simulation_arrays()
    for key in ("carrier_matrix", "signal_matrix"):
        ref = np.asarray(up[key])
        assert ref.shape == mine[key].shape, key
        np.testing.assert_allclose(mine[key], ref, rtol=0, atol=1e-13 * max(1.0, np.abs(ref).max()), err_msg=key)


def _carrier_fields(plan, seed):
    d = plan.dimensions
    rng = np.random.default_rng(seed)
    return rng.normal(size=(d["carrier_values"], d["carrier_ports"])) + 1j * rng.normal(
        size=(d["carrier_values"], d["carrier_ports"]))


@pytest.mark.parametrize("scale", [1.0, 0.97])
def test_assembled_matrices_match(pair, scale):
    name, up, _, plan, x = pair
    x = x * scale
    carrier = _carrier_fields(plan, 1)
    ref_c, ref_s, ref_n = upstream_assembly(up, jnp.asarray(x), jnp.asarray(carrier))
    for kind, ref, with_rhs in ((0, ref_c, True), (1, ref_s, True), (2, ref_n, False)):
        indptr, indices, values, rhs = plan.assemble(kind, x, carrier if kind == 1 else None)
        mine = dense(indptr, indices, values, rhs if with_rhs else None)
        ref = np.asarray(ref)
        assert mine.shape == ref.shape, (name, kind)
        scale_ref = max(1.0, np.abs(ref).max())
        np.testing.assert_allclose(mine, ref, rtol=0, atol=1e-12 * scale_ref, err_msg=f"{name} kind {kind}")
        # Sparsity: the same entries are non-zero (beyond round-off).
        tol = 1e-12 * scale_ref
        np.testing.assert_array_equal(np.abs(mine) > tol, np.abs(ref) > tol, err_msg=f"{name} kind {kind} pattern")


def test_assembly_gradients_match(pair):
    name, up, _, plan, x = pair
    carrier = _carrier_fields(plan, 2)
    rng = np.random.default_rng(7)
    for kind in (0, 1, 2):
        indptr, indices, values, rhs = plan.assemble(kind, x, carrier if kind == 1 else None)
        n = len(indptr) - 1
        cols = np.repeat(np.arange(n), np.diff(indptr))
        # Cotangents scaled per entry so that every slot contributes O(1).
        vcot = (rng.normal(size=values.shape) + 1j * rng.normal(size=values.shape)) / np.maximum(1.0, np.abs(values))
        rcot = (rng.normal(size=rhs.shape) + 1j * rng.normal(size=rhs.shape)) / np.maximum(1.0, np.abs(rhs))
        if kind == 2:
            rcot[:] = 0  # the noise matrix has no right-hand side upstream
        grad_x, grad_carrier = plan.assembly_vjp(kind, x, carrier if kind == 1 else None, vcot, rcot)

        cdense = np.zeros((values.shape[0], n, n + 1), dtype=np.complex128)
        cdense[:, indices, cols] = vcot
        cdense[:, :, n] = rcot
        if kind == 2:
            cdense = cdense[:, :, :n]

        def loss(xv, cr, ci):
            mats = upstream_assembly(up, xv, cr + 1j * ci)
            return jnp.sum(jnp.real(jnp.conj(cdense) * mats[kind]))

        gx, gcr, gci = jax.grad(loss, argnums=(0, 1, 2))(jnp.asarray(x), jnp.asarray(carrier.real),
                                                         jnp.asarray(carrier.imag))
        gx = np.asarray(gx)
        np.testing.assert_allclose(grad_x, gx, rtol=0, atol=1e-9 * max(1.0, np.abs(gx).max()),
                                   err_msg=f"{name} kind {kind} d/dx")
        if kind == 1:
            gc = np.asarray(gcr) + 1j * np.asarray(gci)
            np.testing.assert_allclose(grad_carrier, gc, rtol=0, atol=1e-9 * max(1.0, np.abs(gc).max()),
                                       err_msg=f"{name} d/dcarrier")

