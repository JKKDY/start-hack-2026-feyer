"""Full native simulator against the unmodified upstream simulator.

References come from the repository's correctness harness (tests/cases.py,
tests/reference/*.npz, docs/correctness-harness.md). They were computed by the
unmodified JAX simulator at each case's nominal parameters and 4 fuzzed
parameter sets, and include carrier, signal, noise, sensitivity, powers, the
readout loss and its full gradient. The scored UIFO case has 193 parameters.

Tolerances are the harness's own: each quantity's round-off floor at each
point (from the stored 1e-15 probe evaluations, tests/reference/meta.json)
times 10, clamped. Signal
matrices are ill-conditioned, so the floors, not bit equality, set the bar.

Forward values come from the native simulator. Gradients come from the native
reverse pass (`simulate_vjp`); JAX is used only for the readout (sensitivity
and loss from simulator outputs), exactly as the harness and benchmark do.

Needs the repository checkout: set DFM_REPO to its root when the tests run
from a copy of native/.
"""

import importlib.util
import os
import pathlib
import sys

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

import differometor_native as dn  # noqa: E402

simulate_module = sys.modules["differometor_native.simulate"]

REPO = pathlib.Path(os.environ.get("DFM_REPO", pathlib.Path(__file__).resolve().parents[3]))
HARNESS = REPO / "tests"
if not (HARNESS / "cases.py").exists():
    pytest.skip(f"harness not found under {HARNESS} (set DFM_REPO)", allow_module_level=True)


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# tests/cases.py imports `differometor`; point it at the native package.
_saved = {k: sys.modules.get(k) for k in ("differometor", "differometor.setups")}
sys.modules["differometor"] = dn
sys.modules["differometor.setups"] = sys.modules["differometor_native.setups"]
try:
    cases = _load_module("harness_cases", HARNESS / "cases.py")
finally:
    for k, v in _saved.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v
harness = _load_module("harness_check_general", HARNESS / "check_general.py")
import json as _json  # noqa: E402

META = _json.loads((HARNESS / "reference" / "meta.json").read_text())


def _readout(ports, case):
    """The harness readout (tests/check_general.py), in JAX."""

    def readout(outputs):
        carrier, signal, noise = outputs
        powers = dn.power_detector(carrier) if not isinstance(carrier, jax.Array) else 0.5 * jnp.abs(carrier) ** 2
        nfreq = signal.shape[1]
        noise2 = noise.reshape(-1, nfreq)
        has_signal = noise2.shape[0] > 0 and ports.shape[0] > 0 and case["changing_pairs"][0][0] == "f"
        if has_signal:
            c, s = carrier, signal
            n = c.shape[0]
            sig = (jnp.conj(c) * s[:n] + c * s[n:2 * n])[ports]
            detected = sig[0] if sig.shape[0] == 1 else sig[0] - sig[1]
            sensitivity = noise2 / jnp.abs(detected)[None, :]
            loss = jnp.mean(jnp.log10(sensitivity))
        else:
            sensitivity = jnp.zeros((0, nfreq))
            loss = jnp.sum(powers[ports]) if ports.shape[0] else jnp.sum(powers)
        return loss, (sensitivity, powers)

    return readout


@pytest.fixture(scope="module", params=list(cases.CASES))
def case_results(request):
    name = request.param
    ref = np.load(HARNESS / "reference" / f"{name}.npz")
    case = cases.CASES[name]()
    arrays, ports, *_ = dn.run_build_step(case["setup"], case["changing_pairs"],
                                         np.asarray(case["changing_values"]), case["pairs"])
    ports = jnp.asarray(ports)
    readout = _readout(ports, case)
    rows = []
    for params in ref["parameters"]:
        outputs, pullback = simulate_module.simulate_vjp(arrays, params)
        jouts = tuple(jnp.asarray(o) for o in outputs)
        loss, read_pullback, (sensitivity, powers) = jax.vjp(readout, jouts, has_aux=True)
        output_cotangents = read_pullback(jnp.asarray(1.0))[0]
        (gradient,) = pullback(tuple(np.asarray(c) for c in output_cotangents))
        rows.append(dict(carrier=outputs[0], signal=outputs[1], noise=outputs[2],
                         sensitivity=np.asarray(sensitivity), powers=np.asarray(powers),
                         loss=float(loss), gradient=gradient))
    return name, ref, rows, arrays


QUANTITIES = ("carrier", "signal", "noise", "sensitivity", "powers", "loss", "gradient")


# A reference value whose round-off floor exceeds this relative level changes
# by more than 10% when the inputs move by 1e-15: it is set by round-off, not
# physics (e.g. voyager_fm at its nominal point, where the detected signal is
# ~1e-21, so log-sensitivity gradients have norm ~1e15). Such readout
# comparisons are reported, not asserted; the physical fields at those points
# (carrier, signal, noise, powers) are still asserted.
ROUNDOFF_DOMINATED_FLOOR = 0.1
ROUNDOFF_REPORT = []


@pytest.mark.parametrize("quantity", QUANTITIES)
def test_matches_unmodified_simulator(case_results, quantity):
    name, ref, rows, _ = case_results
    failures = []
    for i, mine in enumerate(rows):
        reference = harness.point(ref, i)
        floor = META["cases"][name]["floors"][i][quantity]
        error = harness.errors(mine, reference)[quantity]
        tol, ill = harness.tolerance(quantity, floor)
        if np.isfinite(error) and error <= tol:
            continue
        if ill and floor > ROUNDOFF_DOMINATED_FLOOR and quantity in ("gradient", "loss", "sensitivity"):
            ROUNDOFF_REPORT.append((name, i, quantity, error, floor))
            continue
        failures.append(f"point {i}: error {error:.3e} > tol {tol:.3e} (floor {floor:.3e}{', ill' if ill else ''})")
    assert not failures, f"{name} {quantity}: " + "; ".join(failures)


def test_zz_roundoff_dominated_points_are_rare():
    """Runs after the comparisons: only round-off-dominated references may be exempted."""
    for name, i, quantity, error, floor in ROUNDOFF_REPORT:
        print(f"round-off dominated reference: {name} point {i} {quantity}: error {error:.2e}, floor {floor:.2e}")
    assert len(ROUNDOFF_REPORT) <= 3, ROUNDOFF_REPORT


def test_outputs_are_finite_and_gradient_nonzero(case_results):
    name, _, rows, _ = case_results
    for mine in rows:
        for key, value in mine.items():
            assert np.all(np.isfinite(value)), (name, key)
        assert np.any(mine["gradient"] != 0), name


def test_signal_solution_residuals_are_small(case_results):
    """Physical solution quality: backward error of the native signal solves."""
    name, ref, rows, arrays = case_results
    plan = arrays.plan
    x = ref["parameters"][0]
    carrier = rows[0]["carrier"]  # (ports, carrier_values)
    indptr, indices, values, rhs = plan.assemble(1, np.ascontiguousarray(x), np.ascontiguousarray(carrier.T))
    n = len(indptr) - 1
    cols = np.repeat(np.arange(n), np.diff(indptr))
    signal = rows[0]["signal"]  # (n, batch)
    for b in range(values.shape[0]):
        y = signal[:, b]
        my = np.zeros(n, dtype=np.complex128)
        np.add.at(my, indices, values[b] * y[cols])
        norm_m = np.max(np.abs(values[b]))
        scale = norm_m * np.linalg.norm(y) + np.linalg.norm(rhs[b])
        residual = np.linalg.norm(my - rhs[b])
        if scale == 0:  # no signal source: the exact solution is zero
            assert residual == 0, (name, b)
            continue
        assert residual / scale < 1e-12, (name, b, residual / scale)


# --------------------------------------------------------- benchmark checks
def test_benchmark_fixture_sensitivity_and_finite_difference():
    """The benchmark's own criteria, on the native simulator."""
    import json
    data = json.loads((REPO / "examples" / "data" / "uifo_800_3000.json").read_text())
    reference = json.loads((REPO / "examples" / "data" / "uifo_800_3000_test_data.json").read_text())
    setup = dn.Setup.from_data(data)
    pairs = dn.setups.constrain_inter_grid_cell_spaces(setup.parameters, cases.PROPERTIES)
    x = cases._values(data, pairs)
    assert x.shape == (193,)
    frequencies = np.asarray(reference["frequencies"])
    expected = np.asarray(reference["sensitivities"])
    arrays, ports, *_ = dn.run_build_step(setup, [("f", "frequency")], frequencies, pairs)
    ports = jnp.asarray(ports)

    def readout(outputs):
        carrier, signal, noise = outputs
        n = carrier.shape[0]
        signals = (jnp.conj(carrier) * signal[:n] + carrier * signal[n:2 * n])[ports]
        detected = signals.squeeze() if ports.shape[0] == 1 else signals[0] - signals[1]
        sensitivity = (noise / jnp.abs(detected)).squeeze()
        return jnp.mean(jnp.log10(sensitivity / expected)), sensitivity

    outputs, pullback = simulate_module.simulate_vjp(arrays, x)
    (loss, sensitivity), read_pullback = jax.vjp(readout, tuple(jnp.asarray(o) for o in outputs))
    np.testing.assert_allclose(np.log(np.asarray(sensitivity)), np.log(expected), atol=1e-5, rtol=0)
    assert np.all(np.asarray(sensitivity) > 0)
    (gradient,) = pullback(tuple(np.asarray(c) for c in read_pullback((jnp.asarray(1.0), jnp.zeros_like(sensitivity)))[0]))
    assert gradient.shape == (193,) and np.all(np.isfinite(gradient)) and np.any(gradient != 0)

    # Central differences on mb11.tuning, steps 1e-3 and 5e-4, rtol 1% (benchmark rule).
    index = pairs.index(["mb11", "tuning"])

    def loss_at(p):
        arrays["optimized_parameters"] = p
        return float(readout(tuple(jnp.asarray(o) for o in dn.simulate(**arrays)))[0])

    for step in (1e-3, 5e-4):
        up, down = x.copy(), x.copy()
        up[index] += step
        down[index] -= step
        fd = (loss_at(up) - loss_at(down)) / (2 * step)
        np.testing.assert_allclose(gradient[index], fd, rtol=1e-2)


@pytest.mark.parametrize("name", ["cavity_tuning", "aligo_space", "voyager_space", "uifo3_scored"])
def test_run_api_matches_reference_at_build_values(name):
    """Upstream-style `run` (no optimisation pairs) on four geometries, one binary."""
    ref = np.load(HARNESS / "reference" / f"{name}.npz")
    case = cases.CASES[name]()
    result = dn.run(case["setup"], case["changing_pairs"], np.asarray(case["changing_values"]))
    assert len(result) == 8
    carrier, signal, noise, detector_indices, mirror_indices, beamsplitter_indices, isolator_indices, ports = result
    assert isinstance(ports, dict) and all(isinstance(v, int) for v in ports.values())
    assert carrier.shape == ref["carrier"][0].shape and carrier.dtype == np.complex128
    assert signal.shape == ref["signal"][0].shape and signal.dtype == np.complex128
    assert noise.shape == ref["noise"][0].shape and noise.dtype == np.float64
    # Build values are the nominal point (point 0) of the reference.
    errors = harness.errors(dict(carrier=carrier, signal=signal, noise=noise, sensitivity=ref["sensitivity"][0],
                                 powers=0.5 * np.abs(carrier) ** 2, loss=ref["loss"][0], gradient=ref["gradient"][0]),
                            harness.point(ref, 0))
    floors = META["cases"][name]["floors"][0]
    for quantity in ("carrier", "signal", "noise", "powers"):
        tol, _ = harness.tolerance(quantity, floors[quantity])
        assert errors[quantity] <= tol, (name, quantity, errors[quantity], tol)
