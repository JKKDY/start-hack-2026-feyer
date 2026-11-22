"""API-compatibility tests for differometor_native.

Checks names, signatures, defaults, setup round trips, error types,
detector semantics and the run/simulate result ordering. The numerical
engine is a later milestone: real planning stops at NotImplementedError, and
the result-ordering tests use a fake backend that returns arrays of the
right shapes (never presented as physical results).
"""

import importlib
import inspect
import types

import numpy as np
import pytest

import differometor_native as dn
from differometor_native import native
from differometor_native.setups import Setup, aligo, voyager

simulate_module = importlib.import_module("differometor_native.simulate")


# ------------------------------------------------------------ helpers
def shape_of(func):
    """Parameter names, kinds and defaults (annotations deliberately ignored)."""
    out = []
    for p in inspect.signature(func).parameters.values():
        default = p.default
        if default is not inspect.Parameter.empty and callable(default):
            default = default.__name__
        out.append((p.name, p.kind.name, default))
    return out


E = inspect.Parameter.empty
POS = "POSITIONAL_OR_KEYWORD"
KW = "KEYWORD_ONLY"


def tiny_setup():
    s = Setup()
    s.add("frequency", "f")
    s.add("laser", "l0", power=2.0)
    s.add("mirror", "m0", reflectivity=0.9)
    s.space("l0", "m0", length=10.0)
    s.add("detector", "d0", target="m0", port="right", direction="out")
    return s


# ---------------------------------------------------- names/signatures
def test_top_level_exports_match_upstream():
    upstream = {
        "Setup", "run", "run_build_step", "run_simulation_step", "simulate",
        "visualize_setup", "power_detector", "signal_detector",
    }
    assert upstream <= set(dir(dn))
    for module in ("setups", "simulate", "components", "utils"):
        importlib.import_module(f"differometor_native.{module}")


def test_simulate_module_signatures_match_upstream():
    common = [("setup", POS, E), ("changing_pairs", POS, None), ("changing_values", POS, None),
              ("optimization_pairs", POS, None)]
    assert shape_of(simulate_module.run) == common
    assert shape_of(simulate_module.run_build_step) == common + [
        ("timeit", POS, False), ("signal_solver", KW, "packed"), ("pack_noise", KW, None)]
    assert shape_of(simulate_module.run_simulation_step) == [
        ("simulation_arrays", POS, E), ("jit_simulation", POS, False), ("timeit", POS, False)]
    assert shape_of(simulate_module.run_setups) == [("setups", POS, E), ("frequencies", POS, E)]

    sim = shape_of(simulate_module.simulate)
    assert [name for name, kind, _ in sim if kind == POS] == list(simulate_module.SIMULATE_ARGUMENTS)
    assert len(simulate_module.SIMULATE_ARGUMENTS) == 31
    assert [(n, d) for n, k, d in sim if k == KW] == [
        ("signal_sparsity", None), ("packed_signal", None), ("packed_noise", None)]


def test_setup_and_utils_signatures_match_upstream():
    assert shape_of(Setup.add) == [
        ("self", POS, E), ("component", POS, E), ("name", POS, E), ("not_optimizable", POS, None),
        ("target", POS, None), ("port", POS, None), ("direction", POS, None), ("auxiliary", POS, None),
        ("detector1", POS, None), ("detector2", POS, None), ("properties", "VAR_KEYWORD", E)]
    assert shape_of(Setup.space) == [
        ("self", POS, E), ("source", POS, E), ("target", POS, E), ("not_optimizable", POS, None),
        ("source_port", POS, "right"), ("target_port", POS, "left"), ("properties", "VAR_KEYWORD", E)]
    from differometor_native import utils
    assert shape_of(utils.update_setup)[-1] == ("bounding_function", POS, "sigmoid_bounding")
    assert shape_of(utils.calculate_sensitivities)[-2:] == [
        ("homodyne", POS, True), ("transfer_floor", POS, 0.0)]
    assert shape_of(dn.power_detector) == [("solution", POS, E)]
    assert shape_of(dn.signal_detector) == [("carrier", POS, E), ("signal", POS, E)]


def test_component_registry_preserved():
    from differometor_native import components
    assert len(components.FUNCTIONS) == 19
    assert components.FUNCTIONS[1].__name__ == "laser"
    assert components.DEFAULT_PROPERTIES["mirror"] == {"loss": 5e-6, "reflectivity": 0.5, "tuning": 0.0}
    # Registered kernels are native now: laser(power=2, phase=0) = sqrt(4).
    np.testing.assert_allclose(components.laser(np.array([2.0, 0.0])), [2.0, 0, 0, 0])
    with pytest.raises(ValueError):
        components.laser(np.array([2.0 + 1j, 0.0]))


# --------------------------------------------------------------- setups
def test_setup_round_trip_and_views():
    s = tiny_setup()
    data = s.to_data()
    assert set(data) == {"parameters", "nodes", "edges"}
    assert "l0_m0" in data["edges"]
    r = Setup.from_data(data)
    assert list(r.nodes(data=False)) == list(s.nodes(data=False))
    assert list(r.edges(data=False)) == [("l0", "m0")]
    assert r.edges["l0_m0"]["properties"]["length"] == 10.0
    assert r.parameters == s.parameters


def test_setup_validation_errors_match_upstream():
    s = Setup()
    with pytest.raises(ValueError):
        s.add("laser", "bad_name")
    with pytest.raises(ValueError):
        s.add("unknown_component", "x")
    with pytest.raises(ValueError):
        s.space("a", "b")


@pytest.mark.parametrize("factory", [voyager, aligo])
def test_predefined_setups_convert_to_native(factory):
    setup = factory()[0]
    spec = native.setup_spec(setup)
    assert spec.num_nodes == len(list(setup.nodes(data=False)))
    assert spec.num_edges == len(list(setup.edges(data=False)))
    assert spec.parameters == [tuple(p) for p in setup.parameters]


def test_unsupported_link_callable_is_rejected():
    s = tiny_setup()
    s.nodes["l0"]["properties"]["power"] = ("m0_reflectivity", lambda x: x * 2)
    with pytest.raises(ValueError, match="Unsupported parameter-link"):
        native.setup_spec(s)


# ----------------------------------------------------- planner boundary
def test_build_plan_validates_and_builds():
    s = tiny_setup()
    freqs = np.linspace(10, 100, 4)
    with pytest.raises(ValueError):
        native.build_plan(s, [("nope", "frequency")], freqs)
    with pytest.raises(ValueError):
        native.build_plan(s, [("f", "frequency")], np.ones((2, 4)))
    plan = native.build_plan(s, [("f", "frequency")], freqs, [("l0", "power"), [("m0", "tuning"), ("m0", "loss")]])
    assert plan.dimensions["optimized_values"] == 2
    assert plan.dimensions["signal_values"] == 4
    # The full native pipeline runs: upstream's 8 return values.
    result = dn.run(s, [("f", "frequency")], freqs)
    assert len(result) == 8
    carrier, signal, noise, *_ = result
    assert carrier.shape == (plan.dimensions["carrier_ports"], 1)
    assert signal.shape == (plan.dimensions["signal_variables"], 4)
    assert np.all(np.isfinite(carrier)) and np.all(np.isfinite(signal))


def test_snapshot_is_independent_of_later_mutation():
    s = tiny_setup()
    spec = native.setup_spec(s)
    s.add("laser", "l1")
    assert spec.num_nodes == 4 and len(list(s.nodes(data=False))) == 5


# ------------------------------------------- result ordering (fake engine)
class _FakePlan:
    metadata = (np.array([3, 4]), np.array([0, 1, 2, 3]), np.array([], dtype=np.int64),
                np.array([], dtype=np.int64), {"m0.left.in": 0, "m0.left.out": 1})

    def export_simulation_arrays(self):
        arrays = {name: np.zeros(1) for name in simulate_module.SIMULATE_ARGUMENTS}
        arrays["optimized_parameters"] = np.array([1.0, 2.0])
        return arrays


class _FakeSimulator:
    def __init__(self, plan):
        self.plan = plan

    def forward(self, x):
        n_values = 4
        carrier = np.full((5, 1), complex(x.sum()))
        signal = np.zeros((12, n_values), dtype=np.complex128)
        noise = np.ones((1, n_values))
        return carrier, signal, noise


@pytest.fixture
def fake_backend(monkeypatch):
    backend = types.SimpleNamespace(build_plan=lambda *args: _FakePlan(), Simulator=_FakeSimulator)
    monkeypatch.setattr(simulate_module, "_backend", backend)


def test_run_returns_upstream_eight_tuple(fake_backend):
    # upstream `run` returns carrier, signal, noise and build()'s five metadata
    # items (its docstring lists seven; the code returns port_to_index too).
    result = dn.run(tiny_setup(), [("f", "frequency")], np.linspace(1, 2, 4), [("l0", "power")])
    assert len(result) == 8
    carrier, signal, noise, det, mir, bs, iso, port_to_index = result
    assert port_to_index == {"m0.left.in": 0, "m0.left.out": 1}
    assert carrier.shape == (5, 1) and carrier.dtype == np.complex128
    assert signal.shape == (12, 4) and signal.dtype == np.complex128
    assert noise.shape == (4,) and noise.dtype == np.float64  # squeezed like upstream
    np.testing.assert_array_equal(det, [3, 4])
    np.testing.assert_array_equal(mir, [0, 1, 2, 3])


def test_build_step_return_forms_and_optimized_parameter_replacement(fake_backend):
    arrays, *metadata = dn.run_build_step(tiny_setup(), [("f", "frequency")], np.ones(4))
    assert len(metadata) == 5
    assert set(arrays) == set(simulate_module.SIMULATE_ARGUMENTS)
    elapsed, (arrays2, metadata2) = dn.run_build_step(tiny_setup(), [("f", "frequency")], np.ones(4), timeit=True)
    assert elapsed >= 0 and len(metadata2) == 5

    # Upstream benchmark pattern: replace optimized_parameters, call simulate.
    arrays["optimized_parameters"] = np.array([10.0, 20.0])
    carrier, _, _ = dn.simulate(**arrays)
    assert carrier[0, 0] == 30.0
    elapsed, (carrier, signal, noise) = dn.run_simulation_step(arrays, jit_simulation=True, timeit=True)
    assert elapsed >= 0


def test_simulate_rejects_arrays_without_plan():
    arrays = {name: np.zeros(1) for name in simulate_module.SIMULATE_ARGUMENTS}
    with pytest.raises(NotImplementedError):
        dn.simulate(**arrays)


# ------------------------------------------------------------ detectors
def test_power_and_amplitude_detectors_match_formulas():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 4, 2)) + 1j * rng.normal(size=(3, 4, 2))
    np.testing.assert_allclose(dn.power_detector(x), 0.5 * np.abs(x) ** 2, rtol=1e-15)
    from differometor_native.components import amplitude_detector
    np.testing.assert_allclose(amplitude_detector(x), np.sqrt(0.5) * np.abs(x), rtol=1e-15)


@pytest.mark.parametrize(
    "carrier_shape,signal_shape",
    [((5,), (10,)), ((5, 1), (10, 7)), ((5, 7), (10, 7)), ((5, 7), (10, 1)), ((5, 1), (12, 3))],
)
def test_signal_detector_matches_upstream_expression(carrier_shape, signal_shape):
    rng = np.random.default_rng(1)
    c = rng.normal(size=carrier_shape) + 1j * rng.normal(size=carrier_shape)
    s = rng.normal(size=signal_shape) + 1j * rng.normal(size=signal_shape)
    n = c.shape[0]
    expected = np.conj(c) * s[:n] + c * s[n : 2 * n]
    np.testing.assert_allclose(dn.signal_detector(c, s), expected, rtol=1e-14)
