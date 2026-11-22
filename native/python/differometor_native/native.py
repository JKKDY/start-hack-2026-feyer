"""Native-facing API (new; not part of upstream Differometor).

    plan = build_plan(setup, [("f", "frequency")], frequencies, optimization_pairs)
    sim = Simulator(plan)
    carrier, signal, noise = sim.forward(x)                  # no tape
    (carrier, signal, noise), state = sim.forward_with_state(x)
    grad = sim.backward(state, carrier=gc, signal=gs, noise=gn)

Ownership: SetupSpec and SimulationPlan are immutable and shared; a
Simulator keeps its plan alive and a SimulationState keeps its plan alive,
so Python's garbage-collection order never matters. A Simulator is not
thread-safe. Only the newest state recorded on a thread can be used for
`backward`; an older one raises RuntimeError.

Complex cotangents use zbar = dL/dRe(z) + 1j * dL/dIm(z). JAX's convention is
the complex conjugate of that.
"""

from __future__ import annotations

from typing import Any

from differometor_native import _native
from differometor_native._convert import (
    normalize_changing_values,
    normalize_optimization,
    normalize_pairs,
    setup_spec,
)

SetupSpec = _native.core.SetupSpec
SimulationPlan = _native.core.SimulationPlan
SimulationState = _native.core.SimulationState
Simulator = _native.core.Simulator


def build_plan(
    setup_or_data: Any,
    changing_pairs: list | None = None,
    changing_values: Any = None,
    optimization_pairs: list | None = None,
    *,
    signal_solver: str = "packed",
    pack_noise: bool | None = None,
) -> SimulationPlan:
    """Snapshot the setup and build an immutable native plan."""
    pairs = normalize_pairs(changing_pairs)
    return _native.core.build_plan(
        setup_spec(setup_or_data),
        pairs,
        normalize_changing_values(changing_values, len(pairs)),
        normalize_optimization(optimization_pairs),
        signal_solver,
        pack_noise,
    )


__all__ = [
    "SetupSpec",
    "SimulationPlan",
    "SimulationState",
    "Simulator",
    "build_plan",
    "setup_spec",
]
