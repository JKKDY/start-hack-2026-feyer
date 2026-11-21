#!/usr/bin/env python3
"""Time one cold run of the public simulation, readout, and gradient."""

from time import perf_counter


def run():
    import json
    import os
    from pathlib import Path

    # The original benchmark configures the CPU before importing JAX.
    from benchmark import PROPERTIES, backward, forward_with_pullback, sensitivity_with_pullback

    import differometor as df
    import jax
    import jax.numpy as jnp
    from differometor.setups import constrain_inter_grid_cell_spaces

    jax.config.update("jax_enable_compilation_cache", False)
    jax.devices("cpu")

    data = Path(__file__).resolve().parent / "examples" / "data"
    fixture = json.loads((data / "uifo_800_3000.json").read_bytes())
    reference = json.loads((data / "uifo_800_3000_test_data.json").read_bytes())
    setup = df.Setup.from_data(fixture)
    pairs = constrain_inter_grid_cell_spaces(setup.parameters, PROPERTIES)
    initial = []
    for pair in pairs:
        members = pair if isinstance(pair[0], list) else [pair]
        values = [
            fixture["nodes" if name in fixture["nodes"] else "edges"][name]["properties"][prop]
            for name, prop in members
        ]
        if any(value != values[0] for value in values):
            raise ValueError("Grouped length parameters must agree in the public fixture")
        initial.append(values[0])

    parameters = jnp.asarray(initial, dtype=jnp.float64)
    frequencies = jnp.asarray(reference["frequencies"], dtype=jnp.float64)
    expected = jnp.asarray(reference["sensitivities"], dtype=jnp.float64)
    arrays, ports, *_ = df.run_build_step(setup, [("f", "frequency")], frequencies, pairs)
    arrays["optimized_parameters"] = parameters
    ports = jnp.asarray(ports)

    if os.environ.get("DIFFEROMETOR_NATIVE") == "1":
        from benchmark import readout

        def objective(values, inputs, detectors, target):
            optics = df.simulate(**dict(inputs, optimized_parameters=values))
            loss, (sensitivity, powers) = readout(optics, detectors, target)
            return loss, (optics, sensitivity, powers)

        arguments = parameters, arrays, ports, expected
        evaluate = jax.jit(jax.value_and_grad(objective, has_aux=True)).lower(*arguments).compile()
        (loss, (outputs, sensitivity, powers)), gradient = evaluate(*arguments)
        return jax.device_get((outputs, sensitivity, powers, loss, gradient))

    seed = jnp.asarray(1.0, dtype=jnp.float64)

    forward = jax.jit(forward_with_pullback).lower(parameters, arrays).compile()
    outputs, simulation_pullback = forward(parameters, arrays)
    readout = jax.jit(sensitivity_with_pullback).lower(outputs, ports, expected).compile()
    loss, readout_pullback, (sensitivity, powers) = readout(outputs, ports, expected)
    reverse = jax.jit(backward).lower(simulation_pullback, readout_pullback, seed).compile()
    gradient = reverse(simulation_pullback, readout_pullback, seed)
    return jax.device_get((outputs, sensitivity, powers, loss, gradient))


if __name__ == "__main__":
    start = perf_counter()
    import os

    if os.environ.get("DIFFEROMETOR_NATIVE") == "1":
        os.environ.setdefault("XLA_FLAGS", "--xla_backend_optimization_level=0")
    run()
    elapsed = perf_counter() - start
    print(f"Complete first run (including JIT): {elapsed:.6f} s")
