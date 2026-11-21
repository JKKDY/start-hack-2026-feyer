"""Compile reusable component kernels and verify their reload in a separate process.

This deployment experiment is independent of the scored benchmark. Load only
trusted bundles produced locally. JAX's executable format uses pickle.
"""

import argparse
import hashlib
import json
import pickle
import platform
import sys
from pathlib import Path
from time import perf_counter

import jax
import jaxlib
import numpy as np
from jax.experimental import serialize_executable


ROOT = Path(__file__).resolve().parents[1]
COMPONENT_SOURCE = ROOT / "differometor" / "components.py"


def platform_key():
    return {
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "platform": platform.system(),
        "architecture": platform.machine(),
        "components_sha256": hashlib.sha256(COMPONENT_SOURCE.read_bytes()).hexdigest(),
    }


def operators(component):
    values = jax.vmap(component)

    def transpose(inputs, cotangent):
        _, pullback = jax.vjp(values, inputs)
        return pullback(cotangent)[0]

    return values, transpose


def make_case_vectors():
    """Derive inputs from real builder plans, with a few representatives per sweep."""
    sys.path.insert(0, str(ROOT / "tests"))
    sys.path.insert(0, str(ROOT))
    import differometor as df
    from cases import CASES
    from differometor.build import LINKING_FUNCTIONS
    from differometor.static import unwrap_static

    def sweep(base, indices, values):
        values = np.asarray(values)[:, :5]
        result = np.tile(base, (values.shape[1], 1))
        result[:values.shape[1], indices] = values.T
        return result

    rng = np.random.default_rng(4)
    for name in ("cavity_tuning", "uifo3_scored", "uifo3_20_5000"):
        case = CASES[name]()
        arrays, *_ = df.run_build_step(
            case["setup"], case["changing_pairs"], case["changing_values"], case["pairs"],
        )
        arrays = unwrap_static(arrays)
        base = np.array(arrays["parameters"], dtype=np.complex128)
        base[0, arrays["optimized_parameter_indices"]] = case["initial"][arrays["optimized_value_indices"]]
        linked = base[0, arrays["linked_indices"]].copy()
        for index, kind, value in zip(arrays["indices_to_link"], arrays["linking_function_indices"],
                                      linked, strict=True):
            base[0, index] = np.asarray(LINKING_FUNCTIONS[kind](value))
        carrier = sweep(base, arrays["carrier_changing_parameter_indices"], arrays["carrier_changing_values"])
        signal = sweep(carrier, arrays["signal_changing_parameter_indices"], arrays["signal_changing_values"])
        groups = {}
        for stage, params in (("carrier", carrier), ("signal", signal), ("noise", signal)):
            for index, packing in enumerate(arrays[f"{stage}_function_indices"]):
                if packing is not None:
                    inputs, _, _ = packing
                    rows = params[:, inputs].reshape(-1, inputs.shape[1])
                    # Some builder dummy groups only require their first three inputs.
                    if rows.shape[1] < 7:
                        rows = np.pad(rows, ((0, 0), (0, 7-rows.shape[1])))
                    groups.setdefault(index, []).append(rows)
        for index, groups_inputs in groups.items():
            inputs = np.ascontiguousarray(np.concatenate(groups_inputs), dtype=np.complex128)
            cotangents = rng.normal(size=(len(inputs), 4)) + 1j * rng.normal(size=(len(inputs), 4))
            yield name, index, inputs, cotangents


def build(destination, tile):
    sys.path.insert(0, str(ROOT))
    from differometor.components import FUNCTIONS

    destination.mkdir(parents=True)
    input_shape = jax.ShapeDtypeStruct((tile, 7), np.complex128)
    output_shape = jax.ShapeDtypeStruct((tile, 4), np.complex128)
    began = perf_counter()
    for index, component in enumerate(FUNCTIONS):
        forward, reverse = operators(component)
        paths = (("forward", forward, (input_shape,)), ("reverse", reverse, (input_shape, output_shape)))
        for label, function, shapes in paths:
            executable = jax.jit(function).lower(*shapes).compile()
            with (destination / f"{index:02d}-{label}.pkl").open("wb") as file:
                pickle.dump(serialize_executable.serialize(executable), file)
    compilation = perf_counter() - began
    cases = {}
    reference_ops = [(jax.jit(forward), jax.jit(reverse)) for forward, reverse in map(operators, FUNCTIONS)]
    for name, index, inputs, cotangents in make_case_vectors():
        forward, reverse = reference_ops[index]
        values, tangent = jax.device_get((forward(inputs), reverse(inputs, cotangents)))
        filename = f"{name}-{index:02d}.npz"
        np.savez(destination / filename, inputs=inputs, cotangents=cotangents, values=values, tangent=tangent)
        cases.setdefault(name, []).append((index, filename))
    manifest = {"platform": platform_key(), "tile": tile, "component_count": len(FUNCTIONS),
                "compile_seconds": compilation, "cases": cases}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Built {2 * len(FUNCTIONS)} executables in {compilation:.3f} s, tile size {tile}")


def tile_calls(compiled, inputs, tile, *extra):
    result = []
    for start in range(0, len(inputs), tile):
        batch = inputs[start:start+tile]
        count = len(batch)
        tail = tile - count
        # Repeat a valid physical input for the unused lanes, then discard those outputs.
        batch = np.pad(batch, ((0, tail), (0, 0)), mode="edge") if tail else batch
        rest = [np.pad(other[start:start+count], ((0, tail), (0, 0))) for other in extra]
        result.append(np.asarray(jax.device_get(compiled(batch, *rest)))[:count])
    return np.concatenate(result, axis=0)


def check(destination):
    manifest = json.loads((destination / "manifest.json").read_bytes())
    if manifest["platform"] != platform_key():
        raise ValueError("The bundle must match the platform, component source, and JAX runtime")
    began = perf_counter()
    programs = {}
    for index in range(manifest["component_count"]):
        for stage in ("forward", "reverse"):
            with (destination / f"{index:02d}-{stage}.pkl").open("rb") as file:
                programs[index, stage] = serialize_executable.deserialize_and_load(*pickle.load(file))
    load_seconds = perf_counter() - began
    print(f"Loaded {len(programs)} executables in {load_seconds:.3f} s")
    for name, groups in manifest["cases"].items():
        count = 0
        finite = 0
        worst = 0.0
        for index, filename in groups:
            with np.load(destination / filename) as expected:
                inputs, cotangents = expected["inputs"], expected["cotangents"]
                result = tile_calls(programs[index, "forward"], inputs, manifest["tile"])
                tangent = tile_calls(programs[index, "reverse"], inputs, manifest["tile"], cotangents)
                for actual, ref in ((result, expected["values"]), (tangent, expected["tangent"])):
                    np.testing.assert_array_equal(np.isfinite(actual), np.isfinite(ref))
                    mask = np.isfinite(ref)
                    np.testing.assert_allclose(actual[mask], ref[mask], rtol=1e-10, atol=1e-12)
                    finite += np.count_nonzero(mask)
                    worst = max(worst, float(np.max(np.abs(actual[mask]-ref[mask]), initial=0)))
                count += len(inputs)
        print(f"{name}: {count} component inputs, {finite} finite value/VJP entries, worst absolute drift {worst:.3g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("build", "check"))
    parser.add_argument("destination", type=Path)
    parser.add_argument("--tile", type=int, default=128)
    args = parser.parse_args()
    if args.tile < 1:
        parser.error("--tile must be positive")
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_enable_compilation_cache", False)
    if args.mode == "build":
        build(args.destination, args.tile)
    else:
        check(args.destination)
