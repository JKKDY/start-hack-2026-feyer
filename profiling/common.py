"""Shared helpers for profiling scripts: build the scored workload exactly as benchmark.py does."""
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["JAX_ENABLE_X64"] = "true"

ROOT = Path(__file__).resolve().parent.parent
PROPERTIES = ["reflectivity", "tuning", "db", "angle", "power", "mass", "length"]


def pin_one_cpu():
    allowed = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, allowed[:1])
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"


def build_scored(fixture_name="uifo_800_3000.json", ref_name="uifo_800_3000_test_data.json"):
    import jax
    import jax.numpy as jnp
    import differometor as df
    from differometor.setups import constrain_inter_grid_cell_spaces

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_enable_compilation_cache", False)
    data = ROOT / "examples" / "data"
    fixture = json.loads((data / fixture_name).read_text())
    reference = json.loads((data / ref_name).read_text())
    setup = df.Setup.from_data(fixture)
    pairs = constrain_inter_grid_cell_spaces(setup.parameters, PROPERTIES)
    initial = []
    for pair in pairs:
        name, prop = pair[0] if isinstance(pair[0], list) else pair
        initial.append(fixture["nodes" if name in fixture["nodes"] else "edges"][name]["properties"][prop])
    parameters = jnp.asarray(initial, dtype=jnp.float64)
    frequencies = jnp.asarray(reference["frequencies"], dtype=jnp.float64)
    expected = jnp.asarray(reference["sensitivities"], dtype=jnp.float64)
    arrays, ports, *_ = df.run_build_step(setup, [("f", "frequency")], frequencies, pairs)
    arrays["optimized_parameters"] = parameters
    return arrays, jnp.asarray(ports), parameters, frequencies, expected, pairs
