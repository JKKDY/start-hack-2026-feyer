"""List every eager JAX compilation triggered by the build step, attributed to differometor source lines."""
import collections, json, logging, re, sys, time
sys.path.insert(0, __import__("os").path.dirname(__file__))
from common import ROOT, PROPERTIES, pin_one_cpu
pin_one_cpu()
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True); jax.config.update("jax_enable_compilation_cache", False)
import differometor as df
from differometor.setups import constrain_inter_grid_cell_spaces
from jax._src import monitoring
import traceback
n = [0]; secs = [0.0]; locs = collections.Counter(); ops = collections.defaultdict(collections.Counter)
def listener(ev, d, **k):
    if not ev.endswith("backend_compile_duration"):
        return
    n[0] += 1; secs[0] += d
    frames = [f for f in traceback.extract_stack() if "/differometor/" in f.filename]
    key = f"{frames[-1].filename.split('/')[-1]}:{frames[-1].lineno} ({frames[-1].name})" if frames else "<outside differometor>"
    locs[key] += 1
monitoring.register_event_duration_secs_listener(listener)
jax.devices("cpu")
fixture = json.loads((ROOT / "examples/data/uifo_800_3000.json").read_text())
ref = json.loads((ROOT / "examples/data/uifo_800_3000_test_data.json").read_text())
t = time.perf_counter()
setup = df.Setup.from_data(fixture)
pairs = constrain_inter_grid_cell_spaces(setup.parameters, PROPERTIES)
freqs = jnp.asarray(ref["frequencies"])
kw = {"signal_solver": "packed", "pack_noise": True} if "signal_solver" in df.run_build_step.__code__.co_varnames else {}
arrays, *_ = df.run_build_step(setup, [("f", "frequency")], freqs, pairs, **kw)
jax.block_until_ready(jax.tree_util.tree_leaves(arrays))
print(f"cold build {time.perf_counter() - t:.3f} s; backend compiles: {n[0]} ({secs[0]:.3f} s)")
for k, v in locs.most_common(40): print(f"  {v:4d}  {k}")
