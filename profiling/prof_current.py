"""Compile anatomy of the CURRENT simulator (packed sparse solver), benchmark structure.

    python profiling/prof_current.py <variant>

variants
  bench     jit(vjp(simulate)) -> readout -> backward, exactly as benchmark.py (packed solver)
  fwd_only  jit(simulate), no AD
  fused     jit(value_and_grad) in one program (reference point only; benchmark can't use it)
Prints one RESULT json line (phase timings, HLO sizes, residuals, op histogram of the optimised HLO).
"""
import collections
import json
import re
import resource
import sys
import time

sys.path.insert(0, __import__("os").path.dirname(__file__))
from common import pin_one_cpu, ROOT, PROPERTIES  # noqa: E402

pin_one_cpu()
variant = sys.argv[1]
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import differometor as df  # noqa: E402
from differometor.setups import constrain_inter_grid_cell_spaces  # noqa: E402

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_enable_compilation_cache", False)
fixture = json.loads((ROOT / "examples/data/uifo_800_3000.json").read_text())
reference = json.loads((ROOT / "examples/data/uifo_800_3000_test_data.json").read_text())
setup = df.Setup.from_data(fixture)
pairs = constrain_inter_grid_cell_spaces(setup.parameters, PROPERTIES)
vals = []
for pair in pairs:
    name, prop = pair[0] if isinstance(pair[0], list) else pair
    vals.append(fixture["nodes" if name in fixture["nodes"] else "edges"][name]["properties"][prop])
parameters = jnp.asarray(vals, dtype=jnp.float64)
expected = jnp.asarray(reference["sensitivities"])
t = time.perf_counter()
arrays, ports, *_ = df.run_build_step(setup, [("f", "frequency")], jnp.asarray(reference["frequencies"]), pairs,
                                      signal_solver="packed", pack_noise=True)
out = {"variant": variant, "build": round(time.perf_counter() - t, 3)}
arrays["optimized_parameters"] = parameters
ports = jnp.asarray(ports)


def readout(outputs, detector_ports, ref):
    carrier, signal, noise = outputs
    signals = df.signal_detector(carrier, signal)[detector_ports]
    detected = signals.squeeze() if detector_ports.shape[0] == 1 else signals[0] - signals[1]
    sensitivity = (noise / jnp.abs(detected)).squeeze()
    return jnp.mean(jnp.log10(sensitivity / ref)), (sensitivity, df.power_detector(carrier))


def timed(key, fn):
    t0 = time.perf_counter()
    r = fn()
    jax.block_until_ready(r)
    out[key] = round(time.perf_counter() - t0, 3)
    return r


def anatomy(prefix, lowered, compiled):
    out[prefix + "_stablehlo_ops"] = lowered.as_text().count(" = ")
    txt = compiled.as_text()
    ops = collections.Counter(m.group(1) for m in re.finditer(r"= \S+ (\w[\w-]*)\(", txt))
    out[prefix + "_hlo_ops"] = sum(ops.values())
    out[prefix + "_top_ops"] = ops.most_common(10)
    out[prefix + "_fusions"] = txt.count("fusion(")
    out[prefix + "_custom_calls"] = re.findall(r'custom_call_target="([^"]+)"', txt)[:12]


_policy = __import__("os").environ.get("CHECKPOINT")
_simulate = df.simulate
if _policy:   # rematerialisation experiment: keep (almost) nothing for the backward pass, recompute instead
    _simulate = jax.checkpoint(df.simulate, policy=getattr(jax.checkpoint_policies, _policy))
    out["checkpoint"] = _policy
if __import__("os").environ.get("XLA_FLAGS"):
    out["xla_flags"] = __import__("os").environ["XLA_FLAGS"]
sim = lambda p, a: _simulate(**dict(a, optimized_parameters=p))  # noqa: E731
if variant == "bench":
    lw = timed("forward_lower", lambda: jax.jit(lambda p, a: jax.vjp(lambda q: sim(q, a), p)).lower(parameters, arrays))
    fc = timed("forward_compile", lw.compile)
    anatomy("forward", lw, fc)
    outputs, pb = timed("forward_exec", lambda: fc(parameters, arrays))
    leaves = jax.tree_util.tree_leaves(pb)
    out["residual_count"] = len(leaves)
    out["residual_MB"] = round(sum(x.nbytes for x in leaves) / 1e6, 1)
    out["largest_residuals"] = [f"{x.shape} {x.dtype}" for x in sorted(leaves, key=lambda x: -x.nbytes)[:5]]
    lw = timed("sens_lower", lambda: jax.jit(lambda o, d, r: jax.vjp(lambda v: readout(v, d, r), o, has_aux=True)).lower(outputs, ports, expected))
    sc = timed("sens_compile", lw.compile)
    loss, rpb, _ = sc(outputs, ports, expected)
    one = jnp.asarray(1.0)
    lw = timed("gradient_lower", lambda: jax.jit(lambda a, b, c: a(b(c)[0])[0]).lower(pb, rpb, one))
    gc = timed("gradient_compile", lw.compile)
    anatomy("gradient", lw, gc)
    g = timed("gradient_exec", lambda: gc(pb, rpb, one))
    import numpy as np
    base = np.load(ROOT / "runs" / "baseline.npz")
    out["grad_rel_vs_baseline"] = float(np.linalg.norm(np.asarray(g) - base["gradient"]) / np.linalg.norm(base["gradient"]))
    out["max_log_noise_vs_baseline"] = float(np.max(np.abs(np.log(np.asarray(outputs[2])) - np.log(base["noise"]))))
elif variant == "fwd_only":
    lw = timed("forward_lower", lambda: jax.jit(sim).lower(parameters, arrays))
    fc = timed("forward_compile", lw.compile)
    anatomy("forward", lw, fc)
    timed("forward_exec", lambda: fc(parameters, arrays))
elif variant == "fused":
    def ev(p):
        o = sim(p, arrays)
        return readout(o, ports, expected)[0]
    lw = timed("fused_lower", lambda: jax.jit(jax.value_and_grad(ev)).lower(parameters))
    fc = timed("fused_compile", lw.compile)
    anatomy("fused", lw, fc)
    timed("fused_exec", lambda: fc(parameters))
out["compile_total"] = round(sum(v for k, v in out.items() if k.endswith(("_lower", "_compile"))), 2)
out["peak_rss_GB"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)
print("RESULT " + json.dumps(out), flush=True)
