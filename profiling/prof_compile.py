"""Compile/runtime anatomy of the scored workload under different program structures.

    python profiling/prof_compile.py <variant>

variants
  bench           exactly benchmark.py: jit(vjp(simulate)) with ALL arrays as jit arguments,
                  then readout vjp, then separately compiled backward
  fwd_only        jit(simulate) without any AD (lower bound for a forward program)
  fused           start_here.py structure: one jit(value_and_grad(...))
  static          bench, but integer index arrays closed over as compile-time constants
                  (what a static-pytree wrapper in run_build_step would achieve)
  static_grouped  static + update() evaluating each *used* function once on its own slice
                  (H1 prototype, monkeypatched from outside differometor/)
  static_fwd_only  static without AD
  grouped_fwd_only static_grouped without AD

Prints one JSON line with all phase timings, HLO sizes, residual bytes and peak RSS.
"""
import json
import resource
import sys
import time

sys.path.insert(0, __import__("os").path.dirname(__file__))
from common import build_scored, pin_one_cpu  # noqa: E402

pin_one_cpu()
variant = sys.argv[1]

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax import vmap  # noqa: E402
import differometor as df  # noqa: E402
import differometor.simulate  # noqa: E402,F401
# NB: differometor/__init__ re-exports the *function* `simulate`, shadowing the submodule
# attribute, so fetch the module object explicitly before monkeypatching.
dsim = sys.modules["differometor.simulate"]
from differometor.components import FUNCTIONS  # noqa: E402

arrays, ports, parameters, freqs, expected, pairs = build_scored()
out = {"variant": variant}


def grouped_update(parameters, matrix, function_input_indices, output_indices, matrix_indices, function_indices):
    """Same semantics as differometor.simulate.update, but each used function is vmapped once."""
    fidx = np.asarray(function_indices)
    fin = np.asarray(function_input_indices)
    oi0, oi1 = np.asarray(output_indices[0]), np.asarray(output_indices[1])
    function_inputs = parameters[fin]                      # (K, 7)
    pieces, offset, where = [], 0, {}
    for f in np.unique(fidx):
        ks = np.where(fidx == f)[0]
        res = vmap(FUNCTIONS[f])(function_inputs[ks])      # (len(ks), X)
        width = res.shape[1]
        for local, k in enumerate(ks):
            where[k] = (offset + local * width, width)
        pieces.append(res.reshape(-1))
        offset += res.size
    flat = jnp.concatenate(pieces) if pieces else jnp.zeros((0,), matrix.dtype)
    gather = np.array([where[k][0] + x for k, x in zip(oi0, oi1)], dtype=np.int64)
    return matrix.at[matrix_indices[0], matrix_indices[1]].set(flat[gather])


def split_static(arrays):
    static = {k: (tuple(np.asarray(x) for x in v) if isinstance(v, tuple) else np.asarray(v))
              for k, v in arrays.items()
              if k != "optimized_parameters" and np.issubdtype(np.asarray(v[0] if isinstance(v, tuple) else v).dtype, np.integer)}
    dynamic = {k: v for k, v in arrays.items() if k not in static}
    return static, dynamic


def readout(outputs, detector_ports, reference):
    carrier, signal, noise = outputs
    signals = df.signal_detector(carrier, signal)[detector_ports]
    detected = signals.squeeze() if detector_ports.shape[0] == 1 else signals[0] - signals[1]
    sensitivity = (noise / jnp.abs(detected)).squeeze()
    loss = jnp.mean(jnp.log10(sensitivity / reference))
    return loss, (sensitivity, df.power_detector(carrier))


def timed(key, fn):
    t = time.perf_counter()
    r = fn()
    jax.block_until_ready(r)
    out[key] = round(time.perf_counter() - t, 3)
    return r


def hlo_size(compiled):
    txt = compiled.as_text()
    return sum(1 for line in txt.splitlines() if " = " in line)


if variant.startswith("static") or variant.startswith("grouped"):
    if "grouped" in variant:
        dsim.update = grouped_update
    static, dynamic = split_static(arrays)
    sim = lambda p, a: df.simulate(**dict(a, **static, optimized_parameters=p))  # noqa: E731
    args = dynamic
else:
    sim = lambda p, a: df.simulate(**dict(a, optimized_parameters=p))  # noqa: E731
    args = arrays

if variant in ("bench", "static", "static_grouped"):
    fwd_fn = lambda p, a: jax.vjp(lambda q: sim(q, a), p)  # noqa: E731
    lowered = timed("forward_lower", lambda: jax.jit(fwd_fn).lower(parameters, args))
    out["forward_stablehlo_ops"] = lowered.as_text().count(" = ")
    fwd = timed("forward_compile", lowered.compile)
    out["forward_hlo_ops"] = hlo_size(fwd)
    outputs, pb = timed("forward_exec", lambda: fwd(parameters, args))
    leaves = jax.tree_util.tree_leaves(pb)
    out["residual_count"] = len(leaves)
    out["residual_GB"] = round(sum(x.nbytes for x in leaves) / 1e9, 3)
    big = sorted(((x.nbytes, x.shape, str(x.dtype)) for x in leaves), reverse=True)[:6]
    out["largest_residuals"] = [f"{s} {d} {b/1e6:.0f}MB" for b, s, d in big]
    sens_fn = lambda o, d, r: jax.vjp(lambda v: readout(v, d, r), o, has_aux=True)  # noqa: E731
    lowered = timed("sens_lower", lambda: jax.jit(sens_fn).lower(outputs, ports, expected))
    sf = timed("sens_compile", lowered.compile)
    loss, rpb, _ = timed("sens_exec", lambda: sf(outputs, ports, expected))
    one = jnp.asarray(1.0)
    bwd_fn = lambda a, b, c: a(b(c)[0])[0]  # noqa: E731
    lowered = timed("gradient_lower", lambda: jax.jit(bwd_fn).lower(pb, rpb, one))
    out["gradient_stablehlo_ops"] = lowered.as_text().count(" = ")
    gf = timed("gradient_compile", lowered.compile)
    out["gradient_hlo_ops"] = hlo_size(gf)
    grad = timed("gradient_exec", lambda: gf(pb, rpb, one))
elif variant in ("fwd_only", "static_fwd_only", "grouped_fwd_only"):
    lowered = timed("forward_lower", lambda: jax.jit(sim).lower(parameters, args))
    out["forward_stablehlo_ops"] = lowered.as_text().count(" = ")
    fwd = timed("forward_compile", lowered.compile)
    out["forward_hlo_ops"] = hlo_size(fwd)
    outputs = timed("forward_exec", lambda: fwd(parameters, args))
    grad = None
elif variant == "fused":
    def evaluate(p, a):
        outs = sim(p, a)
        loss, (sens, powers) = readout(outs, ports, expected)
        return loss, (outs, sens, powers)
    lowered = timed("fused_lower", lambda: jax.jit(jax.value_and_grad(evaluate, has_aux=True)).lower(parameters, args))
    out["fused_stablehlo_ops"] = lowered.as_text().count(" = ")
    fc = timed("fused_compile", lowered.compile)
    out["fused_hlo_ops"] = hlo_size(fc)
    (loss, (outputs, *_)), grad = timed("fused_exec", lambda: fc(parameters, args))
else:
    raise SystemExit("unknown variant")

compile_keys = [k for k in out if k.endswith("_lower") or k.endswith("_compile")]
exec_keys = [k for k in out if k.endswith("_exec")]
out["compile_total"] = round(sum(out[k] for k in compile_keys), 2)
out["exec_total"] = round(sum(out[k] for k in exec_keys), 2)
out["peak_rss_GB"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)

# correctness vs saved baseline arrays
base = np.load(__import__("os").path.join(__import__("os").path.dirname(__file__), "..", "runs", "baseline.npz"))
out["max_rel_signal_vs_baseline"] = float(np.max(np.abs(np.asarray(outputs[1]) - base["signal"])) / np.max(np.abs(base["signal"])))
out["max_log_noise_vs_baseline"] = float(np.max(np.abs(np.log(np.asarray(outputs[2])) - np.log(base["noise"]))))
if grad is not None:
    g = np.asarray(grad)
    out["grad_rel_vs_baseline"] = float(np.linalg.norm(g - base["gradient"]) / np.linalg.norm(base["gradient"]))
print("RESULT " + json.dumps(out), flush=True)
