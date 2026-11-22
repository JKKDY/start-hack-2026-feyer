"""Cold-start timing of the native simulator on the benchmark workload.

Mirrors benchmark.py's phases on the native path: import, build (setup +
plan), forward with AD state, readout (sensitivity, loss), reverse pass.
The readout VJP uses JAX only if JAX is installed; otherwise it uses NumPy.

Usage: python bench/bench_native_simulator.py <repo root>
"""

import json
import sys
import time

t0 = time.perf_counter()
import numpy as np  # noqa: E402

import differometor_native as dn  # noqa: E402

simulate = sys.modules["differometor_native.simulate"]
timings = {"import": time.perf_counter() - t0}

repo = sys.argv[1] if len(sys.argv) > 1 else "."
PROPERTIES = ["reflectivity", "tuning", "db", "angle", "power", "mass", "length"]

t = time.perf_counter()
data = json.loads(open(f"{repo}/examples/data/uifo_800_3000.json").read())
reference = json.loads(open(f"{repo}/examples/data/uifo_800_3000_test_data.json").read())
setup = dn.Setup.from_data(data)
pairs = dn.setups.constrain_inter_grid_cell_spaces(setup.parameters, PROPERTIES)
x = []
for p in pairs:
    name, prop = p[0] if isinstance(p[0], list) else p
    x.append(data["nodes" if name in data["nodes"] else "edges"][name]["properties"][prop])
x = np.asarray(x, dtype=np.float64)
frequencies = np.asarray(reference["frequencies"])
expected = np.asarray(reference["sensitivities"])
arrays, ports, *_ = dn.run_build_step(setup, [("f", "frequency")], frequencies, pairs)
timings["build"] = time.perf_counter() - t

t = time.perf_counter()
(carrier, signal, noise), pullback = simulate.simulate_vjp(arrays, x, cotangent_convention="native")
timings["forward_with_state"] = time.perf_counter() - t

# Readout and its cotangents (analytic, native convention).
t = time.perf_counter()
n = carrier.shape[0]
sig = (np.conj(carrier) * signal[:n] + carrier * signal[n:2 * n])[ports]
detected = sig[0] if len(ports) == 1 else sig[0] - sig[1]
sensitivity = noise / np.abs(detected)
loss = np.mean(np.log10(sensitivity / expected))
f = len(sensitivity)
noise_ct = 1.0 / (f * np.log(10) * noise)
d_bar = -1.0 / (f * np.log(10) * np.abs(detected)) * detected / np.abs(detected)
sign = [1.0] if len(ports) == 1 else [1.0, -1.0]
signal_ct = np.zeros_like(signal)
carrier_ct = np.zeros_like(carrier)
for k, port in enumerate(ports):
    w = sign[k] * d_bar
    signal_ct[port] += carrier[port, 0] * w
    signal_ct[n + port] += np.conj(carrier[port, 0]) * w
    carrier_ct[port, 0] += np.sum(w * np.conj(signal[n + port]) + np.conj(w) * signal[port])
timings["readout"] = time.perf_counter() - t

t = time.perf_counter()
(gradient,) = pullback((carrier_ct, signal_ct, noise_ct))
timings["gradient"] = time.perf_counter() - t

total = sum(timings.values())
log_error = float(np.max(np.abs(np.log(sensitivity) - np.log(expected))))
print(json.dumps(dict(timings=timings, total=total, max_log_sensitivity_error=log_error, loss=float(loss),
                      gradient_norm=float(np.linalg.norm(gradient)), parameters=int(x.size)), indent=2))
