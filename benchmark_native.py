#!/usr/bin/env python3
"""JAX-free variant of benchmark.py: time the same public workload on the native C++ engine.

Same workload, phases, validation and report format as benchmark.py (which
stays unchanged and remains the official scored benchmark). Differences:

- No JAX is imported. The simulator and its reverse-mode gradient run in the
  native engine (`differometor_native`, ahead-of-time compiled C++), so there
  are no trace / lower / compile phases.
- The readout (sensitivity, powers, loss) and its vector-Jacobian product are
  written in NumPy. The readout cotangents use the native complex convention
  (dL/dRe z + i dL/dIm z) and are fed to the native pullback.
- Validation is identical: finite outputs, reference sensitivity within
  |ln| 1e-5, non-zero gradient, and the central-difference spot check on
  mb11.tuning with steps 1e-3 and 5e-4 (rtol 1 %, atol 1e-6).
"""
from time import perf_counter

IMPORT_START = perf_counter()
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess

os.environ.setdefault("MPLBACKEND", "Agg")

# Set Linux CPU affinity before numerical libraries initialize their workers.
cpu_parser = argparse.ArgumentParser(add_help=False)
cpu_parser.add_argument("--cpus", type=int)
cpu_options, _ = cpu_parser.parse_known_args()
if cpu_options.cpus is not None:
    if not hasattr(os, "sched_getaffinity"):
        cpu_parser.error("--cpus requires Linux CPU affinity (Linux or WSL).")
    allowed_cpus = sorted(os.sched_getaffinity(0))
    if not 1 <= cpu_options.cpus <= len(allowed_cpus):
        cpu_parser.error(f"--cpus must be between 1 and {len(allowed_cpus)}.")
    os.sched_setaffinity(0, allowed_cpus[:cpu_options.cpus])
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(cpu_options.cpus)

import numpy as np
import differometor_native as dn
from differometor_native.setups import constrain_inter_grid_cell_spaces
from differometor_native.simulate import simulate_vjp

IMPORT_SECONDS = perf_counter() - IMPORT_START
PUBLIC_REPO = "https://github.com/artificial-scientist-lab/Differometor"
BASELINE_COMMIT = "b0dc7c79e7c72a1b010174f6bdcad01efa51d6ff"
FIXTURE_SHA256 = "ff4ba353a2a03cdce7f3e61bad729c2e00888d842c28ba9cf699f95afb87e138"
REFERENCE_SHA256 = "726f2cfe6986cb991fa48d5a28127b9330c3dc7e69d5bd193c8748acef03adc3"
PROPERTIES = ["reflectivity", "tuning", "db", "angle", "power", "mass", "length"]


def cpu_model():
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    if platform.system() == "Darwin":
        return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    return platform.processor() or platform.machine()


def git_state(repo):
    """Identify the native engine and Python package sources, like benchmark.py does for differometor/."""
    repo = repo.resolve()
    sources = sorted((repo / "native" / "src").glob("*.cpp")) + sorted((repo / "native" / "include").rglob("*.hpp"))
    sources += sorted((repo / "native" / "python").rglob("*.py")) + sorted((repo / "native" / "bindings").glob("*.cpp"))
    state = {
        "commit": None,
        "modified": None,
        "diff_sha256": None,
        "native_files_sha256": {
            path.relative_to(repo).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources if path.is_file()
        },
    }

    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True, stderr=subprocess.DEVNULL).strip()

    try:
        if Path(git("rev-parse", "--show-toplevel")).resolve() != repo:
            return state
        state.update(
            commit=git("rev-parse", "HEAD"),
            modified=bool(git("status", "--porcelain", "--untracked-files=no")),
            diff_sha256=hashlib.sha256(git("diff", "HEAD").encode()).hexdigest(),
        )
    except (OSError, subprocess.SubprocessError):
        pass
    return state


def output_paths(args, parser):
    """Reject invalid or overlapping output destinations before simulation."""
    if args.output.suffix != ".json":
        parser.error("--output must use a .json filename.")
    plot_path = args.plot or args.output.with_suffix(".png")
    if plot_path.suffix.lower() not in {".png", ".svg", ".pdf"}:
        parser.error("--plot must use a .png, .svg or .pdf filename.")
    if args.save_arrays and args.save_arrays.suffix != ".npz":
        parser.error("--save-arrays must use a .npz filename.")
    paths = [args.output, plot_path]
    if args.save_arrays:
        paths.append(args.save_arrays)
    if len({path.resolve() for path in paths}) != len(paths):
        parser.error("The JSON report, plot and saved arrays need different paths.")
    if any(path.is_dir() for path in paths):
        parser.error("Output destinations must be filenames, not directories.")
    return plot_path


def forward_with_pullback(parameters, arrays):
    # Native forward pass with the CoDiPack tape recorded; returns the pullback.
    return simulate_vjp(arrays, parameters, cotangent_convention="native")


def readout(outputs, detector_ports, reference):
    """benchmark.py's readout in NumPy: (loss, (sensitivity, powers))."""
    carrier, signal, noise = outputs
    signals = dn.signal_detector(carrier, signal)[detector_ports]
    detected = signals.squeeze() if detector_ports.shape[0] == 1 else signals[0] - signals[1]
    sensitivity = (noise / np.abs(detected)).squeeze()
    loss = np.mean(np.log10(sensitivity / reference))
    return loss, (sensitivity, dn.power_detector(carrier))


def readout_with_pullback(outputs, detector_ports, reference):
    """Readout plus its VJP (native convention), mirroring jax.vjp(readout, has_aux=True)."""
    carrier, signal, noise = outputs
    loss, aux = readout(outputs, detector_ports, reference)
    n = carrier.shape[0]
    signals = dn.signal_detector(carrier, signal)[detector_ports]
    detected = signals.squeeze() if detector_ports.shape[0] == 1 else signals[0] - signals[1]
    count = np.size(aux[0])

    def pullback(cotangent):
        scale = cotangent / (count * np.log(10.0))
        noise_ct = np.broadcast_to(scale / noise, noise.shape).copy()
        magnitude = np.abs(detected)
        detected_ct = -scale / magnitude * (detected / magnitude)  # d|d| -> dL/dRe d + i dL/dIm d
        signs = [1.0] if detector_ports.shape[0] == 1 else [1.0, -1.0]
        signal_ct = np.zeros_like(signal)
        carrier_ct = np.zeros_like(carrier)
        for sign, port in zip(signs, np.asarray(detector_ports).reshape(-1)):
            w = sign * np.reshape(detected_ct, signal.shape[1:])
            c = carrier[port]
            # d = conj(c) * s_upper + c * s_lower
            signal_ct[port] += np.broadcast_to(c, w.shape) * w
            signal_ct[n + port] += np.broadcast_to(np.conj(c), w.shape) * w
            contribution = w * np.conj(signal[n + port]) + np.conj(w) * signal[port]
            carrier_ct[port] += contribution.sum() if c.shape[0] == 1 else contribution
        return (carrier_ct, signal_ct, noise_ct)

    return loss, pullback, aux


def backward(simulation_pullback, readout_pullback, cotangent):
    output_cotangents = readout_pullback(cotangent)
    return simulation_pullback(output_cotangents)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--repo", type=Path, default=here, help="Checkout containing examples/data and native/")
    parser.add_argument("--cpus", type=int, help="Pin this process to N allocated logical CPUs before imports (Linux/WSL)")
    parser.add_argument("--repeats", type=int, default=10, help="Repeated forward + sensitivity + gradient evaluations")
    parser.add_argument("--output", type=Path, default=Path("benchmark-native.json"), help="JSON report (.json)")
    parser.add_argument("--plot", type=Path, help="Timing plot (.png, .svg or .pdf); defaults to the JSON filename with .png suffix")
    parser.add_argument("--save-arrays", type=Path, help="Save physical outputs, sensitivity, powers, parameters and gradient (.npz)")
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("Use at least three timed repetitions.")
    plot_path = output_paths(args, parser)
    repo = args.repo.resolve()

    phases = [{"key": "imports", "label": "Python / library imports", "category": "setup", "seconds": IMPORT_SECONDS}]

    def timed(key, label, category, function):
        start = perf_counter()
        result = function()
        seconds = perf_counter() - start
        phases.append({"key": key, "label": label, "category": category, "seconds": seconds})
        print(f"{label}: {seconds:.6f} s", flush=True)
        return result

    version = timed("initialize", "Initialize native engine", "setup", dn._native.version)
    print(f"CPU: {cpu_model()}; affinity: {len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else 'unknown'}; "
          f"native engine {version}", flush=True)

    data_dir = repo / "examples" / "data"
    fixture_path = data_dir / "uifo_800_3000.json"
    reference_path = data_dir / "uifo_800_3000_test_data.json"

    def load_inputs():
        fixture_bytes = fixture_path.read_bytes()
        reference_bytes = reference_path.read_bytes()
        return fixture_bytes, reference_bytes, json.loads(fixture_bytes), json.loads(reference_bytes)

    fixture_bytes, reference_bytes, fixture, reference = timed("load", "Read and parse public inputs", "setup", load_inputs)
    if (hashlib.sha256(fixture_bytes).hexdigest(), hashlib.sha256(reference_bytes).hexdigest()) != (FIXTURE_SHA256, REFERENCE_SHA256):
        parser.error("Restore the unchanged public fixtures from the baseline commit.")
    source = git_state(repo)

    def build_inputs():
        setup = dn.Setup.from_data(fixture)
        pairs = constrain_inter_grid_cell_spaces(setup.parameters, PROPERTIES)
        initial = []
        for pair in pairs:
            members = pair if isinstance(pair[0], list) else [pair]
            values = [fixture["nodes" if name in fixture["nodes"] else "edges"][name]["properties"][prop] for name, prop in members]
            if any(value != values[0] for value in values):
                raise ValueError("Grouped length parameters must agree in the public fixture")
            initial.append(values[0])
        parameters = np.asarray(initial, dtype=np.float64)
        frequencies = np.asarray(reference["frequencies"], dtype=np.float64)
        expected = np.asarray(reference["sensitivities"], dtype=np.float64)
        arrays, ports, *_ = dn.run_build_step(setup, [("f", "frequency")], frequencies, pairs)
        arrays["optimized_parameters"] = parameters
        return arrays, np.asarray(ports), parameters, frequencies, expected, pairs, 1.0

    arrays, ports, parameters, frequencies, expected, pairs, one = timed("build", "Build optical setup and prepare inputs", "setup", build_inputs)
    print(f"Workload: {len(frequencies)} frequencies; {len(parameters)} physical design variables; no optimizer", flush=True)

    def no_compilation(key, label):
        # Same phase set as benchmark.py (plot_timings expects it); the native
        # engine is compiled ahead of time, so these phases are exactly zero.
        for suffix, text in (("_lower", "trace / lower"), ("_compile", "compile")):
            phases.append({"key": key + suffix, "label": f"{label}: {text} (none, ahead-of-time)", "category": "compile",
                           "seconds": 0.0})

    no_compilation("forward", "Forward")
    outputs, simulation_pullback = timed("forward", "Forward simulation + AD tape", "execute", lambda: forward_with_pullback(parameters, arrays))
    no_compilation("sensitivity", "Sensitivity")
    loss, readout_pullback, (sensitivity, powers) = timed("sensitivity", "Sensitivity, powers and scalar loss", "execute", lambda: readout_with_pullback(outputs, ports, expected))
    no_compilation("gradient", "Gradient")
    gradient = timed("gradient", "Reverse-mode gradient", "execute", lambda: backward(simulation_pullback, readout_pullback, one))
    host = timed("host_transfer", "Collect results", "output", lambda: (outputs, sensitivity, powers, loss, gradient))
    first_phases = list(phases)
    total = sum(phase["seconds"] for phase in first_phases)
    print(f"Complete first-run total (excluding validation): {total:.6f} s", flush=True)

    repeat_samples = []
    for index in range(args.repeats):
        del simulation_pullback, readout_pullback
        phases.clear()
        outputs, simulation_pullback = timed("forward", "Forward", "execute", lambda: forward_with_pullback(parameters, arrays))
        loss, readout_pullback, (sensitivity, powers) = timed("sensitivity", "Sensitivity", "execute", lambda: readout_with_pullback(outputs, ports, expected))
        gradient = timed("gradient", "Gradient", "execute", lambda: backward(simulation_pullback, readout_pullback, one))
        host = timed("host_transfer", "Collect results", "output", lambda: (outputs, sensitivity, powers, loss, gradient))
        sample = {phase["key"]: phase["seconds"] for phase in phases}
        repeat_samples.append(sample)
        print(f"Repeat {index + 1:02d}/{args.repeats}: {sum(sample.values()):.6f} s", flush=True)

    # Validation only: no timers below contribute to the benchmark total.
    del simulation_pullback, readout_pullback
    print("Validating outputs, reference sensitivity and finite-difference gradient...", flush=True)
    host_outputs, host_sensitivity, host_powers, host_loss, host_gradient = host
    for value in (*host_outputs, host_sensitivity, host_powers, host_loss, host_gradient):
        if not np.isfinite(value).all():
            raise ValueError("Non-finite physical output, sensitivity, loss or gradient")
    if host_sensitivity.shape != expected.shape or np.any(host_sensitivity <= 0):
        raise ValueError("Expected one positive sensitivity per frequency")
    log_error = np.abs(np.log(host_sensitivity) - np.log(expected))
    np.testing.assert_allclose(np.log(host_sensitivity), np.log(expected), atol=1e-5, rtol=0)
    if host_gradient.shape != parameters.shape or not np.any(host_gradient != 0):
        raise ValueError("Expected a finite, nonzero full parameter gradient")

    check_index = pairs.index(["mb11", "tuning"])

    def evaluate_loss(p):
        return float(readout(dn.simulate(**dict(arrays, optimized_parameters=p)), ports, expected)[0])

    autodiff = float(host_gradient[check_index])
    finite_differences = []
    for check_step in (1e-3, 5e-4):
        upper_p, lower_p = parameters.copy(), parameters.copy()
        upper_p[check_index] += check_step
        lower_p[check_index] -= check_step
        estimate = (evaluate_loss(upper_p) - evaluate_loss(lower_p)) / (2 * check_step)
        np.testing.assert_allclose(autodiff, estimate, rtol=0.01, atol=1e-6)
        finite_differences.append({"step": check_step, "central_difference": estimate, "absolute_error": abs(autodiff - estimate)})
    np.testing.assert_allclose(finite_differences[0]["central_difference"], finite_differences[1]["central_difference"], rtol=0.01, atol=1e-6)

    totals = [sum(sample.values()) for sample in repeat_samples]
    report = {
        "benchmark": "start-munich-2026-public-differometor-uifo-v2-native",
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "starter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source": {"public_repository": PUBLIC_REPO, "baseline_commit": BASELINE_COMMIT, **source},
        "inputs": {"setup": fixture_path.name, "setup_sha256": FIXTURE_SHA256, "reference": reference_path.name, "reference_sha256": REFERENCE_SHA256, "frequency_count": len(frequencies), "gradient_parameter_count": len(parameters), "optimization_pairs": pairs},
        "workload": {"objective": "mean(log10(sensitivity / public_reference))", "gradient_coordinates": "physical parameter values; public grid-length tying", "optimizer_steps": 0, "forward_residuals": "native CoDiPack tape, replayed by the reverse pass", "total_scope": "imports, engine initialization, input I/O, setup, forward, sensitivity/powers/loss, reverse-mode gradient and result collection; no compilation (ahead-of-time native engine)", "excluded": "validation and benchmark harness bookkeeping, repeated profiling, plot/report generation; interpreter launch before the first timer"},
        "environment": {
            "cpu_model": cpu_model(), "cpu_affinity_count": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "cpu_affinity_ids": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "requested_logical_cpus": args.cpus,
            "logical_cpus_on_host": os.cpu_count(), "platform": platform.platform(), "python": platform.python_version(),
            "packages": {name: importlib.metadata.version(name) for name in ("differometor-native", "numpy")},
            "backend": "native (C++20, CoDiPack, Eigen, SuperLU)", "jax_imported": False,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "thread_limits": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        },
        "phases": first_phases,
        "repeat_samples": repeat_samples,
        "timing_seconds": {"end_to_end": total, "trace_lower_and_compile": sum(p["seconds"] for p in first_phases if p["category"] == "compile"), "runtime_samples": totals, "runtime_median": statistics.median(totals), "runtime_min": min(totals), "runtime_max": max(totals), "phase_runtime_medians": {key: statistics.median(sample[key] for sample in repeat_samples) for key in repeat_samples[0]}},
        "validation": {
            "all_outputs_and_gradients_finite": True, "upstream_sensitivity_check_passed": True, "max_absolute_log_sensitivity_error": float(log_error.max()),
            "gradient_check": {"parameter": "mb11.tuning", "autodiff": autodiff, "finite_differences": finite_differences, "rtol": 0.01, "atol": 1e-6, "passed": True},
            "gradient_shape": list(host_gradient.shape), "gradient_norm": float(np.linalg.norm(host_gradient)), "loss": float(host_loss),
            "output_shapes": {name: list(value.shape) for name, value in zip(("carrier", "signal", "noise"), host_outputs)},
            "output_dtypes": {name: str(value.dtype) for name, value in zip(("carrier", "signal", "noise"), host_outputs)},
        },
    }
    if args.save_arrays:
        args.save_arrays.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save_arrays, carrier=host_outputs[0], signal=host_outputs[1], noise=host_outputs[2], sensitivity=host_sensitivity, powers=host_powers, gradient=host_gradient, parameters=parameters, frequencies=frequencies)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    from plot_timings import render
    render(report, plot_path)
    print(f"PASS: reference sensitivity and gradient spot check; full runtime median {statistics.median(totals):.6f} s")
    print(f"JSON: {args.output}\nPlot: {plot_path}")


if __name__ == "__main__":
    main()
