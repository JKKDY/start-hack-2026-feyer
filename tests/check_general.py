#!/usr/bin/env python3
"""Correctness and timing harness: compare the current simulator against cached references.

    python tests/check_general.py --save                 # once: references from the unmodified simulator (git ref main)
    python tests/check_general.py                        # after every change: scored fixture only
    python tests/check_general.py --suite all            # every setup in tests/cases.py
    python tests/check_general.py --cases aligo_am,uifo3_jitter
    python tests/check_general.py --candidate-ref baseline_plus_sparsity   # test another commit's simulator

See docs/correctness-harness.md for the design, tolerances and report format.

Reference simulator. `--save` exports `differometor/` from `--reference-ref` (default `main`, the
unmodified upstream simulator) into tests/.tmp/simulator-<sha>/ and runs every case with that code,
while the harness and case definitions come from the working tree. Check mode runs the working tree.

Fuzzing. Every case is evaluated at its nominal parameters plus `--samples` perturbed parameter sets
(rules in cases.FUZZ_RULES, seeded). The parameter sets are stored with the references, so check mode
replays exactly the same points.

Tolerances. In --save mode each evaluated point is re-run with every optimised parameter (except lengths)
nudged by 1e-15 relative (random sign; a few ulps), roughly the backward error a stable solver injects.
The output change is that point's round-off floor. In check mode the tolerance per quantity and point is
clamp(FLOOR_FACTOR * floor, MIN_TOL, MAX_TOL); quantities whose floor already exceeds MAX_TOL are
ill-conditioned and get a loose FLOOR_FACTOR * floor gate (flagged).

Each case runs in a fresh subprocess (fresh compile, bounded memory) through the same three-stage
structure benchmark.py uses: jit(vjp(simulate)) -> jit(readout vjp) -> jit(backward). Custom VJPs and
pytree tricks are therefore exercised exactly as scored, and the per-case timings mirror the benchmark's
first-run total (import, build, compile, first evaluation) and warm evaluation time.
"""
import argparse
import io
import json
import os
import platform
import statistics
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REF_DIR = HERE / "reference"
TMP = HERE / ".tmp"
QUANTITIES = ("carrier", "signal", "noise", "sensitivity", "powers", "loss", "gradient")

# tolerance = clamp(FLOOR_FACTOR * floor, MIN_TOL, MAX_TOL)
MAX_TOL = {"carrier": 1e-8, "signal": 1e-6, "powers": 1e-8, "noise": 1e-7, "sensitivity": 1e-7,
           "loss": 1e-7, "gradient": 1e-4}
MIN_TOL = {
    "carrier": 1e-9,      # normwise relative error per sweep column
    "signal": 1e-8,
    "powers": 1e-9,       # relative where power is non-negligible
    "noise": 1e-8,        # max |log(a) - log(b)|
    "sensitivity": 1e-8,
    "loss": 1e-9,         # absolute
    "gradient": 1e-6,     # normwise relative
}
FLOOR_FACTOR = 10.0
PROBE_NUDGE = 1e-15


def pin_one_cpu():
    """Pin this process to one logical CPU, as the benchmark does. Returns a description."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    if hasattr(os, "sched_setaffinity"):
        cpu = sorted(os.sched_getaffinity(0))[0]
        os.sched_setaffinity(0, [cpu])
        return f"cpu {cpu}"
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        mask = ctypes.POINTER(ctypes.c_size_t)
        kernel.GetProcessAffinityMask.argtypes = (wintypes.HANDLE, mask, mask)
        kernel.SetProcessAffinityMask.argtypes = (wintypes.HANDLE, ctypes.c_size_t)
        process = kernel.GetCurrentProcess()
        allowed, system = ctypes.c_size_t(), ctypes.c_size_t()
        if not kernel.GetProcessAffinityMask(process, ctypes.byref(allowed), ctypes.byref(system)):
            raise ctypes.WinError(ctypes.get_last_error())
        selected = allowed.value & -allowed.value
        if not kernel.SetProcessAffinityMask(process, selected):
            raise ctypes.WinError(ctypes.get_last_error())
        return f"cpu {selected.bit_length() - 1}"
    return "unpinned (unsupported platform)"


# --------------------------------------------------------------------------
# worker: evaluate one case at every parameter set and write an .npz
# --------------------------------------------------------------------------
def worker(args):
    t_start = time.perf_counter()
    pinned = pin_one_cpu() if args.pin else "unpinned"
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ["JAX_ENABLE_X64"] = "true"
    simulator = Path(args.simulator).resolve()
    sys.path.insert(0, str(simulator))
    import jax
    import jax.numpy as jnp
    import numpy as np
    import differometor as df

    if simulator not in Path(df.__file__).resolve().parents:
        raise SystemExit(f"imported differometor from {df.__file__}, expected {simulator}")
    sys.path.insert(1, str(HERE))
    from cases import CASES, fuzz_parameters

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_enable_compilation_cache", False)
    timings = {"import": time.perf_counter() - t_start}

    t = time.perf_counter()
    case = CASES[args.worker]()
    solver = os.environ.get("HARNESS_SIGNAL_SOLVER")
    extra = {"signal_solver": solver, "pack_noise": solver == "packed"} if solver else {}
    arrays, ports, *_ = df.run_build_step(
        case["setup"], case["changing_pairs"], jnp.asarray(case["changing_values"]), case["pairs"], **extra)
    ports = jnp.asarray(ports)
    jax.block_until_ready((arrays, ports))
    timings["build"] = time.perf_counter() - t

    if args.params:
        parameter_sets = np.load(args.params)["parameters"]
    else:
        parameter_sets = fuzz_parameters(case, args.samples, args.seed, args.scale)
    points = [parameter_sets]
    if args.probe:
        # Lengths are excluded: a relative nudge on a 4000 m length is a physical phase change, not round-off.
        signs = np.random.default_rng(args.seed + 1).choice([-1.0, 1.0], parameter_sets.shape)
        is_length = np.array([(p[0] if isinstance(p[0], list) else p)[1] == "length" for p in case["pairs"]], bool)
        nudge = np.where(is_length, 0.0, signs * PROBE_NUDGE * np.maximum(np.abs(parameter_sets), 1.0))
        points.append(parameter_sets + nudge)

    def readout(outputs):
        carrier, signal, noise = outputs
        powers = df.power_detector(carrier)
        nfreq = signal.shape[1]
        noise2 = noise.reshape(-1, nfreq)
        has_signal = noise2.shape[0] > 0 and ports.shape[0] > 0 and case["changing_pairs"][0][0] == "f"
        if has_signal:
            sig = df.signal_detector(carrier, signal)[ports]
            detected = sig[0] if sig.shape[0] == 1 else sig[0] - sig[1]
            sensitivity = noise2 / jnp.abs(detected)[None, :]
            loss = jnp.mean(jnp.log10(sensitivity))
        else:
            sensitivity = jnp.zeros((0, nfreq))
            loss = jnp.sum(powers[ports]) if ports.shape[0] else jnp.sum(powers)
        return loss, (sensitivity, powers)

    fwd = jax.jit(lambda p, a: jax.vjp(lambda q: df.simulate(**dict(a, optimized_parameters=q)), p))
    sens = jax.jit(lambda o: jax.vjp(readout, o, has_aux=True))
    bwd = jax.jit(lambda ps, pr, c: ps(pr(c)[0])[0])
    compiled = {}
    timings["compile"] = 0.0

    def stage(name, function, *values):
        # Compile once (timed separately), then execute and wait.
        if name not in compiled:
            t = time.perf_counter()
            compiled[name] = function.lower(*values).compile()
            timings["compile"] += time.perf_counter() - t
        return compiled[name](*values)

    def evaluate(params):
        params = jnp.asarray(params)
        outputs, pb_sim = jax.block_until_ready(stage("forward", fwd, params, arrays))
        loss, pb_read, (sensitivity, powers) = jax.block_until_ready(stage("sensitivity", sens, outputs))
        gradient = jax.block_until_ready(stage("gradient", bwd, pb_sim, pb_read, jnp.asarray(1.0)))
        return jax.device_get(dict(carrier=outputs[0], signal=outputs[1], noise=outputs[2], sensitivity=sensitivity,
                                   powers=powers, loss=loss, gradient=gradient))

    results = []
    warm = []
    for group in points:
        rows = []
        for params in group:
            t = time.perf_counter()
            rows.append(evaluate(params))
            seconds = time.perf_counter() - t
            if "first_run_total" not in timings:
                timings["first_evaluation"] = seconds - timings["compile"]
                timings["first_run_total"] = time.perf_counter() - t_start
            else:
                warm.append(seconds)
        results.append({k: np.stack([r[k] for r in rows]) for k in rows[0]})
    timings["warm_evaluations"] = warm
    timings["warm_median"] = statistics.median(warm) if warm else None

    for k, v in results[0].items():
        if not np.all(np.isfinite(v)):
            bad = [i for i in range(len(v)) if not np.all(np.isfinite(v[i]))]
            raise SystemExit(f"{args.worker}: non-finite {k} at parameter set(s) {bad}")
    if not np.any(results[0]["gradient"] != 0):
        raise SystemExit(f"{args.worker}: gradient is identically zero")
    payload = dict(parameters=parameter_sets, **results[0])
    if args.probe:
        payload.update({f"probe_{k}": v for k, v in results[1].items()})
    meta = dict(timings=timings, pinned=pinned, differometor=str(Path(df.__file__).resolve().parent),
                signal_solver=solver, jax=jax.__version__, python=platform.python_version())
    np.savez_compressed(args.out, meta=json.dumps(meta), **payload)


# --------------------------------------------------------------------------
# comparison helpers
# --------------------------------------------------------------------------
def errors(a, b):
    """Errors of output dict `a` against reference `b` for one parameter set."""
    import numpy as np
    out = {}
    for k in ("carrier", "signal"):
        x, y = a[k], b[k]
        denom = np.maximum(np.linalg.norm(y, axis=0), 1e-300)
        out[k] = float(np.max(np.linalg.norm(x - y, axis=0) / denom)) if y.size else 0.0
    p, q = a["powers"], b["powers"]
    mask = q > 1e-12 * max(q.max(), 1e-300) if q.size else q.astype(bool)
    out["powers"] = float(np.max(np.abs(p[mask] - q[mask]) / q[mask])) if mask.any() else 0.0
    for k in ("noise", "sensitivity"):
        x, y = np.abs(a[k]), np.abs(b[k])
        if y.size and np.all(y > 0):
            out[k] = float(np.max(np.abs(np.log(x) - np.log(y))))
        else:
            out[k] = float(np.max(np.abs(x - y))) if y.size else 0.0
    out["loss"] = float(abs(a["loss"] - b["loss"]))
    g, h = a["gradient"], b["gradient"]
    out["gradient"] = float(np.linalg.norm(g - h) / max(np.linalg.norm(h), 1e-300))
    return out


def tolerance(quantity, floor):
    """Return (tolerance, ill_conditioned) for one quantity at one parameter set."""
    if floor > MAX_TOL[quantity]:
        return FLOOR_FACTOR * floor, True
    return min(MAX_TOL[quantity], max(MIN_TOL[quantity], FLOOR_FACTOR * floor)), False


def point(data, i, prefix=""):
    return {k: data[prefix + k][i] for k in QUANTITIES}


def run_case(name, out, simulator, *, pin, samples=0, seed=0, scale=1.0, probe=False, params=None):
    cmd = [sys.executable, __file__, "--worker", name, "--out", str(out), "--simulator", str(simulator),
           "--samples", str(samples), "--seed", str(seed), "--scale", str(scale)]
    cmd += (["--probe"] if probe else []) + ([] if pin else ["--no-pin"]) + (["--params", str(params)] if params else [])
    t = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or proc.stdout).strip().splitlines()[-15:])
        return False, time.perf_counter() - t, tail
    return True, time.perf_counter() - t, ""


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True).stdout.strip()


def export_simulator(ref):
    """Extract differometor/ at `ref` into tests/.tmp/simulator-<sha>/ and return (path, sha)."""
    sha = git("rev-parse", "--verify", f"{ref}^{{commit}}")
    target = TMP / f"simulator-{sha[:12]}"
    if not (target / "differometor" / "__init__.py").exists():
        archive = subprocess.run(["git", "-C", str(ROOT), "archive", "--format=tar", sha, "differometor"],
                                 capture_output=True, check=True).stdout
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(target, filter="data")
    return target, sha


def working_tree_state():
    try:
        return dict(commit=git("rev-parse", "HEAD"), branch=git("rev-parse", "--abbrev-ref", "HEAD"),
                    dirty=bool(git("status", "--porcelain", "--", "differometor")))
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {}


# --------------------------------------------------------------------------
# modes
# --------------------------------------------------------------------------
def save(names, args):
    import numpy as np
    from cases import FUZZ_VERSION
    simulator, sha = export_simulator(args.reference_ref)
    print(f"Reference simulator: {args.reference_ref} = {sha[:12]} ({simulator})", flush=True)
    meta_path = REF_DIR / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if meta and (meta.get("commit"), meta.get("fuzz_version")) != (sha, FUZZ_VERSION):
        print("Existing references used another simulator or fuzz version; replacing them.")
        meta = {}
    meta.update(reference_ref=args.reference_ref, commit=sha, fuzz_version=FUZZ_VERSION, samples=args.samples,
                seed=args.seed, scale=args.scale, probe_nudge=PROBE_NUDGE)
    meta.setdefault("cases", {})
    failed = []
    for name in names:
        out = TMP / f"{name}_reference.npz"
        ok, sec, msg = run_case(name, out, simulator, pin=args.pin, samples=args.samples, seed=args.seed,
                                scale=args.scale, probe=True)
        if not ok:
            print(f"[SAVE-FAIL] {name}\n{msg}", flush=True)
            failed.append(name)
            continue
        data = dict(np.load(out))
        n = len(data["parameters"])
        floors = [errors(point(data, i, "probe_"), point(data, i)) for i in range(n)]
        worker_meta = json.loads(str(data["meta"]))
        np.savez_compressed(REF_DIR / f"{name}.npz", parameters=data["parameters"],
                            **{k: data[k] for k in QUANTITIES})
        meta["cases"][name] = dict(floors=floors, timings=worker_meta["timings"], pinned=worker_meta["pinned"],
                                   jax=worker_meta["jax"], python=worker_meta["python"],
                                   created=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        meta_path.write_text(json.dumps(meta, indent=1))
        out.unlink()
        worst = {k: max(f[k] for f in floors) for k in QUANTITIES}
        print(f"[saved] {name:18s} {sec:6.1f}s  {n} points  worst floor: "
              + " ".join(f"{k}={v:.1e}" for k, v in worst.items()), flush=True)
    print("\nREFERENCES SAVED" if not failed else f"\nSAVE FAILED: {', '.join(failed)}")
    return failed


def check(names, args):
    import numpy as np
    from cases import FUZZ_VERSION
    meta_path = REF_DIR / "meta.json"
    if not meta_path.exists():
        raise SystemExit("No references in tests/reference/. Run: python tests/check_general.py --save")
    meta = json.loads(meta_path.read_text())
    if meta.get("fuzz_version") != FUZZ_VERSION:
        raise SystemExit(f"References use fuzz version {meta.get('fuzz_version')}, harness uses {FUZZ_VERSION}. "
                         "Regenerate with --save.")
    simulator, candidate = ROOT, working_tree_state()
    if args.candidate_ref:
        simulator, sha = export_simulator(args.candidate_ref)
        candidate = dict(ref=args.candidate_ref, commit=sha)
        print(f"Candidate simulator: {args.candidate_ref} = {sha[:12]} ({simulator})", flush=True)
    report = dict(created=datetime.now(timezone.utc).isoformat(timespec="seconds"), suite=args.suite,
                  signal_solver=args.signal_solver, candidate=candidate,
                  platform=platform.platform(), processor=platform.processor(),
                  reference={k: meta[k] for k in ("reference_ref", "commit", "fuzz_version", "samples", "seed", "scale")},
                  cases={})
    failed = []
    for name in names:
        if name not in meta["cases"]:
            print(f"[FAIL ] {name:18s} no reference (run --save --cases {name})")
            failed.append(name)
            report["cases"][name] = dict(passed=False, error="no reference")
            continue
        ref_path = REF_DIR / f"{name}.npz"
        out = TMP / f"{name}.npz"
        ok, sec, msg = run_case(name, out, simulator, pin=args.pin, params=ref_path)
        if not ok:
            print(f"[FAIL ] {name:18s} crashed\n{msg}", flush=True)
            failed.append(name)
            report["cases"][name] = dict(passed=False, error=msg)
            continue
        ref, new = dict(np.load(ref_path)), dict(np.load(out))
        floors = meta["cases"][name]["floors"]
        timings = json.loads(str(new["meta"]))["timings"]
        ref_timings = meta["cases"][name]["timings"]
        per_point, bad, ill = [], [], set()
        for i in range(len(ref["parameters"])):
            err = errors(point(new, i), point(ref, i))
            entry = {}
            for k, e in err.items():
                tol, loose = tolerance(k, floors[i][k])
                ill.update([k] if loose else [])
                entry[k] = dict(error=e, tolerance=tol, ill_conditioned=loose, passed=bool(e <= tol))
                if not e <= tol:
                    bad.append(f"point {i} {k}: {e:.2e} > tol {tol:.1e}")
            per_point.append(entry)
        worst = {k: max(p[k]["error"] for p in per_point) for k in QUANTITIES}
        speedup = ref_timings["first_run_total"] / timings["first_run_total"]
        warm_speedup = (ref_timings["warm_median"] / timings["warm_median"]
                        if ref_timings.get("warm_median") and timings.get("warm_median") else None)
        report["cases"][name] = dict(passed=not bad, failures=bad, worst_error=worst, points=per_point,
                                     timings=timings, reference_timings=ref_timings,
                                     first_run_speedup=speedup, warm_speedup=warm_speedup)
        warm_text = f"warm {timings['warm_median']:.3f}s ({warm_speedup:.1f}x)" if warm_speedup else ""
        print(f"[{'FAIL ' if bad else 'ok   '}] {name:18s} first {timings['first_run_total']:6.1f}s "
              f"({speedup:4.1f}x)  {warm_text}  max err: " +
              " ".join(f"{k}={v:.1e}" for k, v in worst.items()), flush=True)
        if ill:
            print("         ill-conditioned at some points (loose gate): " + ", ".join(sorted(ill)))
        for b in bad:
            print("         " + b)
        if bad:
            failed.append(name)
        out.unlink()

    passed = [c for c in report["cases"].values() if c.get("passed")]
    report["passed"] = not failed
    report["totals"] = dict(first_run=sum(c["timings"]["first_run_total"] for c in passed),
                            reference_first_run=sum(c["reference_timings"]["first_run_total"] for c in passed))
    Path(args.output).write_text(json.dumps(report, indent=1))
    print(f"\nReport: {args.output}")
    print("ALL PASS" if not failed else f"FAILED: {', '.join(failed)}")
    return failed


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--save", action="store_true", help="write references with the simulator at --reference-ref")
    ap.add_argument("--suite", choices=("scored", "all"),
                    help="scored = benchmark fixture only (check default), all = every case (save default)")
    ap.add_argument("--cases", help="comma-separated subset; overrides --suite")
    ap.add_argument("--reference-ref", default="main", help="git ref of the unmodified simulator (save mode)")
    ap.add_argument("--candidate-ref", help="check mode: test differometor/ from this git ref instead of the working tree")
    ap.add_argument("--samples", type=int, default=4, help="fuzzed parameter sets per case, besides the nominal one")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.0, help="multiplies every fuzz width in cases.FUZZ_RULES")
    ap.add_argument("--output", default="benchmark-harness.json", help="check-mode JSON report")
    ap.add_argument("--signal-solver", help="pass signal_solver=... to run_build_step (branches that support it)")
    ap.add_argument("--no-pin", dest="pin", action="store_false", help="do not pin workers to one CPU")
    # worker-only arguments
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    ap.add_argument("--out", help=argparse.SUPPRESS)
    ap.add_argument("--simulator", help=argparse.SUPPRESS)
    ap.add_argument("--params", help=argparse.SUPPRESS)
    ap.add_argument("--probe", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:
        worker(args)
        return
    if args.signal_solver:
        if args.save:
            raise SystemExit("--signal-solver applies to check mode; references always use the reference defaults")
        os.environ["HARNESS_SIGNAL_SOLVER"] = args.signal_solver   # inherited by worker subprocesses

    sys.path.insert(0, str(HERE))
    from cases import CASES, SUITES
    if args.cases:
        names = args.cases.split(",")
    else:
        args.suite = args.suite or ("all" if args.save else "scored")
        names = SUITES[args.suite]
    unknown = [n for n in names if n not in CASES]
    if unknown:
        raise SystemExit(f"unknown case(s): {', '.join(unknown)}; known: {', '.join(CASES)}")
    REF_DIR.mkdir(exist_ok=True)
    TMP.mkdir(exist_ok=True)
    failed = save(names, args) if args.save else check(names, args)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
