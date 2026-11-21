#!/usr/bin/env python3
"""Generality harness: compare the current simulator against saved references.

    python tests/check_general.py --save            # once, with the UNMODIFIED simulator
    python tests/check_general.py                   # after every change
    python tests/check_general.py --cases aligo_am,uifo3_jitter

Each case runs in a fresh subprocess (fresh compile, bounded memory) through the
same three-stage structure benchmark.py uses: jit(vjp(simulate)) -> jit(readout vjp)
-> jit(backward). So custom VJPs / pytree tricks are exercised exactly as scored.

In --save mode each case is additionally evaluated with every optimised parameter
(except lengths) perturbed by 1e-15 relative (random sign; a few ulps) - roughly the backward error a
stable solver injects. (1e-13 was tried first: it overestimated floors ~100x on the UIFOs.) The resulting output change is a per-case conditioning probe (the
legitimate round-off floor); tolerances in check mode are
clamp(FLOOR_FACTOR * floor, MIN_TOL, MAX_TOL); quantities whose floor already
exceeds MAX_TOL are ill-conditioned and get a loose FLOOR_FACTOR * floor gate (flagged).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REF_DIR = HERE / "reference"

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


# --------------------------------------------------------------------------
# worker: evaluate one case and write an .npz
# --------------------------------------------------------------------------
def worker(case_name, out_path, perturb):
    # 1 CPU like the benchmark; unpinned runs were ~6x slower (thread oversubscription)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:1])
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ["JAX_ENABLE_X64"] = "true"
    t0 = time.perf_counter()
    import jax
    import jax.numpy as jnp
    import numpy as np
    import differometor as df

    sys.path.insert(0, str(HERE))
    from cases import CASES

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_enable_compilation_cache", False)

    case = CASES[case_name]()
    solver = os.environ.get("HARNESS_SIGNAL_SOLVER")
    extra = {"signal_solver": solver, "pack_noise": solver == "packed"} if solver else {}
    arrays, ports, *_ = df.run_build_step(
        case["setup"], case["changing_pairs"], jnp.asarray(case["changing_values"]), case["pairs"], **extra)
    params = np.asarray(case["initial"])
    if perturb:
        # lengths are excluded: a relative nudge on a 4000 m length (1e-13 -> 4e-10 m ~ 0.14 deg of phase)
        # a physical change rather than round-off
        signs = np.random.default_rng(0).choice([-1.0, 1.0], params.shape)
        is_length = np.array([(p[0] if isinstance(p[0], list) else p)[1] == "length" for p in case["pairs"]], bool)
        params = params + np.where(is_length, 0.0, signs * 1e-15 * np.maximum(np.abs(params), 1.0))
    params = jnp.asarray(params)
    arrays["optimized_parameters"] = params
    ports = jnp.asarray(ports)

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
    outputs, pb_sim = fwd(params, arrays)
    sens_fn = jax.jit(lambda o: jax.vjp(readout, o, has_aux=True))
    loss, pb_read, (sensitivity, powers) = sens_fn(outputs)
    bwd = jax.jit(lambda ps, pr, c: ps(pr(c)[0])[0])
    gradient = bwd(pb_sim, pb_read, jnp.asarray(1.0))

    host = jax.device_get(dict(carrier=outputs[0], signal=outputs[1], noise=outputs[2], sensitivity=sensitivity,
                               powers=powers, loss=loss, gradient=gradient))
    seconds = time.perf_counter() - t0
    for k, v in host.items():
        if not np.all(np.isfinite(v)):
            raise SystemExit(f"{case_name}: non-finite {k}")
    np.savez(out_path, seconds=seconds, **host)


# --------------------------------------------------------------------------
# comparison helpers
# --------------------------------------------------------------------------
def errors(a, b):
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


def run_case(name, out, perturb=False):
    cmd = [sys.executable, __file__, "--worker", name, "--out", str(out)] + (["--perturb"] if perturb else [])
    t = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or proc.stdout).strip().splitlines()[-15:])
        return False, time.perf_counter() - t, tail
    return True, time.perf_counter() - t, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true", help="write references (use the unmodified simulator!)")
    ap.add_argument("--cases", help="comma-separated subset")
    ap.add_argument("--worker")
    ap.add_argument("--out")
    ap.add_argument("--perturb", action="store_true")
    ap.add_argument("--signal-solver", help="pass signal_solver=... to run_build_step (branches that support it)")
    args = ap.parse_args()

    if args.signal_solver:
        os.environ["HARNESS_SIGNAL_SOLVER"] = args.signal_solver   # inherited by worker subprocesses
    if args.worker:
        worker(args.worker, args.out, args.perturb)
        return

    import numpy as np
    sys.path.insert(0, str(HERE))
    from cases import CASES
    names = args.cases.split(",") if args.cases else list(CASES)
    REF_DIR.mkdir(exist_ok=True)
    tmp = HERE / ".tmp"
    tmp.mkdir(exist_ok=True)
    floors_path = REF_DIR / "floors.json"
    floors = json.loads(floors_path.read_text()) if floors_path.exists() else {}
    failed = []

    for name in names:
        if args.save:
            ok, sec, msg = run_case(name, REF_DIR / f"{name}.npz")
            ok2, sec2, msg2 = run_case(name, tmp / f"{name}_perturbed.npz", perturb=True) if ok else (False, 0, "")
            if not (ok and ok2):
                print(f"[SAVE-FAIL] {name}\n{msg or msg2}")
                failed.append(name)
                continue
            ref = dict(np.load(REF_DIR / f"{name}.npz"))
            floors[name] = errors(dict(np.load(tmp / f"{name}_perturbed.npz")), ref)
            print(f"[saved] {name:18s} {sec:6.1f}s (perturbed {sec2:5.1f}s)  floor: "
                  + " ".join(f"{k}={v:.1e}" for k, v in floors[name].items()), flush=True)
            floors_path.write_text(json.dumps(floors, indent=1))
            continue

        ok, sec, msg = run_case(name, tmp / f"{name}.npz")
        if not ok:
            print(f"[FAIL ] {name:18s} crashed\n{msg}")
            failed.append(name)
            continue
        ref = dict(np.load(REF_DIR / f"{name}.npz"))
        new = dict(np.load(tmp / f"{name}.npz"))
        err = errors(new, ref)
        bad, ill = [], []
        for k, e in err.items():
            floor = floors.get(name, {}).get(k, 0.0)
            if floor > MAX_TOL[k]:
                # ill-conditioned quantity (e.g. near-zero detected signal): loose gate, flagged
                tol = FLOOR_FACTOR * floor
                ill.append(k)
            else:
                tol = min(MAX_TOL[k], max(MIN_TOL[k], FLOOR_FACTOR * floor))
            if not e <= tol:
                bad.append(f"{k}: {e:.2e} > tol {tol:.1e}")
        speed = f"{sec:6.1f}s (ref {float(ref['seconds']):6.1f}s)"
        print(f"[{'FAIL ' if bad else 'ok   '}] {name:18s} {speed}  " +
              " ".join(f"{k}={v:.1e}" for k, v in err.items()), flush=True)
        if ill:
            print("         ill-conditioned (loose gate): " + ", ".join(ill))
        for b in bad:
            print("         " + b)
        if bad:
            failed.append(name)

    print("\nALL PASS" if not failed else f"\nFAILED: {', '.join(failed)}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
