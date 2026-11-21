# Feyer "Faster Physics" — START Hack Munich 2026 — project brief

Handoff from a planning session. Read this first; it holds the context, constraints, baseline numbers, hypotheses and plan. **Profiling was done on 2026-11-21 (branch `chris/profiling`); full results in `notes/profiling.md`. Hypothesis status is marked in §6.**

## 1. Goal and scoring

Make one complete evaluation of the public Differometor simulator (Python + JAX, CPU) as fast as possible:
`python benchmark.py --cpus 1 --output <name>.json`

- **Prize metric:** the *complete first-run total* — imports, CPU init, input load/build, **all tracing + compilation**, forward simulation, sensitivity, reverse-mode gradient, host transfer. Validation and report/plot generation are outside the timer. Warm runtime is reported separately but is not the prize metric.
- 1st/2nd prize = lowest/second-lowest first-run total among correct submissions. Special prize = "most surprising solution".
- Judged on matching hardware: Intel Xeon Platinum 8468, **pinned to 1 logical CPU**, 32 GB RAM, no GPU. Multithreading won't help.
- Hackathon runs 21–22 Nov 2026 (~24 h of work time).

### Correctness rules (must all hold — benchmark prints `PASS`)
- Full setup, all **101 frequencies** (800–3000 Hz, log-spaced), all **193 design variables**, **float64 / complex128**.
- Sensitivity positive at every frequency and `abs(ln(sens) - ln(ref)) <= 1e-5`.
- Carrier, signal, noise, sensitivity, powers, loss, gradient all finite; gradient not identically zero.
- Finite-difference check on `mb11.tuning` (steps 1e-3 and 5e-4, rtol 1%) must pass.
- No precomputed answers, no omitted calculations. Fixture JSON files must stay byte-identical (SHA-checked).
- Only edit `differometor/`. Don't modify `benchmark.py` (its hash is recorded in the report).

### ⚠️ Generality requirement (from the challenge provider)
> "The goal is a simulator speedup that works beyond a specific setup. So the final evaluation will use a similar optical setup like the one included in the GitHub repo, but with slightly different parameters and components. Your simulator submissions should still work with that slight setup variation."

Consequences for every change:
- Everything must be **derived from the loaded setup at build/trace time** — never hand-derived for this 3×3 UIFO. No hard-coded sizes (286, 751, 193, 101), port positions, detector counts, component lists, or "this fixture only uses functions X, Y".
- Code paths this fixture never exercises (e.g. laser amplitude/frequency modulation functions, multiple detectors, carrier-changing parameter sweeps, empty index arrays) must still work.
- Don't rely on this fixture's numerical headroom (current max log error 5.4e-6 vs 1e-5 limit). Different parameters may give worse-conditioned matrices (higher-finesse cavities) → keep pivoted LU and float64; no approximations.
- The benchmark only checks the scored fixture, so a **generality test harness is mandatory** (see plan step 0b).

### Submission
Code, before/after JSON reports + plots, reproduction commands, short note on approach and correctness checks.

## 2. Environment (user's machine)

Laptop with 32 GB RAM, Windows → **WSL2 Ubuntu 24.04** (ships Python 3.12).
- `C:\Users\<user>\.wslconfig` must contain `[wsl2]` / `memory=24GB` / `swap=16GB` (default WSL cap of 16 GB is too tight; baseline peaks ~14.4 GB). Then `wsl --shutdown`.
- Repo lives in the Linux home dir (`~/start-hack-munich-2026`), not `/mnt/c` (slow I/O).
- Setup:
  ```bash
  sudo apt install -y python3.12-venv git
  python3.12 -m venv .venv && source .venv/bin/activate
  pip install --upgrade pip && pip install -r requirements.txt && pip install -e .
  ```
- Pinned: jax/jaxlib 0.9.0, numpy 2.5.5, optax 0.2.6.
- Laptop timing hygiene: plugged in, performance power mode, heavy apps closed; rerun close comparisons (thermal throttling). Compare before/after on the same machine only — absolute numbers will differ from the Xeon.

### Workflow
- **Always pin to 1 CPU when timing** (`--cpus 1`, or `taskset -c 0` + `OMP_NUM_THREADS=1`). Unpinned runs mislead in both directions: XLA compiles in parallel (start_here.py: 5.0 s unpinned vs 13.6 s pinned), and execution oversubscribes threads under WSL (200 s vs 9.8 s).
- Generality harness: `python tests/check_general.py` (13 cases, ~4 min, compares against `tests/reference/` saved from the unmodified simulator; `--cases a,b` for a subset). Must print ALL PASS after every change. References are git-ignored; regenerate with `--save` from an unmodified checkout (commit c10e851).
- Profiling scripts live in `profiling/` (`prof_compile.py <variant>`, `prof_solve.py`, `prof_structure.py`, `prof_schur.py`).
- Work on a git branch (`git checkout -b fast`); commit each experiment separately.
- Fast check: `python benchmark.py --cpus 1 --repeats 3 --output runs/<name>.json` (~2–3 min; full 10 repeats ≈ 5 min).
- Save baseline arrays once: `--save-arrays runs/baseline.npz`; compare every candidate's `.npz` against it (carrier, signal, noise, sensitivity, powers, gradient) with tight tolerances, not just the PASS check.
- Keep a running log of each change → first-run total, compile total, warm median, max log error.

## 3. Baseline (official, Xeon, 1 CPU) — first-run phases

| Phase | s |
|---|---:|
| Imports | 2.61 |
| Init + load + build setup | 1.05 |
| Forward: trace/lower | 2.01 |
| **Forward: compile** | **14.95** |
| **Forward: execute (+ AD residuals)** | **13.29** |
| Sensitivity: lower + compile + run | 0.40 |
| Gradient: trace/lower | 0.37 |
| **Gradient: compile** | **5.18** |
| **Gradient: execute** | **4.34** |
| **Total** | **44.20** |

Compile total 22.9 s (≈52%). Warm median 17.7 s. Max log-sensitivity error 5.4e-6 (limit 1e-5 — note there is not a huge amount of numerical headroom).

**Local baseline (Chris's laptop, Core Ultra 7 255H, WSL2, `--cpus 1`; `runs/baseline.json`):** total 35.6 s — imports 0.25, build 0.68, fwd lower 0.98, **fwd compile 10.22**, **fwd exec 15.64**, sens 0.29, grad lower 0.27, **grad compile 3.53**, **grad exec 3.73**. Compile total 15.3 s, warm median 15.3 s, same 5.4e-6 log error. The laptop compiles ~1.5× faster than the Xeon but executes the forward slower → compile savings weigh ~1.5× more on the scoring machine. Official imports 2.61 s vs 0.25 s locally is most likely a cold file cache (`import differometor` itself is only ~30 ms).

## 4. Repo map

- `differometor/simulate.py` (602 lines) — **hot path**. `simulate()` ~L152–414; `solve()` L33 (`jnp.linalg.solve`, casts to complex128); `update()` L62–111 (matrix-entry evaluation via `vmap(lax.switch(i, FUNCTIONS, x))`); `run_build_step()` L416.
- `differometor/build.py` (1812 lines) — NumPy build step: setup graph → parameter vector + index arrays (which parameters feed which function, where outputs go in which matrix). Runs once, untimed-ish (inside the 1.03 s "build" phase). Carrier/signal "changing parameter" split ~L1505–1760.
- `differometor/components.py` (1387 lines) — physics functions; `FUNCTIONS` list (19 entries: laser, surface, space, space_modulation, susceptibility, force_calculation_*, corrected_optomechanical_phase_*, vacuum/loss_quantum_noise, squeezer, …); `signal_detector` L831, `power_detector` L809; constants (LAMBDA=1064e-9, F0=c/λ, H_PLANCK, UNIT_VACUUM=1, X_SCALE=1e-9, EPSILON_0C=1).
- `differometor/__init__.py` — imports `plot.py` → **plotly at import time** (unused by benchmark).
- `benchmark.py` — compiles 3 stages separately via `jax.jit(...).lower().compile()`: (1) `forward_with_pullback` = `jax.vjp(simulate)`, (2) sensitivity readout vjp, (3) backward applying both pullbacks. Disables the persistent compilation cache. Sets `JAX_ENABLE_X64`, CPU platform, thread env vars, CPU affinity.
- `media/documentation.pdf` — 99-page thesis by the authors explaining the physics and implementation (ch. 2.3 and 3.3 are the relevant parts).
- Data: `examples/data/uifo_800_3000.json` (setup), `uifo_800_3000_test_data.json` (101 frequencies + reference sensitivities).

## 5. Physics / maths summary

Frequency-domain, steady-state, plane-wave interferometer simulator (follows Finesse). Every port field is a complex unknown; components give linear couplings; solve `M E = E_in`.

**Scored setup:** 3×3 UIFO (quasi-universal interferometer): 48 mirrors, 6 beam splitters, 3 directional beam splitters, 71 spaces (0–4000 m), 7 lasers (main ≈200 W & 182 W), 4 squeezers, 54 free masses (0.14–172 kg), 71 GW signal injections (36 at phase 0°, 35 at 180°), 1 signal detector + 1 quantum-noise detector on the same port.
**193 design variables:** reflectivity ×54, tuning ×54, mass ×54, laser power ×7, squeezer dB ×4, squeezer angle ×4, length ×16 (4 of them tied groups of 3).
Distances are split into macroscopic length L (multiple of λ) + microscopic tuning (degrees; 360° = λ) for float precision. Some reflectivities are ~1e-9 (effectively absent) or ~1−2e-10 (extremely high finesse).

**Pipeline inside `simulate()`:**
1. **Carrier solve** — matrix 286×286 (+ augmented RHS column). Solved **once** (carrier doesn't depend on GW frequency).
2. **Signal solve** — per GW frequency Ω (101): matrix **751×751** = 286 upper-sideband + 286 lower-sideband (stored conjugated) + 71 signal rows + 108 optomechanics rows (force F and displacement z per free mass; susceptibility −1/(mΩ²)). GW modulation entries are scaled by carrier fields. Transfer function at detector: `T = conj(E_c)·a_upper + E_c·a_lower`.
3. **Quantum noise** — `S ∝ s^H M^{-1} V_i M^{-H} s` (MISO method), computed without inverses as: `w = solve(M^H, s)` → `u = V_i w` → `c = solve(M, u)` → `Re(s^H c)`. V_i = input noise covariance (vacuum ½, loss L/2, squeezer blocks coupling upper/lower sidebands). Noise ASD = `sqrt(2·S·u·h·f0·0.25)`.
4. **Sensitivity** = noise / |T| (≈1.3e-25 … 1.8e-24 /√Hz). **Loss** = mean(log10(sens/ref)) ≈ 1e-7 (fixture *is* the reference). Powers = ½|E_c|².
5. **Gradient** — reverse-mode AD of the scalar loss w.r.t. 193 parameters. For `x = M^{-1} b`: `b̄ = M^{-H} x̄`, `M̄ = −b̄ x^H` (adjoint method).

Output shapes: carrier (286, 1), signal (751, 101), noise (101,).

## 6. Hypotheses (profiled 2026-11-21 — see `notes/profiling.md` for numbers)

> **Status H1: CONFIRMED (needs static indices too).** Prototype (static int indices + per-function grouping, `profiling/prof_compile.py`) is bit-identical and cuts compile 16.1 → 8.9 s locally (fwd 10.8 → 5.9, grad 3.7 → 1.9). Static indices alone: only 16.1 → 14.8. Used functions: carrier 3/19, signal 11/19, noise 3/19. Exec unchanged.

**H1 — compile time is bloated by `vmap(lax.switch)` in `update()`.** With a batched index, `vmap` of `switch` lowers to "evaluate all 19 branches for every entry, then select". This is applied to every carrier/signal/noise matrix entry and vmapped over 101 frequencies, and reverse-mode AD needs a backward version of every branch too → large HLO graph (forward compile 15 s; likely contributes to the 5.2 s gradient compile). Runtime waste is minor (~1e8 flops vs ~3e11 for the LUs) — the win, if any, is compile time. Other graph bloat candidates: big scatters (`.at[].set`) on 751×751 matrices, conjugation block updates, noise einsums — confirm with an HLO dump.
- **Fix:** the build step already knows which entry uses which function → group entries per function (`np.unique` over `function_indices` at trace time) and call each used function once via `vmap(f)` on its own slice; scatter results into place. Only used functions get compiled; works for any setup because grouping is derived from the build output.
- **Catch:** `benchmark.py` passes the whole `arrays` dict into `jax.jit` as an *argument*, so inside `simulate()` every index array is a tracer (no `np.unique` possible), and `benchmark.py` can't be changed. Workaround inside `differometor/`: have `run_build_step()` wrap index arrays in a custom pytree class registered with `jax.tree_util.register_pytree_node` that stores them as **static aux_data** (must be hashable — e.g. hash/eq on dtype+shape+bytes). `jit` then treats them as compile-time constants, which also lets XLA simplify gathers/scatters. Must cover every index array build.py produces, including empty ones.
- **Bonus:** evaluating all branches on dummy inputs risks inf in discarded branches (e.g. −1/(mΩ²) with m=0) → 0·inf = NaN in the backward pass (classic JAX `where`/select gotcha). Grouping removes that hazard.

> **Status H2: PARTLY CONFIRMED — 2 factorisations, not 3** (XLA already shares the LU of the two `M` solves; only `Mᴴ` refactorises). Isolated: current pattern 5.3 s vs `lu_factor` + 3× `lu_solve` 2.9 s (agree to 5e-16) → saves ~2.5 s forward exec. Dense LU runs at ~39 GFLOP/s on one core (near peak).

**H2 — the 751×751 signal matrix is LU-factorised 3× per frequency in the forward pass** (signal solve, `M^H` noise-weight solve, covariance solve), because `jnp.linalg.solve` refactorises each call. One `lu_factor` + three `lu_solve`s (`trans=2` for the conjugate-transpose solve) does the same maths. Back-of-envelope: complex LU at n=751 ≈ 1e9 flops; 3×101 ≈ 3e11 flops ≈ most of the 13.3 s forward execute. Verify by timing the solves in isolation.

> **Status H3: STRONGLY CONFIRMED.** Pullback = 817 residual arrays, 9.6 GB (≥6 × 0.91 GB batched 751² copies), peak RSS 15 GB. Forward with residual export: compile 10.8 s / exec 12.6 s; forward without AD: 4.9 s / 7.3 s (2.6 s compile with static indices). Single fused `value_and_grad` at 1 CPU: 14.0 s compile / 9.9 s exec / 6.8 GB — the 3-stage split itself costs ~6 s exec because residuals must be exported.

**H3 — default AD through solves is wasteful.** It stores large residuals (~0.9 GB per batched 101×751×751 complex128 copy; peak ~14 GB) and builds a big backward graph (5.2 s gradient compile + 4.3 s execute). A `jax.custom_vjp` implementing the adjoint rule — keep only LU factors (+ solutions), backward = transposed solves with the same factors + outer-product cotangent for M, then pulled back through the matrix-entry functions — should cut gradient compile, execute and memory. Must still pass the finite-difference check.

> **Status H4: premise about space rows WRONG, idea still promising.** Signal matrix 0.45 % dense (2 553 nnz), carrier 0.83 %. `build.py` already folds spaces into component rows: per sideband, rows have {0:12, 1:12, 2:142, 3:120} off-diagonals (all unit diagonal); signal/mech rows {0:71 sources, 1:54 z=χF, 8:48 / 16:6 force sums}. Zero-fill elimination only gives 751 → 612 (1.8×). Allowing ≤4 fill per step: 751 → 264 (23×; uifo_20_5000: 769 → 244, 31×). Signal matrix cond₂ ≈ 1e22, ≈ 1e12–1e13 after equilibration → an unpivoted static elimination order must be validated numerically (Schur partition with pivoted D checked OK: 1e-8 vs full LU, same as reference round-off). The "space rows" text below is outdated.

**H4 — the matrix is very sparse** (each port couples to a few others; likely <1% nonzeros — not yet counted). Exact static condensation / Schur complement at build time can shrink n: signal rows are fixed sources (move to RHS); space equations just relate a port's input to another port's output via a phase (substitute); optomechanical F/z rows can likely be folded in. LU cost ∝ n³, so halving n ≈ 8× cheaper factorisation. **Must be algorithmic** — detect eliminable unknowns from the sparsity/structure produced by build.py for whatever setup is loaded; never hand-derive eliminations for this topology.

**Condensation details (H4):**
- Maths: split unknowns into keep **x** / eliminate **y**: `[[A, B], [C, D]] [x; y] = [f; g]` → `y = D⁻¹(g − Cx)`, `(A − B D⁻¹ C) x = f − B D⁻¹ g` (Schur complement). Only worthwhile when D is trivial (identity/diagonal/unit-diagonal single-coupling rows). Exact, not an approximation.
- Eliminable row types (in priority order):
  1. **Space rows** — `E_in,B − exp(−iΔω nL/c)·E_out,A = 0`: unit diagonal + exactly one off-diagonal entry; each input port is fed by ≤1 space. ~Half of the field unknowns per sideband. **Zero fill-in** (pure column substitution) — start here.
  2. **Signal-source rows (71)** — fixed amplitudes, i.e. known values → move their columns to the RHS.
  3. **Optomechanics rows (F, z per free mass; 108)** — `z = χF`, `χ = −1/(mΩ²)`, F = sum over that mirror's port fields → fold into the mirror's rows; small local fill-in only.
- Rough size estimate (unverified, count real row types in profiling): 751 → ≈290 unknowns ⇒ ≈(751/290)³ ≈ 17× cheaper per LU; combined with factor-once (H2) ≈ 50× less forward LU work.
- **Catches:**
  - *Fill-in:* eliminating an unknown couples all its neighbours; never eliminate hubs. Use a structural rule (e.g. "unit diagonal + exactly one off-diagonal nonzero") based on which entries are nonzero, not on their values → generalises to setup variants.
  - *Full outputs required:* benchmark returns full carrier (286) and signal (751×101) — reconstruct eliminated unknowns by back-substitution after the reduced solve (cheap elementwise).
  - *Noise + adjoint consistency:* noise sources entering eliminated rows become RHS contributions; the M^H solve uses the conjugate transpose of the same reduced matrix (Schur(M)^H = Schur(M^H)). Easy to get subtly wrong → verify with the generality harness.
  - *Gradients:* elimination is just phase multiplications → autodiff-friendly; the custom VJP (H3) then operates on the smaller matrix.
- CFD analogy for Chris: like eliminating ghost/boundary cells that are a direct function of an interior cell before the linear solve, then recovering them afterwards.

> **Status H5: REJECTED as a lever** (~30 ms locally; do it anyway, it's free).

**H5 — import time:** make the plotly/`plot.py` import lazy (~0.5–1 s of the 2.6 s).

**Ruled out:** AMG / iterative solvers (complex, non-symmetric, indefinite, no grid hierarchy; high-finesse cavities → near-singular, iterative convergence ~ number of light round trips; n=751 too small; tolerance + gradient checks punish inexact solves; while_loops add compile time). Deleting near-transparent components (changes results and drops gradients — not allowed). float32 (not allowed).

## 7. Plan (in order) — steps 0, 0b, 1 DONE; recommended next: H1 → (H3 + H2 together as one custom_vjp solve core) → H4 → rest

0. Install, run unmodified baseline locally: `python benchmark.py --cpus 1 --output runs/baseline.json --save-arrays runs/baseline.npz`.
0b. **Build a generality test harness before changing anything** (e.g. `tests/check_general.py`, outside `differometor/`). Using the *unmodified* simulator, save reference outputs (carrier, signal, noise, sensitivity, powers, and gradient of a loss) for several setups: the scored fixture, `examples/data/uifo_20_5000.json`, the aLIGO setup (`examples/aligo.py`, `aligo_test_data.json`), Voyager (`examples/voyager.py`), the cavity example, plus a few **perturbed variants** of the scored UIFO (jittered parameters; if feasible, a component added/removed or a different detector port). After every change, rerun and compare with tight tolerances. A change that passes `benchmark.py` but fails here is not acceptable.
1. **Profile** (≈1 h): dump HLO (`XLA_FLAGS=--xla_dump_to=dump`) to measure forward/gradient program size; time `update()` vs the solves in isolation under jit; count nonzeros in carrier/signal/noise matrices; confirm or reject H1–H4.
2. Quick wins: lazy plotly import (H5).
3. Replace `vmap(switch)` with static per-function grouping (H1). Measure compile + runtime. Check PASS + npz comparison.
4. Factorise once, reuse for all three forward solves (H2). Measure.
5. Custom VJP for the linear solves / whole simulate core (H3). Measure compile, execute, memory, FD check.
6. Static condensation / exploit sparsity (H4) if solves still dominate.
7. Then: XLA flag tuning for compile speed (check with organisers first); remaining compile overhead; build-step (1.0 s) and import trimming.

Rough target if hypotheses hold: 44 s → ~5–10 s first-run total. Post-profiling estimate (local): H1+H2+H3 → ~11–13 s (from 35.6), + H4 → ~8 s.

## 8. Grey areas — ask the Feyer team before relying on them
- Whether calling SciPy (e.g. sparse LU via `pure_callback` + `custom_vjp`) is acceptable vs pure JAX ops.
- Persisting/serialising compiled executables or any compilation cache across runs (they deliberately disabled the cache → assume NOT allowed).
- Setting `XLA_FLAGS` from inside `differometor/` (probably fine, confirm).
- How different the final evaluation setup may be (new component types? different frequency count/range? multiple detectors?) — worth asking so the generality harness covers the right variations.

## 9. Gotchas found while profiling
- `differometor.simulate` (attribute) is the *function* — `__init__` re-exports it and shadows the submodule. Monkeypatch via `sys.modules["differometor.simulate"]`.
- JAX caches traces per function object: re-lowering the same `jit(f)` after monkeypatching a global silently reuses the old trace → one process per variant.
- `pkill -f <pattern>` inside a Bash tool call can kill the calling shell itself (its command line contains the pattern).

## 10. Notes for the person (Chris)
Mechanical engineer with a strong CFD background (Star-CCM+, OpenFOAM) — explain in CFD analogies where helpful (adjoint method, static condensation, sparse vs dense solvers). Wants to understand *why* each change works, not just apply it.
