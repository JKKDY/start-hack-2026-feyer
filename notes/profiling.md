# Profiling findings — 2026-11-21 (branch `chris/profiling`)

Machine: Intel Core Ultra 7 255H laptop, WSL2, pinned to 1 logical CPU (same as the benchmark).
Nothing in `differometor/` has been changed. All prototypes are monkeypatched from outside the package.
Raw outputs: `runs/prof_compile.txt`, `runs/prof_solve.txt`, `runs/prof_structure.txt`, `runs/prof_schur.txt`.

## 1. Baseline: local vs official

| Phase | Official Xeon 8468 (s) | Local laptop (s) |
|---|---:|---:|
| Imports | 2.61 | 0.25 |
| Init + load + build | 1.05 | 0.68 |
| Forward: trace/lower | 2.01 | 0.98 |
| **Forward: compile** | **14.95** | **10.22** |
| **Forward: execute (+ AD residuals)** | **13.29** | **15.64** |
| Sensitivity (lower + compile + run) | 0.40 | 0.29 |
| Gradient: trace/lower | 0.37 | 0.27 |
| **Gradient: compile** | **5.18** | **3.53** |
| **Gradient: execute** | **4.34** | **3.73** |
| **Total** | **44.20** | **35.60** |
| Compile total (lower + compile) | 22.9 | 15.3 |
| Warm runtime median | 17.7 | 15.3 |
| Max abs log-sensitivity error | 5.4e-6 | 5.4e-6 |

Things to note:
- The laptop **compiles about 1.5× faster** than the Xeon (higher single-core clock) but **executes the
  forward pass slower** (15.6 s vs 13.3 s, probably memory bandwidth: the forward pass writes about 9.6 GB of
  AD residuals, see H3). So the official split is even more compile-heavy than ours. Compile savings
  count for roughly 1.5× more on the scoring machine.
- **Imports: 0.25 s locally vs 2.61 s officially.** `import differometor` itself costs about 30 ms (plotly about 23 ms);
  `import jax` is about 0.75 s even under load. The official figure is most likely a cold OS file cache on
  the first run (thousands of `.py`/`.so` files read from disk). We can only influence that by importing less.
- Run-to-run spread is about ±10 % (warm forward varied 10.7–13.3 s), so compare close candidates repeatedly.

### The `start_here.py` "5.7 s compile" observation
`start_here.py` compiles simulation + gradient as a single `jit(value_and_grad)`, but **it does not
pin the CPU**. Measured on this laptop:

| `start_here.py` | Compile | Execute |
|---|---:|---:|
| unpinned (16 cores) | 4.96 s | **200.5 s** (!) |
| pinned to 1 CPU | 13.65 s | 9.84 s |

So the low compile number comes from XLA compiling in parallel over 16 cores. It says nothing about the program
structure. The judged machine is pinned to 1 CPU, so it doesn't apply. (The 200 s unpinned execution is
thread oversubscription under WSL. The same effect made the harness 6× slower until it was pinned.)

What *is* real, at equal pinning (from `prof_compile.py fused` vs `bench`):

| Structure (1 CPU) | Compile | Execute | Peak RSS |
|---|---:|---:|---:|
| benchmark: `jit(vjp)` → readout → separately compiled backward | 16.1 s | 15.7 s | 15.0 GB |
| fused `jit(value_and_grad)` (start_here) | 14.0 s | 9.9 s | 6.8 GB |

The split structure costs about 2 s more compile and **about 6 s more execution**, because the forward program has
to *export* every AD residual as an output (9.6 GB written to memory). We can't change `benchmark.py`,
but a `custom_vjp` lets us choose what the residuals are (see H3).

## 2. Generality harness (plan step 0b) — `tests/check_general.py`

Moved to [docs/correctness-harness.md](../docs/correctness-harness.md). That document now covers the case list, the tolerance rule and its calibration, fuzzed parameter sets, timings, references generated from `main`, and results for this branch. The earlier single-point floors agree with the new nominal floors. For example, `uifo3_finesse` has a sensitivity floor of 6.6e-7 and `voyager_fm` 2.7.

## 3. Hypotheses

### H1 — `vmap(lax.switch)` bloats compile: **CONFIRMED, but only together with static indices**

`vmap(switch)` with a batched index is lowered to *evaluate all 19 branches + `select`*. Only 3/19
(carrier), 11/19 (signal) and 3/19 (noise) functions are actually used:

```
carrier:  77 evaluations  — laser×7, surface×54, space×16
signal:  679 evaluations  — 11 functions (space, space_modulation(_lower), signal, susceptibility, force_*, corrected_optomech_*…)
noise:    66 evaluations  — vacuum×8, loss×54, squeezer×4
```

Prototype (`profiling/prof_compile.py`, `grouped_update`): integer index arrays become compile-time constants,
and each *used* function is `vmap`ped once over its own slice. Results are **bit-identical**
(gradient difference 3e-23).

| Variant (benchmark 3-stage structure) | fwd compile | grad compile | compile total | StableHLO ops fwd / grad | exec total |
|---|---:|---:|---:|---:|---:|
| baseline (`bench`) | 10.78 | 3.74 | **16.1** | 10 839 / 5 470 | 15.7 |
| static indices only | 9.48 | 3.78 | 14.8 | 10 764 / 5 470 | 17.8* |
| **static + grouped** | **5.94** | **1.86** | **8.9** | 5 475 / 2 746 | 16.4* |

\* within run-to-run noise of the baseline; H1 is a compile-time fix only, as predicted.

Without AD (forward only): 4.90 s (args), 2.57 s (static), 2.99 s (static + grouped). Grouping halves the program
size (6 931 → 3 494 ops) but matters most for the **AD program**, which differentiates every branch.

*Why static indices are needed:* with the indices as traced arguments, XLA sees "gather/scatter at unknown
positions" and has to keep every branch. As constants, it knows the exact positions and which function feeds
which entry. CFD analogy: compiling a solver for one fixed mesh connectivity rather than a generic
unstructured-mesh kernel that looks the connectivity up at runtime.

*Implementation note:* `benchmark.py` passes the whole `arrays` dict as a jit **argument**, so the static
treatment has to come from the build output itself, e.g. `run_build_step` wraps integer index arrays in a
registered pytree node whose aux_data (hashable: dtype + shape + bytes) carries the values. Float data
(matrices, `parameters`, `changing_values`) must stay dynamic.

### H2 — signal matrix factorised 3× per frequency: **PARTLY CONFIRMED (2×, not 3×)**

Isolated, batched 101 × n×n complex128, 1 CPU (`prof_solve.py`):

| n | 1× `solve` | current pattern (solve M, solve Mᴴ, solve M) | `lu_factor` + 3× `lu_solve` |
|---:|---:|---:|---:|
| 751 | 2.90 s | 5.32 s | **2.86 s** |
| 612 | 1.54 s | 2.92 s | 1.62 s |
| 264 | 0.20 s | 0.37 s | 0.23 s |

XLA already shares the LU between the two solves with the same `M` (common-subexpression elimination). Only the `Mᴴ` solve pays for a
second factorisation. Reusing one LU (with `trans=2` for `Mᴴ`) saves **about 2.5 s of forward
execution**, agrees to 5e-16, and compiles no slower. Dense LU runs at about **39 GFLOP/s** on one core, near
peak, so there is no kernel speed left to find. Only reducing n (H4) or the number of factorisations helps.

### H3 — default AD through the solves is wasteful: **STRONGLY CONFIRMED**

The forward pullback carries **817 residual arrays totalling 9.6 GB**: at least 6 batched 101×751×751
complex128 copies of about 0.91 GB each (matrices, LU factors, conjugate-transposed copies). Peak RSS is 15 GB.

| | compile | execute | peak RSS |
|---|---:|---:|---:|
| forward with residual export (`bench`) | 10.8 s | 12.6 s | 15.0 GB |
| forward only, no AD (`fwd_only`) | 4.9 s | 7.3 s | 5.1 GB |
| forward only, static indices | 2.6 s | 7.3 s | 5.0 GB |

So about **5 s of forward execution and about 6 s of forward compile** go purely into building the AD tape. A
`custom_vjp` around the linear-solve core that stores only the LU factors + pivots (1 × 0.91 GB) and the
solutions brings the forward program close to the `fwd_only` row. The backward pass then reuses those factors:
for `x = M⁻¹b`, `b̄ = M⁻ᴴ x̄` (one triangular-solve pair, no new factorisation) and `M̄ = −b̄ xᴴ`.
This is the discrete adjoint you know from CFD shape optimisation: one extra *transposed* solve with the
*same* operator, rather than differentiating through the factorisation algorithm.

### H4 — exploit sparsity / static condensation: **PREMISE WRONG, IDEA STILL PROMISING**

Sparsity: signal matrix 751×751 has **2 553 nonzeros (0.45 %)**, carrier 680 (0.83 %), noise 272.
The rows are *not* what the brief assumed. Spaces are already folded into the component rows by
`build.py`. There are no separate "input = phase × output" rows:

```
upper sideband rows 0–285:   all unit diagonal; off-diagonals per row {0:12, 1:12, 2:142, 3:120}
lower sideband rows 286–571: same
signal/mech rows 572–750:    {0:71 (GW sources), 1:54 (z = χF), 8:48 / 16:6 (force sums)}
max row degree 16, max column degree 8
```

Structural elimination (rule: unit diagonal, derived purely from the sparsity pattern, so it generalises):

| Rule | scored UIFO n | `uifo_20_5000` n | dense LU flop ratio |
|---|---:|---:|---:|
| ≤1 off-diagonal, zero fill-in | 751 → 612 | 769 → 608 | 1.8–2.0× |
| ≤4 off-diagonals, ≤4 fill-in per step | **751 → 264** | **769 → 244** | **23–31×** |

Measured: one batched LU at n=264 takes 0.20 s, against 2.90 s at n=751.

**Conditioning is the risk.** The 2-norm condition number of the signal matrix is about 1e22. Most of that is unit scaling
(entries span 2e-10 … 6e9); after row/column equilibration it is still **about 1e12–1e13**. That is why the
reference error of 5.4e-6 is not far below the limit. `prof_schur.py` checked that the
block partition itself is sound: a Schur-complement solve over the fill≤4 set agrees with the full pivoted LU to
1e-8 (x) and 1e-14 (adjoint w), identical to the zero-fill partition, so this is the reference's own round-off level. But
D was solved *with* pivoting there (cond(D) = 6e12). A real implementation eliminates in a fixed order
**without pivoting**, and that must be verified on all harness cases, especially `uifo3_finesse`.
CFD analogy: this is a sparse direct solver's symbolic phase (ordering and fill analysis) done once at build
time. The open question is whether the static pivot order is stable, as it is for diagonally dominant
FV matrices but not for general indefinite ones.

### H5 — lazy plotly import: **REJECTED as a meaningful lever** (about 30 ms locally)
Free to do, but it won't move the score. See the import note in §1.

## 4. Where the time would go after H1 + H2 + H3 (rough estimate, local)

| Phase | now | after H1+H2+H3 (est.) |
|---|---:|---:|
| imports + build | 0.9 | 0.9 |
| forward lower + compile | 11.2 | ~3 (static fwd-only: 0.46 + 2.6) |
| forward execute | 15.6 | ~5 (7.3 fwd-only − 2.5 LU reuse) |
| sensitivity | 0.3 | 0.3 |
| gradient lower + compile | 3.8 | ~1–2 |
| gradient execute | 3.7 | ~1 (2 triangular-solve sets + entry-function VJPs) |
| **total** | **35.6** | **~11–13** |

After that, the remaining forward time is about 3 s of dense LU at n=751. H4 (n → ~264) would cut it to about
0.3 s, which suggests about 8 s total. These are estimates: measure each step.

## 5. Recommendation (order)

1. **H1: static index pytree + per-function grouping.** Smallest, mechanical, already prototyped
   (`grouped_update` in `profiling/prof_compile.py`), bit-identical, **−7 s compile locally (≈ −10 s on the
   Xeon)**. Also removes the NaN-in-discarded-branch hazard.
2. **H3 + H2 together: `custom_vjp` around the solve core with one LU per frequency reused for the signal solve,
   the Mᴴ noise-weight solve, the covariance solve *and* the backward pass.** This is the biggest lever: about −5 s forward
   execute, −6 s forward compile, gradient execute 3.7 s → ~1 s, peak memory 15 → ~5 GB.
   Validate with the harness (its gradients run through exactly this pullback path) and the benchmark FD check.
3. **H4: structural condensation**, only after 1 + 2 are in. Start with the zero-fill rule (safe, 1.8×),
   then try the fill-limited elimination and check stability on `uifo3_finesse` / `uifo3_20_5000`.
4. Remaining: build step (0.7 s), lazy plotly import, and XLA flags after asking the organisers.

## 6. Gotchas discovered along the way
- `differometor.simulate` (the attribute) is the **function**, not the module, because `__init__` re-exports it.
  To monkeypatch module globals use `sys.modules["differometor.simulate"]`.
- JAX caches traces per function object: lowering the same `jit(f)` twice after monkeypatching a global
  silently reuses the old trace. Use a fresh process (or a fresh function) per variant.
- Always pin to 1 CPU for timing (`taskset -c 0` or the benchmark's `--cpus 1`): unpinned numbers are
  meaningless here in both directions (parallel compile ↓, oversubscribed execution ↑↑).

## 7. Questions for the Feyer team
- Is setting `XLA_FLAGS` from inside `differometor/` allowed?
- How different can the final setup be: new component types, other frequency counts, several detectors?
  (The harness already covers multiple detectors, qhd, AM/FM modulation, sweeps.)
- Is calling SciPy (e.g. sparse LU via `pure_callback` + `custom_vjp`) acceptable, or must everything
  be JAX ops? (Relevant only if a sparse direct solver beats dense LU on the condensed system.)
