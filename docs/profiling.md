# Profiling findings

This profile of the original simulator was taken on 21 November 2026 on branch `chris/profiling`. It guided the optimizations recorded in [Performance experiments](performance.md). Each hypothesis below notes what was done with it.

Machine: Intel Core Ultra 7 255H laptop, WSL2, pinned to 1 logical CPU, the same as the benchmark. This is a different machine from the one in the performance tables, so the absolute times here are not comparable with those. The prototypes patched the unmodified package from outside. The scripts are in `profiling/`. Their raw outputs went to `runs/`, which was not committed.

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

### Confirmation on the i7-11370H laptop

The original dense simulator was profiled on 21 November 2026 on the native Windows laptop used in [Performance experiments](performance.md). It ran pinned to one CPU, with each stage compiled separately. Its baseline run took 95.27 s first run.

| Stage | Compile (s) | Warm execution (s) |
| --- | ---: | ---: |
| Carrier parameter and matrix assembly | 2.84 | 0.001 |
| Signal matrix assembly | 4.94 | 1.02–1.10 |
| Signal solve | 0.40 | 2.68–2.84 |
| Noise detector weights and adjoint matrix | 0.15 | 0.53–0.58 |
| Noise adjoint solve | 0.32 | 2.68–2.90 |
| Noise covariance assembly and multiplication | 3.28 | 0.84–1.02 |
| Noise propagation solve and variance reduction | 0.47 | 2.77–2.92 |

The three dense solves took about 8.5 s of the roughly 10.9 s warm forward pass. Because each stage was compiled on its own, XLA could not share the factorization between the signal solve and the noise propagation solve here. In the full program, XLA shares that LU (H2).

The residual cost (H3) was measured separately. A forward-only program compiled in 11.56 s and ran in 6.9–8.5 s warm. The same forward pass exporting its autodiff residuals compiled in 20.97 s, wrote 9.59 GB of residual outputs, and ran in 24.8–33.2 s.

## 2. Hypotheses

### H1 — `vmap(lax.switch)` bloats compile: **CONFIRMED, but only together with static indices**

**Outcome:** implemented as grouped component evaluation (`3e030f4`) and static index arrays (`859f7aa`).

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

**Outcome:** superseded by the sparse solver. One SuperLU factorization per frequency serves both the signal solve and the detector adjoint solve (`5191d2f`).

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

**Outcome:** implemented with `jax.lax.custom_linear_solve` around the sparse solve. JAX then differentiates the linear systems implicitly instead of through the factorization (`5191d2f`). Packed assembly removed the dense batched matrices; the memory figures are in [Structural solver experiments](performance.md#structural-solver-experiments).

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

**Outcome:** sparsity is exploited with a pivoted sparse LU (SciPy SuperLU through `jax.pure_callback`) on the full pattern, not with an unpivoted static condensation. The conditioning risk described here turned out to be real; iterative refinement after every solve addresses it (`31e6aac`). See [Signal-field stability](performance.md#signal-field-stability).

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

**Outcome:** done anyway, since it costs nothing (`bfd1ad2`).
Free to do, but it won't move the score. See the import note in §1.

## 3. Outcome

Before any change, the estimate for H1 + H2 + H3 was about 11–13 s first run on the profiling laptop, falling to about 8 s with H4. The implemented build differs from the plan: sparse LU replaced the dense LU reuse and the condensation. It also added a NumPy build step and folded the fixed setup parameters into the compiled program. That build reaches about 12 s with the original `benchmark.py`, measured on the i7-11370H laptop in [Performance experiments](performance.md#accepted-optimization-history). Compile time, 8.8 s of that total, remains the largest phase.

## 4. Gotchas discovered along the way
- `differometor.simulate` (the attribute) is the **function**, not the module, because `__init__` re-exports it.
  To monkeypatch module globals use `sys.modules["differometor.simulate"]`.
- JAX caches traces per function object: lowering the same `jit(f)` twice after monkeypatching a global
  silently reuses the old trace. Use a fresh process (or a fresh function) per variant.
- Always pin to 1 CPU for timing (`taskset -c 0` or the benchmark's `--cpus 1`): unpinned numbers are
  meaningless here in both directions (parallel compile ↓, oversubscribed execution ↑↑).

## 5. Questions for the Feyer team
- Is setting `XLA_FLAGS` from inside `differometor/` allowed?
- How different can the final setup be: new component types, other frequency counts, several detectors?
  (The harness already covers multiple detectors, qhd, AM/FM modulation, sweeps.)
- Is calling SciPy acceptable, or must everything be JAX ops? This is now relevant: the current build's
  sparse LU calls SciPy SuperLU through `jax.pure_callback`.

## 6. Native CPU profile

On 22 November 2026, the original benchmark at `ad27bcc` ran the native CPU backend with XLA level 0 on one pinned i7-11370H logical CPU. The fresh run took 1.772 s, including about 0.80 s of imports and 0.547 s of tracing and compilation. The warm median was 0.323 s. See [the interleaved XLA flag experiments](performance.md#xla-settings-with-the-native-cpu-backend) for a more stable cold baseline.

- **Sparse factor reuse.** In a profiling run, the solver took 0.122 s forward and 0.123 s backward, out of 0.151 s and 0.184 s in the host callbacks. The two calls used byte-identical packed matrices at all 101 frequencies. A seven-repeat replay captured the actual reverse right-hand sides and compared refactor-and-solve with retained forward factors. Retention reduced the solve median from 0.115 s to 0.044 s and returned identical results with both refinement steps. [The integrated benchmark](performance.md#reuse-forward-superlu-factors) measures the full evaluation with a bounded, exact-match cache.
- **Imports.** In a separate `-X importtime` equivalent trace, JAX took 0.428 s cumulative and `differometor` took 0.197 s, of which 0.190 s was the sparse solver module and its dependencies. The benchmark imports JAX directly. The native path still needs SciPy, so delaying that import to the solve shifts time within the cold evaluation. Matplotlib loaded for report generation after the timer. Direct native sparse linkage or a standalone native runner could reduce dependencies, with the latter measured separately from the scored benchmark.
- **Packed assembly and gradients.** Native component evaluation accounted for 0.009 s forward and 0.017 s backward in the profile. NumPy scatter-add calls accounted for 0.010 s in reverse. Combining gather, component evaluation, and scatter in C++ could remove intermediate arrays and dispatch. The observed costs are smaller than sparse factor reuse, so profile the integrated result before retaining extra native kernels.
