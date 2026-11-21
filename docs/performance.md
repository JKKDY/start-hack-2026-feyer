# Differometor performance experiments

The current build is `baseline-optimized`. It combines the packed sparse solver, static index arrays, a NumPy build step, fixed setup parameters folded into the compiled program, and LLVM `-O0` compilation set inside `differometor/` (see [XLA backend optimization level](#xla-backend-optimization-level)). The original `benchmark.py` passes on it. On the laptop described below, the first run including JIT fell from **63.13 s to about 12 s**, and the warm median from **33.278 s to about 0.31 s**. The [correctness harness](correctness-harness.md) passes all 13 setups at nominal and fuzzed parameters. The strict elementwise comparison of raw signal fields still reports differences in the sparse modes; see [Signal-field stability](#signal-field-stability). Profiling that guided these steps is in [Profiling findings](profiling.md).

## How the optimization decisions evolved

1. The dense reference spent most of its time evaluating the optical system. We rewrote noise as a quadratic form to eliminate a redundant solve, then grouped instructions by component type so JAX could trace only the relevant functions. On the historical pinned i7, the first run went from 63.13 to 55.06 to 44.36 seconds.
2. Compact covariance and shared *dense* LU each helped some measurements, but their cold-run gains were inconsistent. The sparse LU implementation shared one factorization per frequency between the signal and adjoint solves. Packing only occupied signal and noise entries then reduced memory traffic. Across the i7 experiments the fully packed version reached 16.80 seconds cold and 0.227 seconds warm.
3. The sparse solver needed more accuracy for small detector weights, particularly on Voyager. Two iterative corrections using the existing factors recovered accuracy. We retained the corrections despite their cost. Moving setup calculations to NumPy avoided small JAX compilations. Supplying structural indices and fixed parameter values at trace time simplified the differentiated program. The fixed-parameter build recorded 12.09 seconds on the i7.
4. Once the solves were cheap, compiler work dominated. Lower LLVM optimization, batching component columns, removing empty indexed updates, and avoiding setup copies reduced cold time. Changing the CPU fusion emitter helped further. The M1 comparison for commit `76ce14b` was 1.990 to 1.549 seconds cold, with warm evaluation rising from 0.171 to 0.180 seconds. This was a separate machine and baseline.
5. A small output-selection shortcut and a later prototype that evaluated entire fixed groups with NumPy did not establish a reproducible end-to-end win in the active checkout. Neither became the baseline for the experiments below.

## Sweep reuse, coefficient placement and packed index construction

We tested three changes against `76ce14b` with the original benchmark and its compilation cache disabled. Each process evaluated the full workload once cold and ten more times warm. Rows below are medians across fresh processes in varied order. The M1 macOS runs were unpinned (8 logical CPUs, 16 GiB, Python 3.12.16). The Linux ARM64 runs used Docker Desktop on the same M1 and one logical CPU selected with `--cpus 1`. Both used JAX and jaxlib 0.9.0.

**macOS**

| Variant | Fresh runs | First run (s) | Trace/lower/compile (s) | Optical build (s) | Warm (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Control, 76ce14b | 5 | 1.594 | 1.015 | 0.046 | 0.178 |
| Sweep reuse | 3 | 1.559 | 0.986 | 0.046 | 0.177 |
| Direct placement | 3 | 1.530 | 0.954 | 0.047 | 0.175 |
| Packed index construction | 3 | 1.591 | 1.014 | 0.042 | 0.178 |
| Placement + index construction | 3 | 1.593 | 0.974 | 0.042 | 0.173 |
| All three | 3 | 1.548 | 0.955 | 0.043 | 0.174 |
| Fully packed construction, dense fields omitted | 3 | 1.564 | 1.006 | 0.038 | 0.175 |

**Linux ARM64**

| Variant | Fresh runs | First run (s) | Trace/lower/compile (s) | Optical build (s) | Warm (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Control, 76ce14b | 3 | 2.081 | 1.344 | 0.051 | 0.294 |
| Sweep reuse | 3 | 2.075 | 1.385 | 0.053 | 0.292 |
| Direct placement | 3 | 1.970 | 1.337 | 0.051 | 0.289 |
| Packed index construction | 3 | 1.998 | 1.367 | 0.045 | 0.295 |
| Placement + index construction | 3 | 1.981 | 1.350 | 0.046 | 0.290 |
| All three | 3 | 1.929 | 1.308 | 0.046 | 0.291 |

- **Sweep reuse.** During tracing, the simulator checks which group inputs intersect either sweep. Independent groups run once per evaluation and their coefficients broadcast across the sweep. Optimized variables still participate in automatic differentiation. This change alone was nearly flat on Linux.
- **Direct placement.** For unique writes, setup composes the group permutation with the signal destination mapping. When every noise slot is written exactly once, the coefficients arrive directly in canonical noise order. Repeated and partial writes retain their previous semantics. Direct placement delivered the shortest individual macOS median.
- **Packed index construction.** Setup derives the CSC structure from the two carrier blocks, auxiliary diagonal and possible writes. It computes packed destinations by sorted-key lookup and avoids scanning an initially empty noise template. All 13 setups reproduced the previous CSC structure and destination indices exactly. The production version retains the dense matrix fields used by existing callers.
- **Fully packed builder trial.** Omitting the dense compatibility fields reduced the M1 build median to 0.038 s, but changes what callers receive from the builder. We dropped this variant.
- **Decision.** We kept the three compatible changes together. The pinned Linux median went from 2.081 to 1.929 s. Placement plus index construction alone took 1.981 s. Three samples per Linux variant establish a candidate worth keeping, with run-to-run variation large enough that the exact gain on the judging Xeon remains open.

The live checkout passed all 13 generality cases. The worst sensitivity and gradient errors excluding Voyager FM were 1.34e-07 and 5.02e-08. Voyager FM reached 0.0667 and 0.00146, within its established round-off tolerances. The sparse solver unit test passed.

Every isolated prototype and both compatible combinations passed all 13 generality cases. The exact live code also passed the pinned Linux benchmark twice (2.148 and 2.007 s). A separate adjacent control run took 2.075 s. Across the adjacent pair, carrier, signal, noise, sensitivity and powers matched exactly. The largest absolute difference over all 193 gradient components was 1.04e-25. Warm medians in this pair were 0.298 s for the control and 0.309 s for the final version.

## Accepted optimization history

The table tracks cumulative builds on one pinned logical CPU of an Intel Core i7-11370H laptop (8 logical CPUs, 16 GB RAM), with native Windows 10, Python 3.12.10, JAX and JAXLIB 0.9.0, and double precision. Every run used the saved 3×3 UIFO, 101 frequencies, and 193 design variables. Timings are one complete first run and the median of three warm evaluations. The first-run total includes JIT tracing, lowering, and compilation.

| Cumulative build | First incl. JIT (s) | JIT (s) | Warm (s) | Reference sensitivity check, max log error | Gradient spot check, max abs. error | Full gradient relative L2 error | Max per-frequency signal relative L2 error | Strict signal entry mismatches |
| --- | ---: | ---: | ---: | --- | --- | --- | --- | ---: |
| Original dense simulator (`3b538f0`) | 63.13 | 26.91 | 33.278 | Pass (5.38e-6) | Pass (3.83e-7) | 0 / 0 | 0 / 0 | 0 / 0 |
| Noise quadratic form (`4a2c3d0`) | 55.06 | 27.08 | 26.028 | Pass (5.38e-6) | Pass (6.97e-7) | 1.90e-8 / 1.31e-7 | 0 / 0 | 0 / 0 |
| Grouped component evaluation (`3e030f4`) | 44.36 | 14.25 | 18.578 | Pass (5.38e-6) | Pass (6.97e-7) | 1.90e-8 / 1.31e-7 | 0 / 0 | 0 / 0 |
| Detector selections without tiling (`089afa2`) | 41.50 | 15.62 | 18.567 | Pass (5.38e-6) | Pass (6.97e-7) | 1.90e-8 / 1.31e-7 | 0 / 0 | 0 / 0 |
| Shared sparse LU with dense assembly | 34.16 | 13.99 | 8.366 | Pass (5.37e-6) | Pass (1.32e-8) | 2.20e-9 / 1.22e-7 | 2.04e-7 / 1.45e-7 | 3,427 / 3,433 |
| Packed sparse signal assembly | 19.10 | 13.29 | 1.959 | Pass (5.37e-6) | Pass (1.32e-8) | 2.20e-9 / 1.22e-7 | 2.04e-7 / 1.45e-7 | 3,427 / 3,433 |
| Packed signal and noise, benchmark default (`47cc00a`) | 16.80 | 12.66 | 0.227 | Pass (5.37e-6) | Pass (1.32e-8) | 2.31e-9 / 1.22e-7 | 2.04e-7 / 1.45e-7 | 3,427 / 3,433 |
| Static index arrays and refined sparse solves (`859f7aa`, `31e6aac`) | 18.10 | 13.80 | 0.337 | Pass (5.37e-6) | Pass (2.34e-8) | Harness | Harness | Not measured |
| NumPy build step, original `benchmark.py`, packed library default (`663abec`, measured at `71dc5eb`) | 17.11 | 13.91 | 0.327 | Pass (5.37e-6) | Pass (2.34e-8) | Harness | Harness | Not measured |
| Fixed setup parameters folded into the compiled program (`67ceed2`, measured at `6e66e96`) | **12.09** | **8.78** | **0.307** | Pass (5.37e-6) | Pass (2.34e-8) | Harness | Harness | Not measured |

The last three rows were measured on 21 November 2026, from checkouts with LF line endings, with 3 repeats and an external pinning wrapper. `31e6aac` ran with its branch's modified `benchmark.py`, whose `--signal-solver` flag defaults to packed. The other two ran with the original file. Single first runs on this laptop vary by about 2 s: `71dc5eb` measured 15.43, 17.11 and 17.39 s, and `6e66e96` 12.09 and 11.94 s. Differences of 1–2 s between rows are therefore not significant. "Harness" means the full field and gradient comparisons for that row come from the [harness results](#correctness-harness-results), not the paired-point check.

The original row is the fresh local dense calibration. The repository's [published CPU baseline](../benchmarks/cpu-baseline.json) was measured on a different host and recorded 44.20 s first run, 22.92 s JIT, and 17.68 s warm. The fully packed prototype previously recorded 18.34 s first run and 0.248 s warm. The table uses its later verification run as the default.

### How to read the numerical checks

- **Sensitivity.** Maximum absolute difference in natural log sensitivity from the public curve across the 101 frequencies. The supplied limit is `1e-5`.
- **Gradient spot check.** Maximum absolute difference between autodiff and the two central-difference estimates for `mb11.tuning`. Each comparison uses the benchmark's `rtol=0.01, atol=1e-6`. The two finite-difference estimates must also agree within that tolerance.
- **Full-gradient and field comparisons.** Paired values are for the saved parameters and for `mb11.tuning` increased by 0.002, each against the original at that same point. All 193 gradient coordinates passed `rtol=1e-3, atol=1e-7`. Carrier fields and powers matched exactly. Quantum noise and sensitivity arrays matched within `rtol=1e-5, atol=0`. The signal comparison uses the largest relative L2 error across individual frequency vectors and counts entries outside the separate `rtol=1e-5, atol=1e-12` elementwise check. All rows produced finite outputs and gradients with the expected shapes, including the full 751 × 101 signal field.

### Correctness harness results

The harness compares each build with references from the unmodified simulator on 13 setups, at the nominal point and 4 fuzzed parameter sets each. Its timings run each case in a fresh process, like the benchmark. See [Correctness and timing harness](correctness-harness.md) for the method and tolerances.

| Build | Result | `uifo3_scored` harness first run, warm (s) | Largest sensitivity / gradient error, all cases except `voyager_fm` |
| --- | --- | --- | --- |
| `31e6aac`, dense solver | All 13 pass. Carrier, signal and powers bit for bit | 49.2, 19.292 | 7.4e-8 / 1.9e-8 |
| `31e6aac`, packed solver | All 13 pass | 16.9, 0.304 | 1.4e-7 / 4.9e-8 |
| `71dc5eb`, packed default | All 13 pass, same errors as `31e6aac` packed | 15.1, 0.304 | 1.4e-7 / 4.9e-8 |
| `6e66e96`, packed default | All 13 pass, same errors as `31e6aac` packed | 13.1, 0.336 | 1.4e-7 / 4.9e-8 |
| `0869f42` + level 0 (WSL laptop, not comparable with the rows above) | All 13 pass | 6.5, 0.201 | 1.4e-7 / 5.1e-8 |

`voyager_fm` has an almost zero detected signal, so its sensitivity and gradient are set by round-off. The packed path's errors there, 6.7e-2 and 1.2e-3, are within its round-off floors of 2.7 and 0.90.

## Earlier dense experiments

The first structural trials below were tested on top of the quadratic-form baseline. Memory figures are Windows process peaks during the measured workload, before validation.

| Variant | First run (s) | JIT (s) | Warm (s) | Working set (GiB) | Private commit (GiB) | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Compact covariance | 66.60 | 29.94 | 18.73 | 7.12 | 12.89 | Dropped |
| Shared dense signal and adjoint LU factors | 56.67 | 27.38 | 16.87 | 8.01 | 13.76 | Dropped |

Compact covariance, shared LU, and grouped component evaluation were each tested on top of the quadratic-form baseline. The detector selection change was tested on top of grouped evaluation.

- **Noise quadratic form.** Given $A^H u=d$, the code evaluates $\operatorname{Re}(u^H Q u)$ directly. This removes one batched signal-system solve and its backward pass. The original trial showed a warm improvement and mixed first-run results.
- **Compact covariance.** Gathering occupied entries of $Q$ and contracting them with the detector weights lowered warm time and peak private commit. On the cold repeat it took 51.33 s against the control's 53.05 s. The first-run results differed in direction, so the experiment was dropped.
- **Shared LU.** One signal factorization serviced both the signal and adjoint solves, with implicit differentiation through the linear systems. Its warm time improved, but its first runs took 56.67 s and 54.29 s against control runs of 55.06 s and 53.05 s.
- **Grouped component evaluation.** The builder groups calls by function and restores the original coefficient order before scattering. This reduced JIT time by about 47% against the quadratic-form baseline.
- **Detector selections without tiling.** The carrier multiplication broadcasts the sideband selections, then concatenates them with the extra degrees of freedom. The repeat took 40.98 s with the change and 41.22 s without it. Warm performance was effectively unchanged.

## Follow-up JAX experiments

These candidates were tested independently on top of `089afa2`, with the same workload and CPU configuration. Both passed the benchmark and matched all 193 gradients at the saved and perturbed points.

| Variant | First run incl. JIT (s) | JIT (s) | Warm median (s) | Repeat first run incl. JIT (s) |
| --- | ---: | ---: | ---: | ---: |
| Detector selection control | 41.50 | 15.62 | 18.57 | 42.57 |
| Simplified parameter-sweep scatter | 75.19 | 14.81 | 17.10 | 44.35 |
| Checkpointed component coefficients | 36.85 | 15.17 | 19.19 | 48.80 |

Neither change was retained. The sweep scatter was faster warm but slower in both first-run measurements. The retest preserved the existing path for parameter arrays already expanded by a carrier sweep. Checkpointing reduced forward compilation time in the repeat from 9.17 s to 6.19 s, while gradient compilation rose from 5.12 s to 9.17 s. Its first-run improvement did not repeat, and the warm median was slightly slower.

To check repeatability faster, a diagnostic used every fifth frequency from the public grid (21 frequencies, including both endpoints) and all 193 design variables. Each variant ran three times in fresh processes, with a different run order each round and three warm evaluations per process. All reduced-grid physical outputs and gradients matched the control within the experiment tolerances.

| Variant | Median first run incl. JIT (s) | Median JIT (s) | Median warm (s) |
| --- | ---: | ---: | ---: |
| Detector selection control | 17.79 | 12.26 | 2.35 |
| Simplified parameter-sweep scatter | 17.33 | 12.01 | 2.47 |
| Checkpointed component coefficients | 19.56 | 13.75 | 2.36 |

The control's first runs ranged from 17.14 s to 18.17 s. Sweep indexing was faster in two rounds and slower in one, while checkpointing was slower in all three. The reduced grid is a quick repeatability check. The complete 101-frequency workload remains the acceptance benchmark.

## Hybrid topology-dispatch branch

`12cd3f7` ("Optimize topology dispatch and linear solves") is the tip of both `feature/hybrid-optimization` and `feature/hybrid-noise-optimized`. It is built directly on the original simulator. It was run twice on 21 November 2026, with identical simulator code, the original `benchmark.py`, 3 repeats, pinned, on the laptop above. Both runs passed the sensitivity and gradient checks.

| Run | First incl. JIT (s) | JIT (s) | Forward execution (s) | Gradient execution (s) | Warm median (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 51.29 | 15.30 | 27.98 | 4.18 | 16.40 |
| 2 | 43.84 | 14.95 | 14.06 | 11.46 | 15.20 |

Warm repeats ranged from 10.4 to 24.2 s. The branch kept dense solves and was not adopted: the packed sparse build was faster in both first-run and warm time.

## XLA backend optimization level

`benchmark_fast_compile.py`, on `chris/static-indices` at `71b231d`, sets `XLA_FLAGS=--xla_backend_optimization_level=1` and runs that branch's modified `benchmark.py`. The flag lowers LLVM optimization effort. Each row below is one run with 3 repeats, pinned, on the laptop above.

| Build | Flag | First incl. JIT (s) | JIT (s) | Warm (s) |
| --- | --- | ---: | ---: | ---: |
| `71b231d` | none | 19.05 | 14.46 | 0.331 |
| `71b231d` | set directly | 16.85 | 12.90 | 0.332 |
| `71b231d` | through the launcher | 17.96 | 13.70 | 0.325 |
| `67ceed2` | none | 12.05 | 8.89 | 0.309 |
| `67ceed2` | set directly | 11.74 | 8.49 | 0.316 |

The flag mainly shortens gradient compilation. After the fixed parameters were folded in, level 1 gains 0.3 s, which is within run-to-run noise. Warm time is unchanged. `71b231d` itself is not part of `baseline-optimized`.

### Adopted: level 0 set inside `differometor/`

The organisers confirmed on 21 November 2026 that compiler flags are allowed, provided the improvement carries over to other optical setups. Level **0**, not 1, is where the gain is: LLVM's machine-code optimisation dominates compilation, while the heavy linear algebra runs in precompiled LAPACK and SuperLU. `differometor/__init__.py` appends `--xla_backend_optimization_level=0` to `XLA_FLAGS` at import. The original `benchmark.py` imports the package before it initialises the CPU backend, so the flag applies, and the report records it under `xla_flags`. A level already set by the user is kept, and `DIFFEROMETOR_KEEP_XLA_FLAGS=1` disables the behaviour. The change was written on `chris/xla-flags` (`2402e5b`) and applied to `baseline-optimized` on top of `0869f42`.

These times come from **a different machine** from the tables above: Intel Core Ultra 7 255H (16 logical CPUs, 1 used via `--cpus 1`), WSL2 Ubuntu 24.04 with 24 GB allocated, Python 3.12.3, JAX and jaxlib 0.9.0. Do not compare them with the Windows laptop's numbers. Each cell is the median of 3 fresh processes, run interleaved with the control, original `benchmark.py`, `--repeats 3`. The control is the same commit with `DIFFEROMETOR_KEEP_XLA_FLAGS=1`.

| Build (WSL laptop) | First incl. JIT (s) | JIT (s) | Warm (s) | Individual first runs (s) |
| --- | ---: | ---: | ---: | --- |
| `0869f42`, no flag | 6.14 | 5.58 | 0.182 | 6.44, 6.10, 6.14 |
| `0869f42` + level 0 | **3.99** | **3.42** | 0.195 | 4.20, 3.99, 3.99 |

Forward compilation fell from 3.26 s to 1.82 s and gradient compilation from 1.47 s to 0.78 s. Every run passed, with a maximum log-sensitivity error of 5.37e-6 and the gradient spot check passing. The gain carries over: on the same machine, every harness setup's per-case first run was 1.9–2.4 s faster with the flag (1.1 s for `cavity_tuning`), with unchanged errors. Warm time rose by about 0.01 s.

## Batched component evaluation and shorter CPU compilation

On the branch based on `a017042`, component functions read parameter columns on demand and evaluate an entire group in one call. Empty sweeps and absent links bypass their indexed updates. Setup initializes matrix diagonals in place and passes the allocated matrices directly to the packing step. The CPU compiler uses `--xla_cpu_use_fusion_emitters=false` alongside LLVM level 0. Explicit user settings retain precedence.

Apple M1, 16 GiB, eight logical CPUs available, macOS 26.5.2, Python 3.12.16, JAX and jaxlib 0.9.0. Each variant ran in five interleaved fresh processes, unpinned, with ten warm evaluations each. Times are medians in seconds.

| macOS build | First run | Trace/lower/compile | Optical build | Warm |
| --- | ---: | ---: | ---: | ---: |
| Control at `a017042` | 1.990 | 1.358 | 0.050 | 0.171 |
| Batched components and compiler change | 1.549 | 0.978 | 0.045 | 0.180 |

One-process screening runs on the revised solver gave the following first-run, compile and warm times in seconds. Existing LLVM settings alone gave 1.726 / 1.171 / 0.169. Disabling the fusion emitters gave 1.526 / 0.961 / 0.174 and was retained for the repeated comparison. Single-module code generation gave 1.738 / 1.175 / 0.172, disabling constant folding gave 1.755 / 1.193 / 0.169, and disabling expensive LLVM passes gave 1.717 / 1.168 / 0.173. Those three settings were dropped.

Linux ARM64 under Docker Desktop on the same M1, with one logical CPU selected by the original benchmark. One fresh process per version and ten warm evaluations. These times include the VM and filesystem overhead.

| Linux build | First run | Trace/lower/compile | Optical build | Warm |
| --- | ---: | ---: | ---: | ---: |
| Control at `a017042` | 3.938 | 3.188 | 0.064 | 0.293 |
| Batched components and compiler change | 2.454 | 1.617 | 0.061 | 0.300 |

The unchanged benchmark passed on both platforms for the full 101-frequency, 193-variable workload. The macOS generality harness passed all 13 cases, including the nominal and four fuzzed parameter sets. Maximum sensitivity and gradient errors excluding Voyager FM were 1.34e-7 and 5.03e-8. On the ill-conditioned Voyager FM case they were 0.0667 and 0.00146, within the reference tolerance. The sparse solver unit test passed.

## Optional Rust kernel for packed signal coefficients

The “Accelerate packed signal coefficients with an optional Rust kernel” change builds on `1988487`. It generates complex-double coefficient functions from the existing optical equations and reuses frequency-independent intermediates within each sweep. A CPU XLA foreign-function call evaluates the coefficients and a second entry point computes their analytic reverse derivatives. The packed scatter, sparse solve, readout and objective still use the existing code. On an explicitly CPU-only run, dispatch goes directly to the selected implementation, avoiding tracing an unused backend branch.

To reproduce on macOS or Linux, install the project dependencies, Cargo and a C++17 compiler, then run `python -m differometor.rust_kernel.build` once before starting the original benchmark. A fresh wheel includes the build sources. The simulator automatically uses a compatible local binary on CPU and falls back to JAX when one is absent. Set `DIFFEROMETOR_DISABLE_NATIVE=1` to reproduce the control. The native build took 7.669 s on macOS and 8.675 s on Linux, with a toolchain available. Include that cost if evaluating a fresh source build. Eligibility for a separate installation step in challenge scoring is unconfirmed.

Each column below gives the median of five fresh processes with ten warm repeats per process, alternating control and candidate order. Both environments used an Apple M1 host with 16 GiB RAM and eight logical CPUs, Python 3.12.16, JAX and jaxlib 0.9.0, x64 enabled and the persistent compilation cache disabled. Native macOS 26.5.2 used default affinity. The ARM64 Linux Docker benchmark used one pinned logical CPU.

| Seconds | macOS JAX | macOS Rust | Linux JAX | Linux Rust |
| --- | ---: | ---: | ---: | ---: |
| Complete first run | 1.6212 | 1.1195 | 1.9985 | 1.4686 |
| Setup, including imports | 0.4198 | 0.3803 | 0.3231 | 0.3257 |
| Simulator build (within setup) | 0.0442 | 0.0405 | 0.0465 | 0.0472 |
| Trace, lower and compile | 0.9789 | 0.5610 | 1.3562 | 0.8344 |
| Forward lower and compile | 0.6830 | 0.3846 | 0.8721 | 0.5446 |
| Gradient lower and compile | 0.2739 | 0.1424 | 0.4385 | 0.2403 |
| Warm total | 0.1751 | 0.1736 | 0.2992 | 0.2881 |
| Warm forward | 0.0878 | 0.0873 | 0.1613 | 0.1530 |
| Warm gradient | 0.0859 | 0.0864 | 0.1371 | 0.1328 |

In the earlier one-CPU Linux screen, the JAX control took 2.012 s cold, 1.367 s compile, and 0.296 s warm (nine processes). Moving both passes to Rust gave 1.589 / 0.919 / 0.301 s (five processes), so we dropped the version that recomputed invariant terms at every frequency. Keeping JAX forward with Rust reverse gave 1.700 / 1.070 / 0.295 s (three processes). Hoisting invariant terms in both Rust passes gave 1.455 / 0.842 / 0.285 s (three processes). We kept that design and added JAX transformation support and backend dispatch. The table reports the final integrated measurements.

The original benchmark passed in all twenty final processes. The full reference suite passed on both platforms, covering 13 setups at their saved values and four perturbations each. At the nominal scored point the relative L2 differences against the JAX control were 3.01e-11 for sensitivity and 5.57e-10 for the full gradient on macOS, and 3.12e-11 and 8.70e-10 on Linux. Carrier and power arrays agreed exactly. Against the committed reference across all five scored-case points, the largest sensitivity and gradient errors were 8.59e-8 and 6.25e-9 across the two platforms. The suite-wide maxima came from the existing ill-conditioned Voyager FM case under its relaxed tolerances. Sensitivity and gradient errors there were 0.0627 and 0.00146 on macOS, and 0.845 and 0.923 on Linux. The prior JAX Linux control also produced 0.846 and 0.925 on that case. The dedicated native test compares every registered coefficient function with JAX, including complex inputs, forward and reverse derivatives, second derivatives and batching.

## Benchmarking further changes

Two tools cover a change:

- **Correctness:** `python tests/check_general.py --suite all` compares carrier, signal, noise, sensitivity, powers and gradients with committed references from the unmodified simulator. It covers 13 setups, each at its saved parameters and at 4 fuzzed parameter sets, and also records timings per case. It pins itself to one CPU on Linux and on native Windows. See [Correctness and timing harness](correctness-harness.md).
- **Scored time:** `benchmark.py`, as below. Its `--cpus` flag works only on Linux or WSL.

Run each candidate and its control in fresh processes on the same hardware and pinned dependency versions. Use the complete first-run total for the main comparison and record JIT, warm runtime, and peak process memory separately. Repeat close first-run comparisons in reverse order. Compare full physical outputs and gradients at both parameter points before retaining a change.

On Linux or WSL, the standard benchmark command is:

```bash
python benchmark.py --cpus 1 --repeats 3 --output benchmark-candidate.json --save-arrays candidate.npz
```

The Windows experiments pinned the Python process externally before importing JAX because the benchmark's `--cpus` flag uses Linux affinity. A local harness recorded process memory and evaluated the second parameter point outside the timed phases. The reference fixture is checked byte for byte, so use a checkout that preserves its LF line endings. The standard command saves arrays for the nominal point.

## Structural solver experiments

The experimental CPU path uses one SuperLU factorization per frequency to solve the signal system and its detector adjoint. JAX differentiates the linear systems implicitly, including the complex transpose solve in the backward pass. Sparse assembly then removes the dense frequency-batched signal matrix. Packing the covariance separately removes the dense noise batch by contracting its occupied entries with the detector weights.

| Cumulative build | Peak working set (GiB) | Peak private commit (GiB) |
| --- | ---: | ---: |
| Noise quadratic form | 7.83 | 15.43 |
| Grouped component evaluation | 8.57 | 15.03 |
| Detector selections without tiling | 8.33 | 15.03 |
| Shared sparse LU with dense assembly | 6.66 | 8.23 |
| Packed sparse signal assembly | 3.85 | 3.89 |
| Packed signal and noise verification run | 0.51 | 0.54 |

In a fresh cold rerun, the dense reference took 42.99 s including JIT and the fully packed mode took 18.05 s.

The table above uses the original simulator as its numerical reference. Against the last optimized dense build, the fully packed prototype's full-gradient relative L2 differences were 1.74×10⁻⁸ at the saved point and 2.53×10⁻⁷ at the perturbed point.

On the 21-frequency grid, the new code's default dense path reproduced the previous physical outputs and loss bit for bit. Its maximum absolute gradient difference was 6.90×10⁻⁹.

`benchmark.py` must stay identical to the original challenge file. The judges run the original, and it calls `run_build_step()` without solver options. The fully packed build is therefore the library default: `signal_solver="packed"`, with `pack_noise` following it. Commit `47cc00a` had added a `--signal-solver` flag to `benchmark.py`, while the library default stayed dense. With the original file, that combination would have scored the dense path. `baseline_plus_sparsity` (`894e6b9`) and `chris/numpy-build` restore the original file and the packed default. Benchmark variants go in new files such as `benchmark_<purpose>.py`. The packed build's acceptance covers the fixed benchmark workload, the supplied checks, and the tested full-gradient comparisons. The mixed-precision refinement below remains a diagnostic.

To compare the packed and dense builds on every harness setup, run `python tests/check_general.py --suite all` and again with `--signal-solver dense`. For the scored time of the dense reference, run the original `benchmark.py` from a checkout of `3b538f0`.

### Signal-field stability

The dense assembly and the reconstructed packed assembly gave identical signal matrices and right-hand sides at all 101 frequencies, both at the saved parameters and with `mb11.tuning` increased by 0.002. The packed layout omitted no nonzero coefficients. The raw matrices have widely differing row scales. Their estimated 1-norm condition numbers were about 10³⁴–10³⁷, and the largest componentwise backward error was 1 for both double-precision solvers, even though their global normwise backward errors were below 1.3×10⁻²⁸.

An independent sparse Gaussian elimination solved the captured binary64 systems at 80 and 110 decimal digits. The two references agreed to within 1.4×10⁻⁷² relative infinity norm across the eight cases. These cases covered both parameter points at 800 Hz, the largest dense-to-sparse field difference at about 821.43 Hz, and detector-response peaks near 1321.97 and 1744.89 Hz. This isolates the linear solve from rounding in the physical coefficient formulas.

At 800 Hz, the exact reference for zero-based field index 238 is zero. The sparse solve returned zero and the dense solve returned 15.35. The dependency graph identified 46 variables with no path from a nonzero right-hand-side entry at every frequency. These variables are exactly zero because the unforced principal block is nonsingular. Across the nominal sweep, 929 of the 3,427 elementwise disagreements were in those variables. Both double-precision solvers also failed strict elementwise comparison with the high-precision reference on driven variables.

A diagnostic solve removed the unforced block, equilibrated the remaining rows and columns, and factored the reduced system in double precision. Equilibration alone still missed 8–10 fields per sampled case at the existing tolerance of `rtol=1e-5, atol=1e-12`. Three iterative refinement steps reused the same factors and evaluated the residual at 110 decimal digits. This eliminated every strict mismatch across the eight reference cases. The worst componentwise backward error was 1.08×10⁻¹⁶ across all 202 refined cases. Only the eight sampled vectors were compared with independent higher-precision solutions.

For the next stability check, require structurally unforced fields to be exactly zero and compare driven fields individually against independent higher-precision references. Use componentwise residuals across the sweep to catch poorly solved rows. The refinement diagnostic covers signal fields. The production solver, adjoint, gradients, and performance of a refined implementation still need separate validation.
