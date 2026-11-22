# Differometor performance experiments

The current build is `baseline-optimized`. It combines the packed sparse solver, static index arrays, a NumPy build step, and fixed setup parameters folded into the compiled program. The original `benchmark.py` passes on it. On the laptop described below, the latest run took **9.243 s** including JIT and **0.272 s** warm. The original dense run took **63.13 s** including JIT and **33.278 s** warm. The [correctness harness](correctness-harness.md) passes all 13 setups at nominal and fuzzed parameters. The strict elementwise comparison of raw signal fields still reports differences in the sparse modes. See [Signal-field stability](#signal-field-stability) and [Profiling findings](profiling.md).

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
| `baseline-optimized` benchmark rerun (`2960b9e`) | **9.243** | **7.856** | **0.272** | Pass (5.37e-6) | Pass (2.34e-8) | Not measured | Not measured | Not measured |

The last four rows were measured on 21 November 2026, from checkouts with LF line endings, with three repeats and an external pinning wrapper. `31e6aac` ran with its branch's modified `benchmark.py`, whose `--signal-solver` flag defaults to packed. The other three ran with the original file. Single first runs on this laptop vary by about 2 s. `71dc5eb` measured 15.43, 17.11 and 17.39 s, and `6e66e96` measured 12.09 and 11.94 s. The `2960b9e` rerun used an LF Git snapshot and ran only the original benchmark. Its three warm repeats ranged from 0.270 to 0.297 s. Treat the result as a single baseline measurement and use paired runs to assess AOT speedups. "Harness" refers to the full field and gradient comparisons in the [harness results](#correctness-harness-results).

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

The flag mainly shortens gradient compilation. After the fixed parameters were folded in, its gain is 0.3 s, which is within run-to-run noise. Warm time is unchanged. The flag is not adopted: the scored run uses the original `benchmark.py`, so it would have to be set inside `differometor/`. The organisers have not said whether that is allowed. `71b231d` itself is not part of `baseline-optimized`.

### Ahead-of-time compilation feasibility

Tracing and compilation accounted for about 85% of the latest [`baseline-optimized` first run](#accepted-optimization-history). The AOT target is one build per supported platform that accepts all geometries supported by the simulator. JAX executable serialization can reload machine code for a compiled input signature. The current simulator also embeds topology indices and fixed parameter values in that signature. A geometry-specific executable or a fallback that compiles new geometries would fail the compile-once requirement. [JAX AOT documentation](https://docs.jax.dev/en/latest/aot.html)

Keep the Python setup API and build a runtime plan with component types, index arrays, sparse patterns, dimensions, sweeps, and detector definitions. A generic native library can allocate buffers and execute that plan with runtime-sized loops. One binary can then evaluate different geometries and differentiate the built-in component operations and implicit solves. Each setup can still require plan construction or sparse symbolic analysis at build time. User-defined Python/JAX parameter links are a separate compatibility requirement. A zero-JIT path would need a way to evaluate both the value and derivative of those links, such as a supported expression representation or explicit differentiable callbacks.

JAX shape-polymorphic export can share tracing and lowering across input shapes, but still compiles for each new concrete shape. A single compiled JAX executable could cover a fixed padded capacity with runtime indices, which imposes limits on the accepted geometries and requires changing the current static plans. For the unrestricted geometry target, Cython can build a Python extension ahead of time and accept typed buffers whose lengths are determined at runtime. It can call the existing sparse library while keeping the setup API in Python. Cython supplies the build and buffer interface. Differometor still needs to implement its derivative rules at the native boundary. [JAX shape polymorphism](https://docs.jax.dev/en/latest/export/shape_poly.html) · [Cython typed memoryviews](https://cython.readthedocs.io/en/latest/src/userguide/memoryviews.html)

A two-process probe on the i7-11370H laptop used one process at a time on native Windows with Python 3.12 and JAX/JAXLIB 0.9.0. A small double-precision `value_and_grad` program compiled and serialized in the first process, then returned the expected value and gradient for a different input after reload in the second process. The CPU loader emitted target-feature warnings about `prefer-no-scatter` and `prefer-no-gather`; the calculation still passed. A small `pure_callback` program executed successfully but failed serialization because its Python callback wrapper could not be pickled. These were feasibility probes. The full simulator and a native FFI call have yet to be tested for executable reload.

The packed solver already factors matrices in native SuperLU through a Python callback. Its matrix dimensions and sparsity patterns are runtime inputs to that callback. The remaining topology-specific JAX graph includes coefficient evaluation, assembly, readout and their derivatives.

A small assembly spike exercised that runtime-size boundary with the actual cavity and scored UIFO builder plans. A single C shared library built with GCC 15.2 under Ubuntu 26.04 WSL was loaded from Python through `ctypes`. It accepted 66 and 3,348 coefficient slots using the same binary, with zero and one noise detector in the respective setups. Five synthetic complex inputs per geometry checked the packed assembly and its two input cotangents against JAX, with a maximum absolute component error of 1.78×10⁻¹⁵. The available WSL Python lacked Cython and development headers, so this tested the native boundary directly. It did not replace a simulator stage or measure a scored speedup.

#### Integrated Cython assembly spike

An isolated candidate at `01a5d4b` replaced the packed signal scatter, carrier coupling and conjugation with one Cython extension. Its reverse-mode rule calls a precompiled Cython backward kernel. The same module handled all topology sizes in the suite. Component evaluation, solves, and the surrounding JAX programs were unchanged. The candidate still uses host callbacks, and the full simulation still compiles.

| Version | First run incl. JIT (s) | JIT (s) | Warm (s) |
| --- | ---: | ---: | ---: |
| Current packed control | 10.95 | 7.86 | 0.293 |
| Cython assembly and backward pass | 10.10 | 7.40 | 0.281 |

Each cell is the median of three fresh runs with three warm evaluations each, using the original `benchmark.py` on the same i7-11370H with one pinned logical CPU, 15.8 GiB RAM, native Windows 10, Python 3.12.10 and JAX/JAXLIB 0.9.0. The extension was built beforehand with Cython 3.1.8 and MSVC 14.43. The candidate's first runs ranged from 10.06 to 10.24 s, versus 10.61 to 11.55 s for the control. The measurements show a modest cold-start gain and similar warm execution on this workload.

All scored runs passed. The candidate's maximum scored log-sensitivity error was 5.37×10⁻⁶. Its nominal gradient differed from the current packed control by 1.97×10⁻⁹ in relative L2. The correctness harness passed all 13 setups and their fuzzed points against the committed references. Its largest sensitivity and gradient errors were 6.75×10⁻² and 1.22×10⁻³ in the known near-null `voyager_fm` case, both within that case's stored round-off tolerances.

Keep this as an isolated experiment. It removes compilation for one part of the graph while leaving about 7.4 s of JIT. The bridge implements reverse-mode differentiation and would need a forward-mode rule before serving as a general replacement. To reproduce it, install Cython, setuptools and a C compiler in the benchmark environment, then prepare an unused directory with `python profiling/cython_assembly/prepare.py <new-directory> --ref 01a5d4b`. Run the original benchmark and correctness harness from that directory. [Reproduction helper](../profiling/cython_assembly/prepare.py)

#### Packed covariance in Cython

The follow-up candidate moved the packed covariance scatter and detector noise contraction, including their reverse pass, into the same extension. It retains the assembly spike and leaves the benchmark original. The two fresh pairs below were measured on the same machine and pinned CPU as the assembly spike. Each warm number is the median of three evaluations.

| Pair | Cython block | First incl. JIT (s) | JIT (s) | Warm (s) |
| --- | --- | ---: | ---: | ---: |
| 1 | Assembly | 8.62 | 7.34 | 0.285 |
| 1 | Assembly and covariance | 8.16 | 6.97 | 0.287 |
| 2 | Assembly | 8.87 | 7.61 | 0.279 |
| 2 | Assembly and covariance | 8.82 | 7.49 | 0.304 |

The covariance variant passed every scored run and all 13 correctness cases. The scored log-sensitivity error was 5.37×10⁻⁶. The largest suite sensitivity and gradient errors remained 6.75×10⁻² and 1.22×10⁻³ in the known near-null `voyager_fm` case, within its reference tolerances. The cold improvement was small, and warm runtime increased in both pairs, so the covariance path remains optional in the spike. Add `--pack-noise` when running the reproduction helper to enable it in the staged checkout.

#### JAX AOT component tiles

A separate application experiment compiled the 19 registered component functions and their reverse-mode pullbacks as a bundle of 38 executables. Each takes a fixed tile of 128 input rows. Python splits runtime-sized batches into as many tiles as needed and pads only the final tile. Matrix dimensions, component counts, and geometry are absent from the executable signatures.

On the same pinned Windows CPU and JAX/JAXLIB 0.9.0 runtime, building the bundle took 6.79 s offline. A fresh process loaded it in 0.73 s, including CPU initialization. It evaluated 7,675 component input rows and 84,425 value and pullback entries from the cavity, scored UIFO, and alternate UIFO builder plans. Values and pullbacks agreed exactly with ordinary JAX references, with no JIT events reported during reload and execution. This is a reusable component library test. The full simulation and geometry assembly have yet to run through this library together.

Run `python profiling/aot_component_tiles.py build <new-directory>`, then `python profiling/aot_component_tiles.py check <same-directory>` in a fresh process with `JAX_LOG_COMPILES=1`. Use a locally produced bundle on the same host and runtime. The scored benchmark never loads the bundle. [AOT probe](../profiling/aot_component_tiles.py)

#### Proposed native engine

The latest baseline spent 5.051 s tracing and compiling the simulator forward pass and 2.493 s on the reverse pass, versus 0.312 s on the objective readout. The proposed engine builds the entire simulator forward and reverse passes into one native extension per target platform. The JAX executable bundle above remains a feasibility experiment.

Keep `Setup`, `run_build_step()` and `simulate(**arrays)` as the public entry points. Prepare a private runtime plan from the builder's component groups, sparse patterns and index maps. Loops in the native engine take matrix sizes, sweep lengths and detector counts from that plan. Geometry and parameter changes require calling the same code with new data.

1. **`forward(plan, inputs) -> optics, state`.** Run parameter updates, component evaluation, packed assembly, carrier and signal solves, and quantum noise in one native call. Compile component formulas together with their callers and write coefficients directly into packed buffers. Reuse pivoted LAPACK and SuperLU solves and preserve refinement.
2. **`pullback(plan, state, optics_cotangents) -> input_gradients`.** Run the reverse pass in one native call. Differentiate the solve equations through transpose solves and accumulate derivatives at the stored coefficient slots. Give each built-in component a local pullback and compare its values and complex derivatives with the current JAX implementation. Callers can use any objective that produces cotangents for the optical outputs.

Use Cython for the Python binding and compile the numerical loops and formulas into the same extension, using C++ where it simplifies native math and solver integration. Keep JAX as the numerical reference and the adapter for caller-defined objectives.

The [builder](../differometor/simulate.py) derives its array keys from the declared argument names of `simulate`. Keep that public signature and put the native dispatch and JAX derivative rules behind it. The [scored benchmark](../benchmark.py) passes the simulation pullback between two separately compiled functions, so the adapter needs an array-based forward state with shapes derived from the plan. A reverse-mode rule limited to `optimized_parameters` is suitable for the initial private candidate. Before switching the default, check the other differentiable inputs and JAX transformations supported by the baseline, as well as solver options and user-defined parameter links.

A CPU probe wrapped the existing SuperLU solve in `pure_callback` and `custom_vjp`. It ran JAX's forward and reverse wrappers as separate executables, held two forward states at once, and consumed them in reverse order. For 2 × 2 and 5 × 5 complex systems with zero and two detector right-hand sides respectively, outputs and pullbacks for coefficients and right-hand sides agreed with dense JAX within 2.48 × 10⁻¹⁶ maximum absolute error. The existing packed solve also passed `jax.jvp`, which the probe's `custom_vjp` adapter rejected. Use this adapter for the first reverse-mode integration and preserve forward-mode support before changing the default. This probe covers the adapter contract at the solve boundary. Full optics parity and performance remain unmeasured for the native engine. [JAX callback documentation](https://docs.jax.dev/en/latest/external-callbacks.html)

The organisers clarified that implementation and build or packaging changes are allowed when the evaluated inputs and outputs are preserved. Build the extension at install time and include extension loading and all first-use work in cold-start measurements. The first integration can reuse the existing SciPy LAPACK and SuperLU bindings while native linkage is evaluated. [Challenge notes](../CLAUDE.md)

Evaluate both passes together. Time a fresh process over the same interval as `benchmark_slim.py`, including extension import and plan construction. Run the cavity and UIFO with the same build, verify the physical outputs, sensitivity and full gradient, and measure warm repeats against a paired baseline. The unchanged scored harness explicitly lowers and compiles three outer JAX wrappers with the persistent cache disabled. Measure it separately through the JAX adapter. [JAX FFI documentation](https://docs.jax.dev/en/latest/ffi.html)

#### Native CPU implementation

The opt-in packed CPU backend runs assembly, the optical solves, quantum noise, and the complete simulator pullback through two host calls. The component formulas and their reverse rules are generated ahead of time from the JAX source and compiled into one C++ extension. NumPy handles packed array operations, and the existing LAPACK and SuperLU paths perform the solves. The Python binding uses the CPython buffer protocol. The JAX wrapper still handles parameter links and the benchmark's readout. No serialized JAX kernels are loaded.

These paired runs used the original `benchmark.py` and three warm repeats per fresh process, on the pinned Windows laptop above on 22 November 2026. The extension was built with MSVC 14.43 before starting the benchmark. The measured total includes loading it.

| Pair | Backend | First incl. JIT (s) | JIT (s) | Warm (s) |
| --- | --- | ---: | ---: | ---: |
| 1 | Packed control | 8.523 | 7.558 | 0.269 |
| 1 | Native CPU | 1.905 | 0.762 | 0.320 |
| 2 | Packed control | 8.852 | 7.786 | 0.286 |
| 2 | Native CPU | 1.785 | 0.754 | 0.338 |

Both native runs passed the scored sensitivity and finite-difference gradient checks. The worst scored log-sensitivity error was 5.37 × 10⁻⁶, and the largest gradient spot-check error was 1.06 × 10⁻⁸. All 13 setups passed the existing tolerances at the saved parameters and four perturbations each. The scored fixture's largest full-gradient error in that suite was 6.2 × 10⁻⁹. One process also ran cavity → UIFO → cavity with a changed parameter through the same loaded extension.

Build with `python -m pip install -e .` using a C++ compiler and set `DIFFEROMETOR_NATIVE=1` before starting the benchmark process. The generated C++ include is part of the source distribution. A clean source archive built and imported on Windows. Regenerate the include with `python profiling/generate_native_components.py` after editing the component formulas. The backend stays opt-in because warm execution is slower and the native adapter currently returns reverse-mode gradients for design parameters. The default path retains forward-mode and other differentiable inputs.

#### Rust signal coefficients and CPU compiler flags

Branch `test/aot-flags-kernel` starts from `2e61add`. We retained the existing low-latency CPU flags (`--xla_backend_optimization_level=0 --xla_cpu_use_fusion_emitters=false`) and adapted the Rust sweep from `ddec3a3` to the AOT host callback. Rust computes packed signal coefficients and their reverse derivative directly from contiguous buffers. It reuses intermediates that are constant across frequency, avoiding a second JAX dispatch. C++ handles other component groups and the existing multi-carrier fallback. The AOT backend retains the upstream reverse-mode design-parameter scope. The normal JAX path remains available for its wider differentiation API.

The original benchmark ran three fresh processes per configuration and ten warm repeats per process. Both platforms used an Apple M1 host with 16 GiB RAM and eight logical CPUs, Python 3.12.16 and JAX/jaxlib 0.9.0. macOS 26.5.2 used default affinity. ARM64 Linux in Docker used one pinned CPU. These are local timings, with variability in imports and setup. Build time was measured separately, 6.8 s on macOS and 7.0 s on Linux for Rust with Cargo installed.

| Platform and configuration | First run (s) | Setup (s) | Compilation (s) | Warm (s) |
| --- | ---: | ---: | ---: | ---: |
| Mac, default JAX before flags | 2.6167 | 0.3971 | 2.0382 | 0.1660 |
| Mac, default JAX with flags | 1.6826 | 0.3737 | 1.1280 | 0.1738 |
| Mac, earlier Rust branch | 1.4794 | 0.6879 | 0.6141 | 0.1811 |
| Mac, original C++ AOT | 0.8572 | 0.3996 | 0.2520 | 0.1891 |
| Mac, C++ AOT with flags | 0.7478 | 0.4237 | 0.1256 | 0.1933 |
| Mac, Rust AOT with flags | 0.7030 | 0.3963 | 0.1211 | 0.1712 |
| Linux, original C++ AOT | 1.0775 | 0.3191 | 0.4407 | 0.3038 |
| Linux, C++ AOT with flags | 0.8192 | 0.3452 | 0.1625 | 0.3134 |
| Linux, Rust AOT with flags | 0.7965 | 0.3213 | 0.1684 | 0.2876 |

The scored benchmark passed in each final comparison. The Rust AOT backend passed all 13 reference setups at five parameter points on both platforms. The all-component unit test checks complex values and pullbacks with constant, partly varying and fully varying inputs. Worst reference-suite errors were in the existing near-null Voyager FM case, under its published tolerances. On macOS sensitivity and gradient errors were 0.0627 and 0.001463. On Linux sensitivity and gradient errors were 0.8453 and 0.9233.
On the scored Mac input, the saved sensitivity, signal and noise arrays matched the C++ AOT values exactly. The relative L2 difference of the full gradient was 2.52e-20.

With the pinned dependencies installed, build the C++ extension with `python -m pip install -e .` and the Rust library with `python -m differometor.rust_kernel.build` (Cargo on macOS or Linux). Run `DIFFEROMETOR_NATIVE=1 python benchmark.py`, adding `--cpus 1` on Linux. To reproduce flagged C++ set `DIFFEROMETOR_DISABLE_NATIVE=1` as well. To reproduce original C++ also set `DIFFEROMETOR_KEEP_XLA_FLAGS=1` with no external XLA flags.

#### One Rust library for ordinary JAX and native AOT execution

The setup changes from `b32d0c2` were reverted in `137df90`, restoring the `82b9785` implementation. The saved plain benchmark had taken 2.292 s. That invocation used the ordinary JAX path. A separate AOT invocation required an environment setting. The acceptance target for this change was 1.4 s total on the developer's M1 with the ordinary benchmark command.

The host build now exports both the direct coefficient interface used by the AOT engine and the XLA foreign-function interface used by the earlier JAX prototype (`ddec3a3`). One library serves both. On CPU, ordinary packed simulation with a compatible library calls Rust to evaluate signal coefficients and their analytic pullback. Other operations still use JAX. The ordinary path retains JAX rules for forward and higher derivatives and batching. Empty parameter links and identity sweeps skip unnecessary work. Packed groups can scatter in their own order when destinations are unique. Repeated destinations retain their original order. The AOT path retains its own instruction ordering. A missing compatible binary falls back to JAX.

| Platform and invocation | Runs | Total (s) | Setup (s) | Build within setup (s) | Compile (s) | Warm (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mac plain, restored | 5 | 1.6740 | 0.3930 | 0.0476 | 1.0995 | 0.1769 |
| Mac plain, shared library before compiler tuning | 5 | 1.1484 | 0.3966 | 0.0418 | 0.5701 | 0.1719 |
| Mac plain, final Apple Silicon setting | 5 | 1.0595 | 0.3718 | 0.0388 | 0.5058 | 0.1671 |
| Mac AOT, restored | 3 | 0.6910 | 0.3903 | 0.0472 | 0.1192 | 0.1830 |
| Mac AOT, shared library | 3 | 0.6714 | 0.3840 | 0.0395 | 0.1055 | 0.1750 |
| Linux plain, restored | 3 | 2.1463 | 0.3129 | 0.0564 | 1.5244 | 0.2953 |
| Linux plain, shared library before compiler tuning | 3 | 1.4086 | 0.3066 | 0.0446 | 0.8100 | 0.2830 |
| Linux plain, final shared library | 5 | 1.3261 | 0.3181 | 0.0473 | 0.6909 | 0.2774 |
| Linux AOT, final shared library | 1 | 0.7273 | 0.3183 | 0.0518 | 0.1141 | 0.2822 |

Medians use fresh processes and ten warm repeats. The M1 host has eight logical CPUs and 16 GiB RAM and runs macOS 26.5.2 with default affinity. ARM64 Linux runs in Docker on one pinned logical CPU on the same host. Both use Python 3.12.16 and JAX/jaxlib 0.9.0 with x64 enabled. The initial separate-library prototype had one Mac run at 1.643 s. The integrated version measured 1.178, 1.166, 1.148, 1.143, 1.134 s. Initialization and imports vary by launch. The main saving was compilation. An extra sparse-buffer reuse variant was dropped after one of five Mac trials took 1.952 s and its one-CPU Linux median was 1.409 s. At JAX optimization level O0, the two Linux screening runs took 1.524 and 1.398 s. Disabling parallel LLVM code-generation splitting on one pinned CPU gave 1.331 and 1.284 s. A final live Mac run before compiler tuning took 1.487 s. Applying the single-split compiler setting on Apple Silicon produced five consecutive plain runs in the actual checkout at 1.047, 1.053, 1.059, 1.223 and 1.072 s. Both accuracy suites and the unit tests passed with this setting. The final default uses a single code-generation split on Apple Silicon and on processes with one-CPU affinity and respects explicit user overrides. The final Linux median was 1.326 s, with one of five plain runs at 1.402 s.

Mac plain passed all 13 setups at five parameter points each. Worst sensitivity and gradient errors were 0.0627 and 0.001463, respectively, in the known near-null Voyager FM case. Linux plain passed all 13 setups at five parameter points each. Worst sensitivity and gradient errors were 0.8453 and 0.9233, respectively, in the known near-null Voyager FM case. The full suites also passed on both native AOT paths and on the Mac with the native library disabled. At the nominal point on Mac the relative L2 differences against the restored plain path were 3.68e-11 for sensitivity and 6.17e-10 for the gradient. Carrier and powers agreed exactly. The coefficient tests exercise complex values, JVPs, VJPs, higher derivatives and batching.

Build the library once per compatible installation with `python -m differometor.rust_kernel.build` (Cargo and a C++17 compiler). The local Mac build took 6.877 s before benchmarking. Use `python benchmark.py` normally or `DIFFEROMETOR_NATIVE=1 python benchmark.py` for the full AOT backend. Use `DIFFEROMETOR_DISABLE_NATIVE=1` without selecting AOT to exercise the JAX fallback. Windows retains the pure JAX option.

## Rust sparse sweep and native CPU pullback

The control is `test/aot-flags-kernel` at `f750e82`. The implementation is on `experiment/rust-packed-solver`. The control already runs Rust optical coefficient kernels and tunes XLA for the CPU. Its default signal solve loops over frequencies in Python, calls SciPy SuperLU, and compiles the surrounding forward and reverse computations with JAX.

The new default for a packed CPU run builds coefficient plans from the supplied topology and evaluates carrier, signal and noise coefficients and their adjoints with the generic Rust library. Rust drives the complex128 sparse sweep. It computes one column ordering per sparsity pattern within the sweep, reuses work buffers, and factors each frequency numerically. A small C++ binding calls SuperLU. Signal and detector solves share those factors, including residual refinement. The fused host pullback consumes the saved forward fields and rebuilds numeric factors during reverse. Python and NumPy still build topology and packed layouts, drive readout, and call LAPACK for the dense carrier. JAX retains the public API and implements forward-mode, higher-order and non-design derivatives through the general path. All setup, library loading and planning occur inside the original timed phases.

### Measured increments

Each cell gives complete first run / tracing and compilation / median warm run in seconds. Each process ran ten warm repeats. Rows are independent screens from different sessions, with the number of fresh processes per platform shown separately. The paired acceptance series below determines the release decision.

| Change | Processes | M1 | Linux, CPU 0 |
| --- | ---: | --- | --- |
| Control | 3 | 1.381 / 0.661 / 0.197 | 1.892 / 1.051 / 0.333 |
| Rust sparse sweep | 3 | 1.286 / 0.729 / 0.172 | 1.822 / 1.087 / 0.314 |
| Reuse sparse ordering and workspace | 3 | 1.268 / 0.725 / 0.159 | 1.561 / 0.962 / 0.293 |
| Call sparse solve directly from XLA | 3 | 1.310 / 0.776 / 0.150 | 1.594 / 1.007 / 0.263 |
| Also move dense carrier solve behind XLA | 1 | 1.495 / 0.747 / 0.171 | 1.554 / 0.881 / 0.285 |
| Expand Rust coefficient assembly in JAX | 1 | 2.038 / 0.759 / 0.274 | 1.256 / 0.472 / 0.305 |
| Combine expanded assembly with direct XLA sparse and dense solves | 1 | 1.479 / 0.754 / 0.170 | 1.186 / 0.521 / 0.293 |
| Fused host forward and reverse with Rust sweep | 3 | 0.664 / 0.123 / 0.153 | 0.776 / 0.154 / 0.289 |
| Also use Rust for all three coefficient plans | 3 | 0.662 / 0.118 / 0.155 | 0.770 / 0.151 / 0.290 |

A direct XLA sparse call improved warm execution but increased first-run time on both platforms. Moving more individual solves behind XLA still left the surrounding program to compile. Moving the boundary around the optical forward and design pullback produced the largest measured compilation saving. The optical graph builder remains in Python. Only sparse ordering and sweep planning were moved or reused.

### Acceptance measurements

Five alternating control/candidate pairs ran in fresh processes. The Mac used ordinary CPU scheduling on an eight-logical-CPU Apple M1 with 16 GiB RAM and macOS 26.5.2. Linux ran in an ARM64 Docker VM on that M1 with affinity restricted to logical CPU 0. Both used Python 3.12.16 and JAX/jaxlib 0.9.0. The original benchmark and both scored fixture hashes matched for every launch. All launches evaluated 101 frequencies, the 193-coordinate physical gradient and ten warm repeats.

| Platform | Version | First run | Setup | Optical build within setup | Trace and compile | Warm |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| mac | base | 1.1997 | 0.4494 | 0.0430 | 0.5686 | 0.1779 |
| mac | final | 0.5864 | 0.3370 | 0.0568 | 0.1030 | 0.1454 |
| linux | base | 1.3371 | 0.3218 | 0.0487 | 0.7170 | 0.2869 |
| linux | final | 0.6675 | 0.2657 | 0.0485 | 0.1248 | 0.2616 |

The five ordinary Mac candidate totals were 0.619, 0.586, 0.578, 0.599 and 0.573 seconds. A separate plain run in the real checkout took 0.847 seconds, with a 0.139-second warm median. The five Linux candidate totals were 0.636, 0.632, 0.680, 0.668 and 0.893 seconds. The optical build alone increased by 14 ms on the Mac, while total setup decreased by 112 ms. Each candidate's warm median beat its paired control. The installation-time Rust kernel build took 9.298 seconds on macOS and 11.305 seconds on Linux.

All benchmark reference and finite-difference checks passed. On both systems the full repository harness passed and all six unit tests passed, including complex adjoints and JAX differentiation transforms. Direct control/candidate comparisons covered every field and gradient entry on the scored setup at its supplied parameters and with `mb11.tuning` increased by 0.002. The table shows the worst difference over both points and platforms. Loss is near zero and is reported in absolute units.

| Quantity | Difference |
| --- | ---: |
| carrier (relative L2) | 0 |
| signal (relative L2) | 3.79e-10 |
| noise (relative L2) | 2.74e-10 |
| sensitivity (relative L2) | 6.49e-11 |
| powers (relative L2) | 0 |
| loss (absolute) | 9.84e-12 |
| gradient (relative L2) | 1.08e-09 |

The largest harness sensitivity and gradient errors occurred in the near-null Voyager FM case. They were 0.0636 and 0.00142 on macOS, and 0.845 and 0.925 on Linux. The stored tolerances at those points are 27.05 and 9.01, respectively. This case offers much weaker evidence of relative accuracy than the scored field-by-field comparison. The Mac fallback with native kernels disabled also passed the full harness.

Install Cargo, a C++17 compiler and SuperLU development headers (`brew install superlu` on Apple Silicon or `apt install libsuperlu-dev` on Debian). From a Python 3.12 virtual environment in the checkout, run the following commands.

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python -m differometor.rust_kernel.build
python benchmark.py
python tests/check_general.py --suite all
PYTHONPATH=. JAX_PLATFORMS=cpu python -m unittest discover -s tests
```

On Linux run `python benchmark.py --cpus 1` inside a container pinned to a single allowed CPU. Apply the same Python environment and each revision's own native build to the control. The Rust sparse solver is optional and requires a compatible system SuperLU installation.

### KLU for circuit-like optical systems

The local branch `experiment/circuit-sparse-sweep` starts at the packed Rust control `e1015a9`. Its new installation-time option uses SuiteSparse KLU to decompose and order the runtime CSC pattern once per forward or reverse frequency sweep. Signal and detector systems share a pivoted complex128 numerical factorization within each frequency. Numeric pivots are selected again at the next frequency. Both residual refinement steps still run for direct, transpose and conjugate solves. All analysis, loading, forward fields and reverse gradients are inside the scorer. Install SuiteSparse headers before building the generic Rust kernel to enable KLU. The SuperLU and SciPy alternatives remain available.

The second change simplifies the two-sideband layout builder. It expands each component's small index lists once in their original write order, including RHS writes and repeated destinations. A direct comparison found exactly equal setup matrices, instructions and metadata. Sixteen interleaved construction measurements on the same M1 process gave 8.70 ms for the control and 5.83 ms for the new builder. These isolated measurements are separate from the official totals.

Fresh processes alternated control and candidate, reversing order every pair. Each ran the unchanged benchmark and all ten warm repeats. The Mac used normal scheduling on the 8-logical-CPU M1, 16 GiB, macOS 26.5.2. Linux used an ARM64 LinuxKit VM on that machine, restricted to CPU 0 with no container memory cap. Both used Python 3.12.16 and JAX/jaxlib 0.9.0. The target Intel Xeon Platinum 8468 was unavailable.

| Platform | Revision | First run | Setup | Optical build within setup | Trace and compile | Warm |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| mac | base | 0.7135 | 0.4052 | 0.0561 | 0.1459 | 0.1601 |
| mac | final | 0.5563 | 0.3564 | 0.0499 | 0.1147 | 0.0784 |
| linux | base | 0.7946 | 0.2816 | 0.0565 | 0.1515 | 0.2824 |
| linux | final | 0.5465 | 0.2604 | 0.0461 | 0.1338 | 0.1382 |

The initial KLU-only development launch on the Mac took 2.580 s, with 2.051 s in setup, 0.442 s tracing/compiling and 0.078 s warm. The five final candidate totals on the Mac were 0.509, 0.574, 0.556, 0.830 and 0.556 s. Linux recorded 0.579, 0.601, 0.520, 0.523 and 0.547 s. Each candidate beat its paired control. The generic build took 9.623 s on macOS and 8.918 s on Linux. Compiler flags and the JAX differentiation adapter were unchanged.

All 13 reference cases passed at all five parameter points on both platforms, as did all seven unit tests. On the scored fixture all gradient and reference checks passed. Comparing nominal and perturbed physical outputs with the control gave identical carrier and powers. The worst relative L2 differences over both machines were 3.60e-10 for signal, 2.73e-10 for noise, 5.79e-11 for sensitivity and 1.18e-9 for the full gradient. The maximum absolute loss difference was 5.42e-12. The near-null Voyager FM harness case still has wide published tolerances. Its worst sensitivity and gradient errors were 0.0663 and 0.00144 on the Mac, and 0.846 and 0.924 on Linux. A separate two-frequency complex test exercised both the KLU and SuperLU bridges with multiple RHS and all transpose modes.

Follow the installation steps above using `brew install suite-sparse` or `apt install libsuitesparse-dev`, then run `python -m differometor.rust_kernel.build`. The Mac command is `python benchmark.py`. On Linux use `python benchmark.py --cpus 1`. The independent final Mac confirmation from the real checkout took 0.833 s.

### Cold input preparation after the KLU change

Branch `experiment/cold-input-setup` uses `6592abc` as the control. Parameter preparation builds a first-occurrence name index, tracks emitted components in sets, and keeps integer instruction arrays in integer form through stacking. Noise packing sorts flat integer keys while preserving lexicographic coordinate order, negative coordinates and the inverse map for repeated writes. The public dense templates, JAX arrays, physical engine and derivative rules are unchanged.

Five alternating fresh-process pairs used the original benchmark, 101 frequencies, 193 design coordinates and ten warm repeats. Columns are medians in seconds. The Mac was an Apple M1 with normal scheduling, eight logical CPUs, 16 GiB RAM and macOS 26.5.2. Linux was an ARM64 LinuxKit VM on the same host pinned to logical CPU 0. Both ran Python 3.12.16 and JAX/jaxlib 0.9.0 with the existing installation-stage KLU build.

| Platform | Revision | Complete | Setup | Optical input build | Trace and compile | Warm |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| M1 | Control | 0.4574 | 0.2889 | 0.0411 | 0.0971 | 0.0674 |
| M1 | Candidate | 0.4528 | 0.2816 | 0.0399 | 0.0958 | 0.0676 |
| Linux CPU 0 | Control | 0.5090 | 0.2461 | 0.0434 | 0.1214 | 0.1319 |
| Linux CPU 0 | Candidate | 0.5002 | 0.2448 | 0.0424 | 0.1196 | 0.1311 |

In 101 interleaved Mac microbenchmarks, median parameter mapping fell from 1.840 to 1.117 ms and noise packing from 0.074 to 0.019 ms. A mapping-only fresh Mac screen took 0.737 s complete, 0.177 s compilation and 0.070 s warm. A separate plain run of the combined change in the real checkout took 0.474 s. The candidate won three of five paired totals on the Mac and four of five on Linux. The isolated improvements explain roughly 0.8 ms of setup. Fresh-process variability contributes to the larger differences between total medians.

In an independent Mac control diagnostic, imports took 240 ms, CPU initialization 2.81 ms, fixture parsing 1.51 ms and the first scored Python-list conversion to JAX 14.07 ms. The scorer imports JAX before the repository. An import trace attributed 445 ms to JAX and 5 ms to Differometor under instrumentation. Within the 15.20 ms synchronized library build, graph construction took 5.58 ms, native library loading 3.89 ms, CSC construction 0.78 ms, signal packing 0.16 ms and noise packing 0.44 ms. Large zero allocations returned in 0.02 ms, diagonal writes took 0.66 ms, instrumented concatenations 0.44 ms and JAX data conversions 0.78 ms. These nested figures overlap. Allocation alone excludes later page touches. The measured costs supported retaining the dense public templates.

The historical 2.580 s launch included 1.696 s in the scorer's combined build phase. It has no internal trace. A separate instrumented control attributed 530 ms to the scorer's five JAX conversions and 70 ms to the library build. This demonstrates first-use variability but cannot identify the cause of the historical outlier.

Both generality suites passed at all 65 points. Worst sensitivity/gradient harness errors remained in the known near-null Voyager case (0.0663/0.00144 on Mac and 0.846/0.924 on Linux). All seven physical outputs including loss and full gradient agreed exactly with the control at nominal and perturbed scored parameters on both systems. All 13 prepared layouts and metadata matched the control byte for byte with a fixed diagnostic hash seed. The original sensitivity and finite-difference checks passed. Reproduce with the KLU installation above and plain `python benchmark.py` on the Mac or `python benchmark.py --cpus 1` on Linux.

## Benchmarking further changes

For a quick cold-start measurement, run `python benchmark_slim.py` in a fresh process. The [slim runner](../benchmark_slim.py) uses the public UIFO fixture and the scored benchmark's forward, readout and gradient functions. Its single wall-clock time includes imports, CPU initialization, setup, JIT and transfer of the results back to the host. Use `benchmark.py` for the scored result, correctness checks, warm repeats and per-phase timings.

On the pinned Windows CPU described above, the slim runner matched the scored benchmark exactly for the carrier and signal fields, noise, sensitivity, powers, loss, and all 193 gradient entries.

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
