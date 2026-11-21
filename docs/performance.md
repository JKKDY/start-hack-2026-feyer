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

#### Proposed end-to-end split

The latest baseline spent 5.051 s tracing and compiling the simulator forward pass and 2.493 s on the reverse pass, versus 0.312 s on the objective readout. Prioritize the simulator forward and reverse passes together. Keep the existing `Setup` API and pass a runtime plan into the following stages.

1. **Evaluate coefficients.** Gather component inputs from the plan and evaluate them in fixed-size batches per component type using the AOT bundle. Compile each component's pullback alongside its forward kernel. Only the last batch per group needs padding. Component and frequency counts determine how many times to call the kernels.
2. **Assemble and solve the optics.** Use Cython buffers and loops for packed indexing, carrier coupling and noise contraction. Reuse LAPACK for carrier systems and the existing SuperLU signal and adjoint solves with refinement. Differentiate the equations through transpose solves and accumulate gradients at the existing coefficient slots. Keep the complex-conjugation and repeated-write semantics of the current reverse pass.
3. **Read the outputs and seed the reverse pass.** For the scored workload, compute sensitivity and the existing loss, then send its output cotangents back through the solver and component pullbacks. The simulator pullback also accepts cotangents from other objectives.

Call the AOT executables at batch boundaries from the Python driver. A Cython call to JAX's `Compiled` object still goes through the Python and JAX runtimes. Cython can reduce packing and scattering overhead with typed buffers. A direct C or C++ call becomes worth investigating if measured dispatch or transfer costs are substantial. [Cython and the GIL](https://cython.readthedocs.io/en/latest/src/userguide/nogil.html)

The [builder](../differometor/simulate.py) derives its array keys from the declared argument names of `simulate`. Keep that public signature and put the AOT dispatch and JAX derivative rules behind it. The [scored benchmark](../benchmark.py) passes the simulation pullback between two separately compiled functions. The adapter therefore needs an array-based forward state with shapes derived from the plan. A reverse-mode rule limited to `optimized_parameters` is suitable for the initial private candidate. Before switching the default, check the other differentiable inputs and JAX transformations supported by the baseline, as well as solver options and user-defined parameter links.

Evaluate the three stages as one candidate. Time a fresh process over the same interval as `benchmark_slim.py`, including bundle loading and plan construction. Run the cavity and UIFO with the same bundle, verify the physical outputs, sensitivity and full gradient, and measure warm repeats against a paired baseline. The unchanged scored harness explicitly lowers and compiles three outer JAX wrappers with the persistent cache disabled. Measure it separately if the native engine is exposed through a JAX adapter. [JAX FFI documentation](https://docs.jax.dev/en/latest/ffi.html)

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
