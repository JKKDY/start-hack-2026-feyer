# Differometor performance experiments

The packed sparse build passed the original, unchanged benchmark harness. All 193 gradient values also agreed with the original solver within the additional comparison tolerance at both tested parameter points. The local first-run time, including JIT, fell from **63.13 s to 16.98 s**. The warm median fell from **33.278 s to 0.245 s**. The strict elementwise comparison of raw signal fields still reports differences in the sparse modes.

## Accepted optimization history

The table tracks cumulative builds on one pinned logical CPU on an Intel Core i7-11370H, with native Windows, Python 3.12.10, JAX and JAXLIB 0.9.0, and double precision. Every run used the saved 3×3 UIFO, 101 frequencies, and 193 design variables. Timings are one complete first run and the median of three warm evaluations. The first-run total includes JIT tracing, lowering, and compilation.

| Cumulative build | First incl. JIT (s) | JIT (s) | Warm (s) | Reference sensitivity check, max log error | Gradient spot check, max abs. error | Full gradient relative L2 error | Max per-frequency signal relative L2 error | Strict signal entry mismatches |
| --- | ---: | ---: | ---: | --- | --- | --- | --- | ---: |
| Original dense simulator (`3b538f0`) | 63.13 | 26.91 | 33.278 | Pass (5.38e-6) | Pass (3.83e-7) | 0 / 0 | 0 / 0 | 0 / 0 |
| Noise quadratic form (`4a2c3d0`) | 55.06 | 27.08 | 26.028 | Pass (5.38e-6) | Pass (6.97e-7) | 1.90e-8 / 1.31e-7 | 0 / 0 | 0 / 0 |
| Grouped component evaluation (`3e030f4`) | 44.36 | 14.25 | 18.578 | Pass (5.38e-6) | Pass (6.97e-7) | 1.90e-8 / 1.31e-7 | 0 / 0 | 0 / 0 |
| Detector selections without tiling (`089afa2`) | 41.50 | 15.62 | 18.567 | Pass (5.38e-6) | Pass (6.97e-7) | 1.90e-8 / 1.31e-7 | 0 / 0 | 0 / 0 |
| Shared sparse LU with dense assembly | 34.16 | 13.99 | 8.366 | Pass (5.37e-6) | Pass (1.32e-8) | 2.20e-9 / 1.22e-7 | 2.04e-7 / 1.45e-7 | 3,427 / 3,433 |
| Packed sparse signal assembly | 19.10 | 13.29 | 1.959 | Pass (5.37e-6) | Pass (1.32e-8) | 2.20e-9 / 1.22e-7 | 2.04e-7 / 1.45e-7 | 3,427 / 3,433 |
| Packed signal and noise, original harness | **16.98** | **12.79** | **0.245** | Pass (5.37e-6) | Pass (1.32e-8) | 2.31e-9 / 1.22e-7 | 2.04e-7 / 1.45e-7 | 3,427 / 3,433 |

The original row is the fresh local dense calibration. The repository's [published CPU baseline](../benchmarks/cpu-baseline.json) was measured on a different host and recorded 44.20 s first run, 22.92 s JIT, and 17.68 s warm. Earlier packed runs used a development harness with solver flags. The table uses the original-harness timing and retains the previously tested perturbed outputs for the same packed implementation.

### How to read the numerical checks

- **Sensitivity.** Maximum absolute difference in natural log sensitivity from the public curve across the 101 frequencies. The supplied limit is `1e-5`.
- **Gradient spot check.** Maximum absolute difference between autodiff and the two central-difference estimates for `mb11.tuning`. Each comparison uses the benchmark's `rtol=0.01, atol=1e-6`. The two finite-difference estimates must also agree within that tolerance.
- **Full-gradient and field comparisons.** Paired values are for the saved parameters and for `mb11.tuning` increased by 0.002, each against the original at that same point. All 193 gradient coordinates passed `rtol=1e-3, atol=1e-7`. Carrier fields and powers matched exactly. Quantum noise and sensitivity arrays matched within `rtol=1e-5, atol=0`. The signal comparison uses the largest relative L2 error across individual frequency vectors and counts entries outside the separate `rtol=1e-5, atol=1e-12` elementwise check. All rows produced finite outputs and gradients with the expected shapes, including the full 751 × 101 signal field.

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

## Benchmarking further changes

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
| Packed signal and noise development run | 0.51 | 0.54 |

In a fresh cold rerun, the dense reference took 42.99 s including JIT and the fully packed mode took 18.05 s.

The table above uses the original simulator as its numerical reference. Against the last optimized dense build, the fully packed prototype's full-gradient relative L2 differences were 1.74×10⁻⁸ at the saved point and 2.53×10⁻⁷ at the perturbed point.

On the 21-frequency grid, the dense reference path reproduced the previous physical outputs and loss bit for bit. Its maximum absolute gradient difference was 6.90×10⁻⁹.

The fully packed build is the simulator default. Its acceptance covers the fixed benchmark workload, the supplied checks, and the tested full-gradient comparisons. Callers can select the dense reference through `run_build_step(..., signal_solver="dense")`. The mixed-precision refinement below remains a diagnostic.

The challenge's `benchmark.py` is unchanged from `3b538f0`. On Linux or WSL, run it on this branch for the packed build:

```bash
python benchmark.py --cpus 1 --repeats 3 --output benchmark-packed.json --save-arrays packed.npz
```

For the original dense reference, run the same command from a separate checkout of `3b538f0`, with a distinct output path and the same Python environment, hardware, and CPU limits.

### Signal-field stability

The dense assembly and the reconstructed packed assembly gave identical signal matrices and right-hand sides at all 101 frequencies, both at the saved parameters and with `mb11.tuning` increased by 0.002. The packed layout omitted no nonzero coefficients. The raw matrices have widely differing row scales. Their estimated 1-norm condition numbers were about 10³⁴–10³⁷, and the largest componentwise backward error was 1 for both double-precision solvers, even though their global normwise backward errors were below 1.3×10⁻²⁸.

An independent sparse Gaussian elimination solved the captured binary64 systems at 80 and 110 decimal digits. The two references agreed to within 1.4×10⁻⁷² relative infinity norm across the eight cases. These cases covered both parameter points at 800 Hz, the largest dense-to-sparse field difference at about 821.43 Hz, and detector-response peaks near 1321.97 and 1744.89 Hz. This isolates the linear solve from rounding in the physical coefficient formulas.

At 800 Hz, the exact reference for zero-based field index 238 is zero. The sparse solve returned zero and the dense solve returned 15.35. The dependency graph identified 46 variables with no path from a nonzero right-hand-side entry at every frequency. These variables are exactly zero because the unforced principal block is nonsingular. Across the nominal sweep, 929 of the 3,427 elementwise disagreements were in those variables. Both double-precision solvers also failed strict elementwise comparison with the high-precision reference on driven variables.

A diagnostic solve removed the unforced block, equilibrated the remaining rows and columns, and factored the reduced system in double precision. Equilibration alone still missed 8–10 fields per sampled case at the existing tolerance of `rtol=1e-5, atol=1e-12`. Three iterative refinement steps reused the same factors and evaluated the residual at 110 decimal digits. This eliminated every strict mismatch across the eight reference cases. The worst componentwise backward error was 1.08×10⁻¹⁶ across all 202 refined cases. Only the eight sampled vectors were compared with independent higher-precision solutions.

For the next stability check, require structurally unforced fields to be exactly zero and compare driven fields individually against independent higher-precision references. Use componentwise residuals across the sweep to catch poorly solved rows. The refinement diagnostic covers signal fields. The production solver, adjoint, gradients, and performance of a refined implementation still need separate validation.
