# Differometor performance experiments

The optimization branch retains three changes. It evaluates quantum noise as a quadratic form, groups optical component calls by function, and constructs detector selections without tiling. The final Windows run took **41.50 s including JIT**, down from **55.06 s** with the quadratic-form change alone. The warm median fell from **26.03 s to 18.57 s**.

## Results

All runs used one pinned logical CPU on an Intel Core i7-11370H, native Windows, Python 3.12.10, JAX and JAXLIB 0.9.0, and double precision. The benchmark evaluated the saved 3×3 UIFO at 101 frequencies with 193 design variables. Each row reports one complete first run and the median of three warm evaluations. JIT time includes tracing, lowering, and compilation. Memory columns report the Windows process peaks during the measured workload, before validation.

| Variant | First run (s) | JIT (s) | Warm (s) | Working set (GiB) | Private commit (GiB) | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Noise quadratic form (`4a2c3d0`) | 55.06 | 27.08 | 26.03 | 7.83 | 15.43 | Retained baseline |
| Compact covariance | 66.60 | 29.94 | 18.73 | 7.12 | 12.89 | Dropped |
| Shared signal and adjoint LU factors | 56.67 | 27.38 | 16.87 | 8.01 | 13.76 | Dropped |
| Grouped component evaluation (`3e030f4`) | 44.36 | 14.25 | 18.58 | 8.57 | 15.03 | Retained |
| Detector selections without tiling (`089afa2`) | 41.50 | 15.62 | 18.57 | 8.33 | 15.03 | Retained |

Compact covariance, shared LU, and grouped component evaluation were each tested on top of the quadratic-form baseline. The detector selection change was tested on top of grouped evaluation.

- **Noise quadratic form.** Given $A^H u=d$, the code evaluates $\operatorname{Re}(u^H Q u)$ directly. This removes one batched signal-system solve and its backward pass. The original trial showed a warm improvement and mixed first-run results.
- **Compact covariance.** Gathering occupied entries of $Q$ and contracting them with the detector weights lowered warm time and peak private commit. On the cold repeat it took 51.33 s against the control's 53.05 s. The first-run results differed in direction, so the experiment was dropped.
- **Shared LU.** One signal factorization serviced both the signal and adjoint solves, with implicit differentiation through the linear systems. Its warm time improved, but its first runs took 56.67 s and 54.29 s against control runs of 55.06 s and 53.05 s.
- **Grouped component evaluation.** The builder groups calls by function and restores the original coefficient order before scattering. This reduced JIT time by about 47% against the quadratic-form baseline.
- **Detector selections without tiling.** The carrier multiplication broadcasts the sideband selections, then concatenates them with the extra degrees of freedom. The repeat took 40.98 s with the change and 41.22 s without it. Warm performance was effectively unchanged.

## Numerical checks

Every trial passed the benchmark's reference sensitivity check and its two-step finite-difference check for `mb11.tuning`. Additional comparisons covered carrier and signal fields, quantum noise, sensitivity, powers, and all 193 gradient components against the original simulator and the relevant control. These comparisons used the saved parameters and a second point with `mb11.tuning` increased by 0.002 in its native units.

The final branch's gradient relative L2 difference from the original was **1.90×10⁻⁸** at the saved point and **1.31×10⁻⁷** at the perturbed point. The detector selection cleanup left the gradient unchanged at both points relative to grouped evaluation.

## Benchmarking further changes

Run each candidate and its control in fresh processes on the same hardware and pinned dependency versions. Use the complete first-run total for the main comparison and record JIT, warm runtime, and peak process memory separately. Repeat close first-run comparisons in reverse order. Compare full physical outputs and gradients at both parameter points before retaining a change.

On Linux or WSL, the standard benchmark command is:

```bash
python benchmark.py --cpus 1 --repeats 3 --output benchmark-candidate.json --save-arrays candidate.npz
```

The Windows experiments pinned the Python process externally before importing JAX because the benchmark's `--cpus` flag uses Linux affinity. A local harness recorded process memory and evaluated the second parameter point outside the timed phases. The reference fixture is checked byte for byte, so use a checkout that preserves its LF line endings. The standard command saves arrays for the nominal point.
