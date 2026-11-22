# Numerical and physics kernels (milestone 2)

The reusable numerical layer of the native engine. It covers the 19
registered component functions, sparse storage, sparse solves and products,
and their reverse-mode derivatives. None of it depends on an optical
geometry: everything works on runtime-sized patterns, parameter rows and
value arrays. The planner and the full simulation pipeline are not part of
this milestone.

## Files

| Header | Purpose |
|---|---|
| `jet.hpp` | `Jet<K>`: complex forward-mode jet with K real seed directions |
| `components.hpp` | The 19 kernels and their helpers, templated on the scalar type |
| `component_ops.hpp` | Passive evaluation, Jacobian and VJP, single entries and batches |
| `component_ad.hpp` | `record_components`: kernels on the CoDiPack tape |
| `sparse.hpp` | `SparsePattern` (square CSC) and passive `A B`, `Aᴴ B` |
| `sparse_lu.hpp` | SuperLU LU: factor once; solve `A x = b` and `Aᴴ x = b`, one or several RHS |
| `sparse_ad.hpp` | `RecordedLinearSystem` and `sparse_matmul_recorded` (CoDiPack external functions) |
| `ad_sparse_solve.hpp` | Active scalar types (`ActiveReal = codi::RealReverse`) and the older RHS-only solve |
| `link.hpp` | Parameter-link functions, which work on `double` and on CoDiPack types |
| `detectors.hpp` | Amplitude, power and signal detectors |

## Conventions

- **Kernel I/O:** each kernel reads the first `arity(f)` real parameters and
  writes 4 complex outputs, which matches upstream `standardize_output`.
  Upstream pads every input row to 7 values (`kComponentInputWidth`); the
  extra values are ignored.
- **Complex arithmetic:** upstream evaluates kernels on `complex128`
  parameters whose imaginary part is zero. The port uses complex arithmetic in
  the same operation order, so it also follows upstream's complex branches,
  for example `asin(z)` with `|z| > 1` under total internal reflection.
- **Cotangents:** for a real loss L and a complex value z, the cotangent is
  `zbar = dL/dRe z + i dL/dIm z`. JAX's convention is the complex conjugate
  of this.
- **Active storage:** complex values are interleaved active reals,
  `(re, im) = (a[2s], a[2s+1])`. Dense `n × m` blocks are column-major: entry
  `(i, c)` is at `2(c·n + i)`.

## Differentiation strategy

**Kernels: preaccumulated Jacobians.**
- One evaluation with `Jet<arity>` gives the value and the exact `4 × arity`
  complex Jacobian. `conj` is valid because the seed directions are real.
- `record_components` pushes one CoDiPack statement per non-zero real output,
  using `codi::StatementPushHelper`, with at most `arity` arguments each.
  Intermediate operations (`exp`, `sqrt`, `asin`, ...) never reach the tape.
  For example, `surface` costs exactly 6 statements per entry. This is checked
  in `test_components.cpp`.
- Entries with only passive inputs, or recorded while the tape is inactive,
  skip the Jacobian and record nothing.
- Dispatch is one `switch` per batch, never per entry. Jets live on the stack,
  so nothing allocates per entry.
- Jet and plain-complex evaluation give bit-identical primal values.

**Sparse operations: one external function each, implicit adjoint.**
- The LU factorisation is never differentiated.
- `RecordedLinearSystem` factorises the primal matrix once. Every `solve`
  (normal or `Aᴴ`, any number of right-hand sides) records one
  `ExternalFunctionHelper` entry. Its user data holds `shared_ptr`s to the
  factors and the pattern, so the tape stays valid after the system object is
  gone.

| Operation | Reverse rule |
|---|---|
| `X = A⁻¹ B` | `B̄ = A⁻ᴴ X̄`, `Ā_ij = −Σ_c B̄_ic conj(X_jc)` |
| `X = A⁻ᴴ B` | `B̄ = A⁻¹ X̄`, `Ā_ij = −Σ_c X_ic conj(B̄_jc)` |
| `Y = A B` | `B̄ = Aᴴ Ȳ`, `Ā_ij = Σ_c Ȳ_ic conj(B_jc)` |
| `Y = Aᴴ B` | `B̄ = A Ȳ`, `Ā_ij = Σ_c B_ic conj(Ȳ_jc)` |

- `Ā` is formed only on occupied slots.
- **Stored primals:** solves keep their outputs and drop their inputs; products
  keep their inputs and drop their outputs. In both cases that is only what
  the reverse rule needs.
- **Reuse:** the signal solve and the noise-weight (`Aᴴ`) solve share one
  factorisation, as the simulator will need. This is tested in
  `test_sparse_ad.cpp`.

## The 19 registered kernels

Order and indices match upstream `FUNCTIONS`. The input names follow
upstream's parameter layout.

| # | Function | Arity | Inputs |
|---:|---|---:|---|
| 0 | force_calculation_right | 3 | alpha, n_left, n_right |
| 1 | laser | 2 | power, phase |
| 2 | surface | 7 | loss, reflectivity, tuning, frequency, n_left, n_right, alpha |
| 3 | space | 3 | frequency, length, n |
| 4 | space_modulation | 3 | frequency, length, n |
| 5 | signal_function | 2 | amplitude, phase |
| 6 | dummy_function | 0 | (none) |
| 7 | vacuum_quantum_noise | 0 | (none) |
| 8 | loss_quantum_noise | 1 | loss |
| 9 | space_lower | 3 | frequency, length, n |
| 10 | laser_amplitude_modulation | 0 | (none) |
| 11 | laser_frequency_modulation | 1 | frequency |
| 12 | susceptibility | 2 | frequency, mass |
| 13 | force_calculation_left | 1 | alpha |
| 14 | corrected_optomechanical_phase_left | 6 | frequency, tuning, reflectivity, loss, n_left, alpha |
| 15 | corrected_optomechanical_phase_right | 7 | frequency, tuning, reflectivity, loss, n_left, n_right, alpha |
| 16 | space_modulation_lower | 3 | frequency, length, n |
| 17 | squeezer | 2 | db, angle |
| 18 | laser_frequency_modulation_lower | 1 | frequency |

The helpers are ported both natively (templated, used inside the kernels) and
in Python: `incidence_angle_left` and `_right`, `phase_shift`,
`reflectivity_left` and `_right`, `transmissivity`,
`optomechanical_phase_left` and `_right`, `tuning_correction`. Python also
has `standardize_output`, `mirror_matrix` and `beamsplitter_matrix`, as upstream
formulas with `jax.numpy` replaced by `numpy`.

## Validation (gcc 15, WSL)

**C++, 72 GoogleTest tests:**
- every kernel's jet Jacobian against central finite differences;
- the recorded tape against the jet VJP, with identical primals;
- the statement count per entry;
- parameter links in reverse mode;
- recorded solves (`A` and `Aᴴ`, three right-hand sides, active matrix) and
  products against finite differences with complex cotangents;
- two solves sharing one factorisation;
- duplicate coordinates mapping to one shared slot;
- multi-RHS LU against single solves.

**Python, 94 pytest tests, with JAX 0.9.0 as the reference** (verbatim
upstream `components.py` in `tests/reference/differometor_upstream/`):
- every kernel's values to 1e-13, through both `_native.kernels` and the
  public `differometor_native.components.<name>`;
- gradients for random complex cotangents to 1e-10, through both the jet VJP
  and the CoDiPack tape;
- batch against single evaluation;
- the total-internal-reflection branch;
- the helper formulas;
- the sparse solve and product adjoints (both orientations) against
  `jax.grad` of the dense equivalent.

MSVC was not re-run for this milestone (gcc was the agreed target).

## Simulator operations identified, and their status

| Operation (from upstream `simulate`) | Status |
|---|---|
| 19 component kernels | done (`components.hpp`, `component_ad.hpp`) |
| Parameter links (`LINKING_FUNCTIONS`) | done (`link.hpp`, works on `ActiveReal`) |
| Detectors (power, amplitude, signal demodulation) | done, passive (`detectors.hpp`) |
| Sparse solve with active matrix (carrier, signal, noise weights, covariances) | done (`RecordedLinearSystem`) |
| Sparse product (noise sources = noise matrix × weights) | done (`sparse_matmul_recorded`) |
| Pattern and slot mapping for assembly | done (`SparsePattern::from_coordinates`) |
| Parameter expansion, scatter of kernel outputs into slots | done in milestone 3 (`assembly.hpp`) |
| Carrier-scaled signal entries (`−entry · carrier`), lower-sideband conjugation blocks | done in milestone 3 (`assembly.hpp`) |
| QHD phase `exp(i·radians(phase))`, noise selection vectors | pipeline milestone (scalar ops) |
| Covariance reduction `Re Σ cov·conj(sel)` and ASD `sqrt(2·S·u·h·f0/4)` | pipeline milestone (one structured reduction per detector) |

## Unsupported or different

- **Complex parameters:** the kernels take real parameters. Upstream accepts
  complex ones, but its simulator only ever passes zero imaginary parts. The
  Python wrappers reject inputs with a non-zero imaginary part (`ValueError`).
- **Return types:** kernels return NumPy `complex128` arrays of length 4 where
  upstream returns `jax.Array`. Helpers called with Python floats follow NumPy
  real arithmetic, as upstream's helpers follow JAX real arithmetic.
- **Forward mode:** the external sparse operations implement primal and
  reverse only. CoDiPack forward types and a forward tape sweep are not
  supported for them.
- **Singular points:** behave like upstream. Derivatives at `sqrt(0)` (zero
  reflectivity or transmissivity) and `1/0` (zero frequency in modulation
  kernels) are non-finite.
- **Threads:** CoDiPack's tape is per thread. Kernels and passive operations
  are thread-safe; recording is per-thread state.
