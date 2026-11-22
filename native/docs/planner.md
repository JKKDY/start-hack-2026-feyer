# Runtime planner and matrix assembly (milestone 3)

The native counterpart of upstream `build.py` and of the matrix-building part
of upstream `simulate`. The two are separate modules that share one data
model:

| File | Role |
|---|---|
| `src/planner.cpp` | Port of `build`, `set_instructions`, `parameter_linking`, `pairs_to_arrays`, `prepare_arrays` |
| `include/dfm/plan_internals.hpp` | Plan data model: instruction tables, matrix layouts, mappings |
| `src/assembly.cpp`, `include/dfm/assembly.hpp` | Carrier, signal and noise assembly; passive and CoDiPack-recorded |
| `src/plan.cpp` | Export of upstream's `simulation_arrays` |

There is still no solver pipeline: carrier fields for the signal matrix are
an input.

## What a plan holds

`build_plan(SetupSpec, PlanRequest)` returns an immutable `SimulationPlan`.

**Parameters and mappings**
- **Parameter vector:** in upstream order. Indices 0–2 are F, the default
  refractive index, and 0; per-component values follow in build order. The
  names are as in upstream (`"m0_tuning"`, `"l0_m0_length"`, ...).
- **Links:** `parameters[target] = f(parameters[source])`, with upstream's
  identity placeholder when there are none.
- **Optimisation mapping:** `parameters[idx] = x[value_idx]`, including tied
  groups and upstream's default (`x[0] → parameter 0`) when there are no
  optimisation pairs.
- **Sweeps:** carrier sweep or signal sweep, with upstream's defaults and its
  "not both" error.

**Instruction tables and layouts**
- **Instruction tables** (carrier, signal, noise): kernel rows (function plus
  7 parameter indices) and placements (kernel output → matrix position; column
  −1 is the right-hand side). The signal table also holds the carrier
  couplings.
- **Matrix layouts:** for each matrix, a `SparsePattern` (CSC) with the
  constant template values (identity, `nothing` and isolator blocks, and the
  build-time carrier entries), the slot of every placement and coupling, the
  conjugated lower-sideband slots, and the kernel rows grouped by function.

**Readout and metadata**
- **Noise readout:** `noise_selection_vectors` and the QHD phase indices and
  placings.
- **Metadata:** detector, mirror, beamsplitter and isolator indices, and
  `port_to_index`, all in upstream order.

Every size is a runtime value, so the same binary handles every setup.

## Assembly semantics

For batch element b, in upstream order:

1. **Parameters:** base values, then the optimised values, then the links,
   then the sweep values. The carrier uses the carrier sweep (`CV` elements);
   signal and noise use the carrier sweep and then the signal sweep
   (`max(CV, SV)` elements).
2. **Kernels:** one batched call per component function.
3. **Placements:** values start from the template, and each placement *sets*
   its slot; for duplicates the later placement wins.
4. **Carrier couplings (signal only):** multiply their slot by
   `−carrier[b][index]`, broadcasting when `CV = 1`.
5. **Conjugation (signal only):** conjugate `[S:2S, S:2S]`, `[2S:, :S]` and
   `[S:2S, 2S:]`, including the right-hand-side entries of rows S..2S.

**Carrier template.** Upstream re-evaluates the carrier entries only for
components named in the optimisation or carrier-sweep pairs. Everything else
is the build-time template. The native plan does the same: the template is
evaluated once at plan time and only those components' kernels run per
call. For a tied group, upstream re-evaluates only the *last* member's
component (a loop variable leaks); this is reproduced.

**Recorded assembly.** It reads kernel inputs in place through pointer tables
(`record_components_indirect`). Shared active parameters are never copied;
per-batch sweep values are passive constants. Each kernel entry adds at most
one statement per non-zero real output. `assembly_vjp` returns the gradient
with respect to the optimised values and, for the signal matrix, the carrier
fields.

## Validation (gcc 15, WSL)

`tests/python/test_planner_vs_upstream.py` runs each configuration through
verbatim upstream code (`tests/reference/differometor_upstream`, JAX
0.9.0) and through the native planner. All configurations use the same
compiled extension.

| Configuration | What it exercises |
|---|---|
| cavity, carrier sweep (upstream example) | carrier-changing path, carrier-only optimisation |
| cavity with frequency, free mass, space signal, qnoised | signal sweep, optomechanics, carrier couplings, noise |
| aLIGO (`setups.aligo`) | beamsplitter, many components |
| Voyager, space modulation | squeezer, directional beamsplitter, QHD |
| Voyager, amplitude modulation | parameter links (`sqrt` of laser power), laser amplitude modulation |
| Voyager, frequency modulation | laser frequency modulation (upper and lower) |
| UIFO 2×2 (`setups.uifo(2)`) | generated grid, tied optimisation groups |
| UIFO 3×3 (benchmark fixture JSON) | the scored setup (S = 286, N = 751), 3 frequencies |

**Checks per configuration:**
- metadata, including `port_to_index` order;
- the parameter vector;
- all name-derived mappings, and the link functions (compared by value);
- carrier and signal instruction arrays (exact);
- noise instructions (as a set);
- build-time templates;
- assembled carrier, signal and noise matrices for 3 frequencies at two
  parameter scales, with identical non-zero patterns (tolerance 1e-12 ×
  largest entry);
- reverse-mode gradients of a random complex linear functional of each
  matrix against `jax.grad` of upstream's assembly, with respect to the
  optimised parameters and the carrier fields (tolerance 1e-9 × largest).

Python totals: 150 tests (56 planner comparisons, plus the kernel, linalg
and API tests). C++: 77 GoogleTest tests, including planner dimensions,
carrier identity and laser source, and assembly gradients against finite
differences for all three matrices.

## Differences from upstream

- **No mutation:** upstream `build` mutates the `Setup` (default ports,
  directions, `"<edge>_source"` targets, default properties). The native
  planner works on an immutable snapshot and leaves the Python `Setup`
  unchanged. Setups saved after an upstream build, whose targets are
  `"<edge>_source"`, are accepted.
- **Deterministic noise order:** upstream orders the noise sources of unused
  ports by Python set iteration, which depends on the hash seed. The native
  planner sorts them. The noise matrix is identical; only the order of the
  noise instruction arrays can differ.
- **Link-function ids:** `linking_function_indices` are native ids (see
  `_native.link_functions()`), not indices into upstream's process-global
  `LINKING_FUNCTIONS` list.
- **Template rounding:** template entries come from the native kernels
  (complex arithmetic) rather than `mirror_matrix`/`beamsplitter_matrix`
  (real arithmetic). They agree to round-off; the tests check 1e-13.
- **Errors:** inputs upstream rejects late with `KeyError` (for example a
  detector at the end of an edge, or a laser without a target) raise
  `ValueError` here, with a message.
