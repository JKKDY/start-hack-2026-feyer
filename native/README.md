# dfm native engine

Ahead-of-time compiled C++ core for the Differometor simulator, with a Python
package `differometor_native` whose public API matches upstream Differometor
(see docs/api-compatibility.md). This directory
contains the complete native simulator (planner, kernels, assembly, solves,
quantum noise and reverse-mode gradients) and its tests. The JAX code in
`differometor/` stays the correctness reference. The original setup snapshot is
archived read-only at `../archive/native-v0-setup`.

## Layout

```
native/
  CMakeLists.txt            top-level build: dfm_core, tests, Python extension
  cmake/Dependencies.cmake  pinned third-party dependencies (FetchContent)
  include/dfm/              public C++ headers (no Python, no nanobind)
  src/                      dfm_core implementation
  bindings/module.cpp       nanobind glue (differometor_native._native), the only Python-aware C++
  python/differometor_native/  upstream-compatible Python API (setups, simulate, components,
                            utils, plot) plus the new native API in native.py
  tools/                    generators (components.py from upstream)
  docs/api-compatibility.md API contract, ownership model, differences from upstream
  docs/kernels.md           component kernels, sparse ops, AD strategy, validation
  docs/planner.md           runtime planner and sparse matrix assembly, validation
  docs/simulator.md         full pipeline, reverse mode, solver accuracy, validation, limits
  tests/reference/          verbatim upstream modules (JAX reference for tests)
  bench/bench_ad.cpp        benchmark: passive chain, tape record, tape replay
  tests/cpp/                GoogleTest unit tests (run by CTest as dfm_tests)
  tests/python/             pytest tests for the bindings
  pyproject.toml            scikit-build-core build backend
```

## Pinned dependencies

| Library  | Release | Source                                      |
|----------|---------|---------------------------------------------|
| Eigen    | 3.4.1   | https://gitlab.com/libeigen/eigen          |
| CoDiPack | v3.1.3  | https://github.com/SciCompKL/CoDiPack      |
| SuperLU  | v7.0.1  | https://github.com/xiaoyeli/superlu        |
| nanobind | v3.1.0  | https://github.com/wjakob/nanobind         |
| GoogleTest | v1.18.0 | https://github.com/google/googletest     |

Each is pinned to an exact commit SHA in `cmake/Dependencies.cmake`. SuperLU
builds its own CBLAS subset, so no system BLAS is needed on Windows or Linux.

## Build and test (Windows, MSVC 2022)

Run these from a Developer PowerShell or after loading the VS environment.

C++ core and tests:

```powershell
cmake -S native -B native/build/cpp -G Ninja -DCMAKE_BUILD_TYPE=Release -DDFM_BUILD_PYTHON=OFF
cmake --build native/build/cpp
ctest --test-dir native/build/cpp --output-on-failure
```

Python extension and tests:

```powershell
cd native
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install ".[test]"
.\.venv\Scripts\python.exe -m pytest tests/python
```

## Benchmark and build options

`dfm_bench_ad` times a chain of sparse solves three ways: passive (plain
doubles), CoDiPack tape record, and tape replay (reverse). Run
`dfm_bench_ad <n> <chain> <reps>`, e.g. `dfm_bench_ad 751 8 5`.

`-DDFM_NATIVE_ARCH=ON` (default) optimises for the build machine's CPU
(`-march=native` / `/arch:AVX2`-style). Turn it off for portable binaries.

## Design notes

- `dfm_core` is independent of Python. `bindings/` is the only code that uses
  nanobind or NumPy types.
- `ComplexSparseLU` factorises once with SuperLU (COLAMD ordering, pivoted LU).
  It serves `A x = b` and `A^H x = b` from the same factors, so the adjoint
  pass never refactorises. This is the building block for implicit adjoint
  differentiation: the gradient never differentiates through LU internals.
- CoDiPack is used for global reverse-mode AD. `complex_gradient_smoke` checks
  that Eigen complex arithmetic works on `std::complex<codi::RealReverse>`.
  Eigen's `squaredNorm()` does not accept CoDiPack types in this setup, so the
  norm is written out in real and imaginary parts.
- Nothing here hard-codes an optical geometry. The geometry will arrive at
  runtime as index structures, as in the existing build step.
- `solve_recorded` records a sparse solve as one CoDiPack external function.
  SuperLU internals are not recorded on the tape; the reverse pass solves
  `A^H` with the stored factors.
