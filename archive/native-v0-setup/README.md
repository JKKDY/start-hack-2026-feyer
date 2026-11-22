# dfm native engine

Ahead-of-time compiled C++ core for the Differometor simulator. This directory
contains only the project setup, dependency integration and smoke tests. The
simulator itself is not ported yet. The JAX code in `differometor/` stays the
correctness reference.

## Layout

```
native/
  CMakeLists.txt            top-level build: dfm_core, tests, Python extension
  cmake/Dependencies.cmake  pinned third-party dependencies (FetchContent)
  include/dfm/              public C++ headers (no Python, no nanobind)
  src/                      dfm_core implementation
  bindings/module.cpp       nanobind glue, the only Python-aware code
  python/dfm_native/        pure-Python package re-exporting the extension
  tests/cpp/                C++ smoke tests (run by CTest)
  tests/python/             Python smoke tests (pytest)
  pyproject.toml            scikit-build-core build backend
```

## Pinned dependencies

| Library  | Release | Source                                      |
|----------|---------|---------------------------------------------|
| Eigen    | 3.4.1   | https://gitlab.com/libeigen/eigen          |
| CoDiPack | v3.1.3  | https://github.com/SciCompKL/CoDiPack      |
| SuperLU  | v7.0.1  | https://github.com/xiaoyeli/superlu        |
| nanobind | v3.1.0  | https://github.com/wjakob/nanobind         |

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
