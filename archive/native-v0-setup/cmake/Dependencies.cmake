# Pinned third-party dependencies, fetched at configure time.
#
# Every entry pins an exact commit SHA. The tag in the comment is the
# human-readable release that SHA corresponds to. Do not switch to branch
# names or floating tags: builds must be reproducible.
#
# Dependency          Release   Source
# ------------------  --------  ------------------------------------------
# Eigen               3.4.1     https://gitlab.com/libeigen/eigen
# CoDiPack            v3.1.3    https://github.com/SciCompKL/CoDiPack
# SuperLU             v7.0.1    https://github.com/xiaoyeli/superlu
# nanobind            v3.1.0    https://github.com/wjakob/nanobind

include(FetchContent)

# Keep dependency builds quiet and lean; we only consume their targets.
set(FETCHCONTENT_QUIET ON)

# ---------------------------------------------------------------- Eigen
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        d71c30c47858effcbd39967097a2d99ee48db464  # 3.4.1
  GIT_SHALLOW    FALSE
)

# -------------------------------------------------------------- CoDiPack
# Header-only; exposes the interface target `CoDiPack`.
FetchContent_Declare(
  codipack
  GIT_REPOSITORY https://github.com/SciCompKL/CoDiPack.git
  GIT_TAG        c25d5c19d8207a1a732cfff0f79ac04cbeef01d8  # v3.1.3
  GIT_SHALLOW    FALSE
)

# --------------------------------------------------------------- SuperLU
# SuperLU's CMake can build its own CBLAS subset, so no system BLAS/LAPACK
# is required (this keeps Windows and Linux builds identical).
set(enable_internal_blaslib ON CACHE BOOL "" FORCE)
set(enable_examples OFF CACHE BOOL "" FORCE)
set(enable_tests OFF CACHE BOOL "" FORCE)
set(enable_fortran OFF CACHE BOOL "" FORCE)
set(enable_matlabmex OFF CACHE BOOL "" FORCE)
set(enable_doc OFF CACHE BOOL "" FORCE)
set(enable_single OFF CACHE BOOL "" FORCE)
set(enable_complex OFF CACHE BOOL "" FORCE)
set(enable_complex16 ON CACHE BOOL "" FORCE)
set(enable_double ON CACHE BOOL "" FORCE)
set(TPL_ENABLE_METISLIB OFF CACHE BOOL "" FORCE)
# SuperLU declares an old cmake_minimum_required; CMake >= 4 rejects < 3.5.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
FetchContent_Declare(
  superlu
  GIT_REPOSITORY https://github.com/xiaoyeli/superlu.git
  GIT_TAG        83a3c82b9e44c966ca2bc26991ffdb2bf63e1e2b  # v7.0.1
  GIT_SHALLOW    FALSE
)

# -------------------------------------------------------------- nanobind
# Only needed for the Python bindings (DFM_BUILD_PYTHON=ON).
FetchContent_Declare(
  nanobind
  GIT_REPOSITORY https://github.com/wjakob/nanobind.git
  GIT_TAG        68480a9e6883bb1bf65e1925423321e2a89d07f9  # v3.1.0
  GIT_SHALLOW    FALSE
)

FetchContent_MakeAvailable(eigen codipack superlu)

if(DFM_BUILD_PYTHON)
  FetchContent_MakeAvailable(nanobind)
endif()
