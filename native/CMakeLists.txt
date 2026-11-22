cmake_minimum_required(VERSION 3.26)

project(
  dfm_native
  VERSION 0.1.0
  DESCRIPTION "Native C++ engine for the Differometor optical simulator"
  LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# The core is linked as a static library into the Python extension (a shared
# object), so every static dependency must be position-independent.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(DFM_BUILD_TESTS "Build the GoogleTest suite" ON)
option(DFM_BUILD_BENCH "Build the CoDiPack record/replay benchmark" ON)
option(DFM_BUILD_PYTHON "Build the nanobind Python extension" ON)
# Compile for the host CPU (AVX2/AVX-512 vectorisation for Eigen kernels).
# Turn OFF for portable binaries.
option(DFM_NATIVE_ARCH "Optimise for the build machine's CPU" ON)

# Instruction set for EVERY target, including the fetched dependencies:
# SuperLU and its CBLAS do the factorisations and triangular solves, so they
# must get AVX2/FMA too, not only our own code. Set before the dependencies
# are added so their targets inherit it.
if(DFM_NATIVE_ARCH)
  if(MSVC)
    add_compile_options(/arch:AVX2)
  else()
    add_compile_options(-march=native -mtune=native)
  endif()
endif()

include(cmake/Dependencies.cmake)

# ------------------------------------------------------------ native core
# Everything in dfm_core is independent of Python. Only the bindings target
# below knows about nanobind.
add_library(dfm_core
  src/version.cpp
  src/sparse_lu.cpp
  src/ad_eigen.cpp
  src/ad_sparse_solve.cpp
  src/setup_spec.cpp
  src/plan.cpp
  src/simulator.cpp
  src/detectors.cpp
  src/component_ops.cpp
  src/component_ad.cpp
  src/sparse.cpp
  src/sparse_ad.cpp
  src/planner.cpp
  src/assembly.cpp
  src/pipeline.cpp
)
add_library(dfm::core ALIAS dfm_core)

target_include_directories(dfm_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

# Eigen and CoDiPack appear in public headers (ad_sparse_solve.hpp), so both
# are PUBLIC. SuperLU stays private: it is only used inside sparse_lu.cpp.
target_link_libraries(dfm_core PUBLIC Eigen3::Eigen CoDiPack)
target_link_libraries(dfm_core PRIVATE superlu)
target_compile_definitions(dfm_core PUBLIC CODI_EnableEigen=1)
target_compile_definitions(dfm_core PRIVATE DFM_VERSION_STRING="${PROJECT_VERSION}")

if(MSVC)
  # CoDiPack uses fopen/strerror; Eigen triggers C5054 on enum operators.
  target_compile_definitions(dfm_core PRIVATE _CRT_SECURE_NO_WARNINGS)
  target_compile_options(dfm_core PRIVATE /W4 /wd5054)
else()
  target_compile_options(dfm_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

# ---------------------------------------------------------------- tests
if(DFM_BUILD_TESTS)
  enable_testing()
  add_executable(dfm_tests
    tests/cpp/test_version.cpp
    tests/cpp/test_sparse_lu.cpp
    tests/cpp/test_ad.cpp
    tests/cpp/test_ad_sparse_solve.cpp
    tests/cpp/test_api.cpp
    tests/cpp/test_components.cpp
    tests/cpp/test_sparse_ad.cpp
    tests/cpp/test_planner.cpp
    tests/cpp/test_simulator.cpp
  )
  target_link_libraries(dfm_tests PRIVATE dfm::core GTest::gtest_main)
  include(GoogleTest)
  gtest_discover_tests(dfm_tests DISCOVERY_TIMEOUT 60)
endif()

# ------------------------------------------------------------ benchmarks
# CoDiPack recording and replay are timed separately. Results are printed to
# stdout, so they can be compared run to run.
if(DFM_BUILD_BENCH)
  add_executable(dfm_bench_ad bench/bench_ad.cpp)
  target_link_libraries(dfm_bench_ad PRIVATE dfm::core)
endif()

# ------------------------------------------------------- Python bindings
if(DFM_BUILD_PYTHON)
  find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
  nanobind_add_module(_native STABLE_ABI NB_STATIC
    bindings/module.cpp)
  target_link_libraries(_native PRIVATE dfm::core)
  install(TARGETS _native LIBRARY DESTINATION differometor_native)
endif()
