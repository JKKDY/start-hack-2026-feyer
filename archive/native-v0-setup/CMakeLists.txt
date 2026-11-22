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

option(DFM_BUILD_TESTS "Build the C++ test executable" ON)
option(DFM_BUILD_PYTHON "Build the nanobind Python extension" ON)

include(cmake/Dependencies.cmake)

# ------------------------------------------------------------ native core
# Everything in dfm_core is independent of Python. Only the bindings target
# below knows about nanobind.
add_library(dfm_core
  src/version.cpp
  src/sparse_lu.cpp
  src/ad_eigen.cpp
)
add_library(dfm::core ALIAS dfm_core)

target_include_directories(dfm_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

# Eigen is part of the public interface (headers expose Eigen types).
target_link_libraries(dfm_core PUBLIC Eigen3::Eigen)

# SuperLU and CoDiPack are implementation details of the .cpp files.
target_link_libraries(dfm_core PRIVATE superlu CoDiPack)
target_compile_definitions(dfm_core PRIVATE CODI_EnableEigen=1)

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
  add_executable(dfm_core_tests tests/cpp/test_core.cpp)
  target_link_libraries(dfm_core_tests PRIVATE dfm::core)
  add_test(NAME dfm_core_tests COMMAND dfm_core_tests)
endif()

# ------------------------------------------------------- Python bindings
if(DFM_BUILD_PYTHON)
  find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
  nanobind_add_module(_dfm_native STABLE_ABI NB_STATIC
    bindings/module.cpp)
  target_link_libraries(_dfm_native PRIVATE dfm::core)
  install(TARGETS _dfm_native LIBRARY DESTINATION dfm_native)
endif()
