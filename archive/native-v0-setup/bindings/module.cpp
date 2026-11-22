// nanobind bindings for the dfm native core.
//
// This is the only file that knows about Python. It converts NumPy arrays to
// the plain C++ types of dfm_core, calls into the core, and wraps results back
// into NumPy arrays. No numerical logic lives here.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

#include <complex>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "dfm/ad_eigen.hpp"
#include "dfm/sparse_lu.hpp"
#include "dfm/version.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

using ComplexArray = nb::ndarray<std::complex<double>, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using IndexArray = nb::ndarray<std::int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

// Moves a std::vector into a NumPy array without copying. The vector is
// owned by a capsule that frees it when the array is garbage collected.
template <typename T>
nb::ndarray<nb::numpy, T, nb::ndim<1>> to_numpy(std::vector<T> values) {
  auto* holder = new std::vector<T>(std::move(values));
  nb::capsule owner(holder, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
  return nb::ndarray<nb::numpy, T, nb::ndim<1>>(holder->data(), {holder->size()}, owner);
}

// Builds a square CSC matrix from NumPy arrays, validating the structure so
// that the core never sees out-of-range indices.
dfm::SparseMatrixC build_csc(const IndexArray& indptr, const IndexArray& indices,
                             const ComplexArray& data) {
  if (indptr.shape(0) < 1) {
    throw std::invalid_argument("indptr must have at least one entry");
  }
  const int n = static_cast<int>(indptr.shape(0)) - 1;
  const int nnz = static_cast<int>(indices.shape(0));
  if (static_cast<int>(data.shape(0)) != nnz) {
    throw std::invalid_argument("indices and data must have the same length");
  }
  if (indptr.data()[0] != 0 || indptr.data()[n] != nnz) {
    throw std::invalid_argument("indptr must start at 0 and end at len(indices)");
  }
  for (int col = 0; col < n; ++col) {
    if (indptr.data()[col] > indptr.data()[col + 1]) {
      throw std::invalid_argument("indptr must be non-decreasing");
    }
  }
  for (int k = 0; k < nnz; ++k) {
    if (indices.data()[k] < 0 || indices.data()[k] >= n) {
      throw std::invalid_argument("row index out of range");
    }
  }

  const Eigen::Map<const dfm::SparseMatrixC> view(n, n, nnz, indptr.data(), indices.data(),
                                                  data.data());
  return dfm::SparseMatrixC(view);
}

dfm::VectorC to_vector(const ComplexArray& array, int expected_size) {
  if (static_cast<int>(array.shape(0)) != expected_size) {
    throw std::invalid_argument("right-hand side has length " + std::to_string(array.shape(0)) +
                                ", expected " + std::to_string(expected_size));
  }
  return Eigen::Map<const dfm::VectorC>(array.data(), expected_size);
}

std::vector<std::complex<double>> to_std_vector(const dfm::VectorC& v) {
  return std::vector<std::complex<double>>(v.data(), v.data() + v.size());
}

}  // namespace

NB_MODULE(_dfm_native, m) {
  m.doc() = "Native C++ Differometor engine (CoDiPack + Eigen + SuperLU).";

  m.def("version", &dfm::version, "Version string of the native engine.");

  nb::class_<dfm::ComplexSparseLU>(m, "ComplexSparseLU",
                                   "Sparse LU factorisation of a complex square matrix (CSC input). "
                                   "Factorise once, then solve for A x = b or A^H x = b.")
      .def(
          "__init__",
          [](dfm::ComplexSparseLU* self, const IndexArray& indptr, const IndexArray& indices,
             const ComplexArray& data) {
            new (self) dfm::ComplexSparseLU(build_csc(indptr, indices, data));
          },
          "indptr"_a, "indices"_a, "data"_a)
      .def_prop_ro("size", &dfm::ComplexSparseLU::size)
      .def(
          "solve",
          [](const dfm::ComplexSparseLU& self, const ComplexArray& rhs) {
            return to_numpy(to_std_vector(self.solve(to_vector(rhs, self.size()))));
          },
          "rhs"_a, "Solve A x = rhs.")
      .def(
          "solve_adjoint",
          [](const dfm::ComplexSparseLU& self, const ComplexArray& rhs) {
            return to_numpy(
                to_std_vector(self.solve_conjugate_transpose(to_vector(rhs, self.size()))));
          },
          "rhs"_a, "Solve A^H x = rhs using the same factors (no refactorisation).");

  m.def(
      "complex_gradient_smoke",
      [](const std::vector<double>& p) {
        const dfm::GradientSmokeResult r = dfm::complex_gradient_smoke(p);
        return nb::make_tuple(r.value, to_numpy(r.gradient), r.max_abs_error);
      },
      "p"_a,
      "Reverse-mode CoDiPack gradient of a complex Eigen expression. Returns "
      "(value, gradient, max_abs_error versus the analytic result).");
}
