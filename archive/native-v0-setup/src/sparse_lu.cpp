#include "dfm/sparse_lu.hpp"

#include <slu_zdefs.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace dfm {

// SuperLU owns L and U (allocated by zgssv) and must release them with its
// own destructors. Everything else is plain C++ storage that outlives the
// SuperMatrix views pointing into it.
struct ComplexSparseLU::Impl {
  int n = 0;
  bool factors_allocated = false;
  SuperMatrix L{};
  SuperMatrix U{};
  std::vector<int> perm_c;
  std::vector<int> perm_r;

  ~Impl() {
    if (factors_allocated) {
      Destroy_SuperNode_Matrix(&L);
      Destroy_CompCol_Matrix(&U);
    }
  }

  VectorC solve(const VectorC& rhs, trans_t trans) {
    if (rhs.size() != n) {
      throw std::invalid_argument("ComplexSparseLU: right-hand side has size " +
                                  std::to_string(rhs.size()) + ", expected " + std::to_string(n));
    }
    if (n == 0) {
      return VectorC(0);
    }

    VectorC x = rhs;  // zgstrs overwrites its right-hand side in place
    SuperMatrix bmat;
    zCreate_Dense_Matrix(&bmat, n, 1, reinterpret_cast<doublecomplex*>(x.data()), n, SLU_DN,
                         SLU_Z, SLU_GE);

    SuperLUStat_t stat;
    StatInit(&stat);
    int info = 0;
    zgstrs(trans, &L, &U, perm_c.data(), perm_r.data(), &bmat, &stat, &info);
    StatFree(&stat);
    Destroy_SuperMatrix_Store(&bmat);

    if (info != 0) {
      throw std::runtime_error("ComplexSparseLU: zgstrs failed with info " + std::to_string(info));
    }
    return x;
  }
};

ComplexSparseLU::ComplexSparseLU(const SparseMatrixC& matrix) : impl_(std::make_unique<Impl>()) {
  if (matrix.rows() != matrix.cols()) {
    throw std::invalid_argument("ComplexSparseLU: matrix must be square");
  }

  // Work on a compressed copy: SuperLU reads raw CSC arrays.
  SparseMatrixC a = matrix;
  a.makeCompressed();

  const int n = static_cast<int>(a.rows());
  impl_->n = n;
  if (n == 0) {
    return;
  }

  const int nnz = static_cast<int>(a.nonZeros());
  std::vector<int> colptr(a.outerIndexPtr(), a.outerIndexPtr() + n + 1);
  std::vector<int> rowind(a.innerIndexPtr(), a.innerIndexPtr() + nnz);
  std::vector<Complex> values(a.valuePtr(), a.valuePtr() + nnz);

  SuperMatrix amat;
  zCreate_CompCol_Matrix(&amat, n, n, nnz, reinterpret_cast<doublecomplex*>(values.data()),
                         rowind.data(), colptr.data(), SLU_NC, SLU_Z, SLU_GE);

  // zgssv factorises and also overwrites B with a solution. We only need the
  // factors here, so B is a throwaway zero right-hand side.
  std::vector<Complex> scratch(static_cast<std::size_t>(n), Complex(0.0, 0.0));
  SuperMatrix bmat;
  zCreate_Dense_Matrix(&bmat, n, 1, reinterpret_cast<doublecomplex*>(scratch.data()), n, SLU_DN,
                       SLU_Z, SLU_GE);

  superlu_options_t options;
  set_default_options(&options);
  options.ColPerm = COLAMD;  // fill-reducing ordering, chosen by SuperLU

  impl_->perm_c.assign(static_cast<std::size_t>(n), 0);
  impl_->perm_r.assign(static_cast<std::size_t>(n), 0);

  SuperLUStat_t stat;
  StatInit(&stat);
  int info = 0;
  zgssv(&options, &amat, impl_->perm_c.data(), impl_->perm_r.data(), &impl_->L, &impl_->U, &bmat,
        &stat, &info);
  StatFree(&stat);
  Destroy_SuperMatrix_Store(&bmat);
  Destroy_SuperMatrix_Store(&amat);

  // On failure SuperLU may still have allocated L and U, so mark them for
  // release before throwing (the Impl destructor runs during unwinding).
  impl_->factors_allocated = (info >= 0);
  if (info > 0) {
    throw std::runtime_error("ComplexSparseLU: matrix is singular (zero pivot at column " +
                             std::to_string(info) + ")");
  }
  if (info < 0) {
    throw std::runtime_error("ComplexSparseLU: SuperLU zgssv argument " + std::to_string(-info) +
                             " had an illegal value");
  }
}

ComplexSparseLU::~ComplexSparseLU() = default;
ComplexSparseLU::ComplexSparseLU(ComplexSparseLU&&) noexcept = default;
ComplexSparseLU& ComplexSparseLU::operator=(ComplexSparseLU&&) noexcept = default;

int ComplexSparseLU::size() const noexcept { return impl_->n; }

VectorC ComplexSparseLU::solve(const VectorC& rhs) const {
  return impl_->solve(rhs, NOTRANS);
}

VectorC ComplexSparseLU::solve_conjugate_transpose(const VectorC& rhs) const {
  // CONJ solves A^H x = b with the same L and U, i.e. no refactorisation.
  return impl_->solve(rhs, CONJ);
}

}  // namespace dfm
