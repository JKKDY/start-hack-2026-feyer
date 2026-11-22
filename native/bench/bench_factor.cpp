// Sparse LU variants on real optical signal matrices.
//
// Input: a binary dump (n, nnz, count, indptr[n+1], indices[nnz],
// values[count * nnz] complex128, rhs[count * n] complex128), e.g. the 101
// scored-UIFO signal matrices. Times factorisation and one solve + one
// adjoint solve per matrix for:
//   superlu_gssv    current ComplexSparseLU path (COLAMD every time, zgssv)
//   superlu_cached  COLAMD ordering computed once, then sp_preorder + zgstrf
//   eigen_sparselu  Eigen::SparseLU, analyzePattern once, factorize per matrix
// and reports the worst relative residual of each (no refinement).
//
// Usage: dfm_bench_factor <file> [reps]

#include <slu_zdefs.h>

#include <Eigen/OrderingMethods>

#include "dfm/sparse.hpp"
#include "dfm/sparse_lu.hpp"
#include <Eigen/SparseLU>

#include <algorithm>
#include <chrono>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

namespace {

using C = std::complex<double>;
using Clock = std::chrono::steady_clock;
using SpMat = Eigen::SparseMatrix<C, Eigen::ColMajor, int>;
using Vec = Eigen::Matrix<C, Eigen::Dynamic, 1>;

struct Dump {
  int n = 0, nnz = 0, count = 0;
  std::vector<int> indptr, indices;
  std::vector<C> values, rhs;
};

Dump load(const char* path) {
  Dump d;
  std::ifstream f(path, std::ios::binary);
  int head[3];
  f.read(reinterpret_cast<char*>(head), sizeof head);
  d.n = head[0];
  d.nnz = head[1];
  d.count = head[2];
  d.indptr.resize(d.n + 1);
  d.indices.resize(d.nnz);
  d.values.resize(static_cast<std::size_t>(d.count) * d.nnz);
  d.rhs.resize(static_cast<std::size_t>(d.count) * d.n);
  f.read(reinterpret_cast<char*>(d.indptr.data()), sizeof(int) * d.indptr.size());
  f.read(reinterpret_cast<char*>(d.indices.data()), sizeof(int) * d.indices.size());
  f.read(reinterpret_cast<char*>(d.values.data()), sizeof(C) * d.values.size());
  f.read(reinterpret_cast<char*>(d.rhs.data()), sizeof(C) * d.rhs.size());
  return d;
}

double seconds(Clock::time_point a) { return std::chrono::duration<double>(Clock::now() - a).count(); }

double residual(const Dump& d, int k, const Vec& x, const Vec& b, bool adjoint) {
  const C* v = d.values.data() + static_cast<std::size_t>(k) * d.nnz;
  Vec r = b;
  for (int j = 0; j < d.n; ++j) {
    for (int s = d.indptr[j]; s < d.indptr[j + 1]; ++s) {
      const int i = d.indices[s];
      if (!adjoint) r(i) -= v[s] * x(j);
      else r(j) -= std::conj(v[s]) * x(i);
    }
  }
  double amax = 0;
  for (int s = 0; s < d.nnz; ++s) amax = std::max(amax, std::abs(v[s]));
  return r.norm() / (amax * x.norm() + b.norm());
}

struct Result {
  double factor = 0, solve = 0, worst = 0;
  long fill = 0;  // nnz(L) + nnz(U) of the last matrix
  unsigned long long hash = 1469598103934665603ull;
};

// ---------------------------------------------------------------- SuperLU
Result run_superlu(const Dump& d, bool cached, colperm_t ordering = COLAMD) {
  Result res;
  std::vector<int> perm_c(d.n), perm_r(d.n), etree(d.n);
  superlu_options_t opt;
  set_default_options(&opt);
  opt.ColPerm = ordering;
  std::vector<C> vals(d.nnz);
  std::vector<int> rowind(d.indices), colptr(d.indptr);
  if (cached) {
    SuperMatrix A;
    std::copy(d.values.begin(), d.values.begin() + d.nnz, vals.begin());
    zCreate_CompCol_Matrix(&A, d.n, d.n, d.nnz, reinterpret_cast<doublecomplex*>(vals.data()), rowind.data(),
                           colptr.data(), SLU_NC, SLU_Z, SLU_GE);
    get_perm_c(ordering, &A, perm_c.data());  // once per pattern
    Destroy_SuperMatrix_Store(&A);
  }
  for (int k = 0; k < d.count; ++k) {
    std::copy(d.values.begin() + static_cast<std::size_t>(k) * d.nnz,
              d.values.begin() + static_cast<std::size_t>(k + 1) * d.nnz, vals.begin());
    SuperMatrix A, L, U;
    zCreate_CompCol_Matrix(&A, d.n, d.n, d.nnz, reinterpret_cast<doublecomplex*>(vals.data()), rowind.data(),
                           colptr.data(), SLU_NC, SLU_Z, SLU_GE);
    SuperLUStat_t stat;
    StatInit(&stat);
    int info = 0;
    auto t = Clock::now();
    if (!cached) {
      std::vector<C> zero(d.n);
      SuperMatrix B;
      zCreate_Dense_Matrix(&B, d.n, 1, reinterpret_cast<doublecomplex*>(zero.data()), d.n, SLU_DN, SLU_Z, SLU_GE);
      zgssv(&opt, &A, perm_c.data(), perm_r.data(), &L, &U, &B, &stat, &info);
      Destroy_SuperMatrix_Store(&B);
    } else {
      SuperMatrix AC;
      GlobalLU_t glu;
      sp_preorder(&opt, &A, perm_c.data(), etree.data(), &AC);
      int_t finfo = 0;
      zgstrf(&opt, &AC, sp_ienv(2), sp_ienv(1), etree.data(), nullptr, 0, perm_c.data(), perm_r.data(), &L, &U,
             &glu, &stat, &finfo);
      Destroy_CompCol_Permuted(&AC);
      info = static_cast<int>(finfo);
    }
    res.factor += seconds(t);
    if (info != 0) std::printf("superlu info %d at %d\n", info, k);
    Vec b = Eigen::Map<const Vec>(d.rhs.data() + static_cast<std::size_t>(k) * d.n, d.n);
    Vec x = b, y = b;
    t = Clock::now();
    for (int mode = 0; mode < 2; ++mode) {
      Vec& z = mode == 0 ? x : y;
      SuperMatrix B;
      zCreate_Dense_Matrix(&B, d.n, 1, reinterpret_cast<doublecomplex*>(z.data()), d.n, SLU_DN, SLU_Z, SLU_GE);
      int sinfo = 0;
      zgstrs(mode == 0 ? NOTRANS : CONJ, &L, &U, perm_c.data(), perm_r.data(), &B, &stat, &sinfo);
      Destroy_SuperMatrix_Store(&B);
    }
    res.solve += seconds(t);
    res.worst = std::max({res.worst, residual(d, k, x, b, false), residual(d, k, y, b, true)});
    res.fill = static_cast<SCformat*>(L.Store)->nnz + static_cast<NCformat*>(U.Store)->nnz;
    StatFree(&stat);
    Destroy_SuperMatrix_Store(&A);
    Destroy_SuperNode_Matrix(&L);
    Destroy_CompCol_Matrix(&U);
  }
  return res;
}

// ------------------------------------------------------------------ Eigen
Result run_eigen(const Dump& d) {
  Result res;
  Eigen::SparseLU<SpMat, Eigen::COLAMDOrdering<int>> lu;
  auto view = [&](int k) {
    return Eigen::Map<const SpMat>(d.n, d.n, d.nnz, d.indptr.data(), d.indices.data(),
                                   d.values.data() + static_cast<std::size_t>(k) * d.nnz);
  };
  SpMat A = view(0);
  auto t = Clock::now();
  lu.analyzePattern(A);  // once per pattern
  res.factor += seconds(t);
  for (int k = 0; k < d.count; ++k) {
    A = view(k);  // same pattern, new values (copy; mirrors filling slot values)
    t = Clock::now();
    lu.factorize(A);
    res.factor += seconds(t);
    if (lu.info() != Eigen::Success) std::printf("eigen failed at %d\n", k);
    Vec b = Eigen::Map<const Vec>(d.rhs.data() + static_cast<std::size_t>(k) * d.n, d.n);
    t = Clock::now();
    Vec x = lu.solve(b);
    Vec y = lu.adjoint().solve(b);
    res.solve += seconds(t);
    res.worst = std::max({res.worst, residual(d, k, x, b, false), residual(d, k, y, b, true)});
  }
  return res;
}

// ------------------------------------------------------------ dfm native
Result run_dfm(const Dump& d) {
  Result res;
  dfm::SparsePattern pattern(d.n, d.indptr, d.indices);
  for (int k = 0; k < d.count; ++k) {
    const C* v = d.values.data() + static_cast<std::size_t>(k) * d.nnz;
    auto t = Clock::now();
    dfm::ComplexSparseLU lu(pattern, v);
    res.factor += seconds(t);
    Vec b = Eigen::Map<const Vec>(d.rhs.data() + static_cast<std::size_t>(k) * d.n, d.n);
    Vec x = b, y = b;
    t = Clock::now();
    lu.solve_inplace(x.data(), 1);
    lu.solve_conjugate_transpose_inplace(y.data(), 1);
    res.solve += seconds(t);
    res.worst = std::max({res.worst, residual(d, k, x, b, false), residual(d, k, y, b, true)});
    for (const Vec* v : {&x, &y}) {
      const unsigned char* bytes = reinterpret_cast<const unsigned char*>(v->data());
      for (std::size_t i = 0; i < sizeof(C) * static_cast<std::size_t>(v->size()); ++i) res.hash = (res.hash ^ bytes[i]) * 1099511628211ull;
    }
  }
  std::printf("  dfm solution hash %016llx\n", res.hash);
  return res;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf("usage: %s <dump> [reps]\n", argv[0]);
    return 1;
  }
  const Dump d = load(argv[1]);
  const int reps = argc > 2 ? std::atoi(argv[2]) : 5;
  std::printf("n=%d nnz=%d matrices=%d reps=%d (best of reps; totals over all matrices)\n", d.n, d.nnz, d.count, reps);
  auto best = [&](auto fn) {
    Result b{1e9, 1e9, 0};
    for (int r = 0; r < reps; ++r) {
      const Result x = fn();
      b.factor = std::min(b.factor, x.factor);
      b.solve = std::min(b.solve, x.solve);
      b.worst = std::max(b.worst, x.worst);
      b.fill = x.fill;
    }
    return b;
  };
  auto show = [](const char* name, const Result& r) {
    std::printf("%-22s factor %8.2f ms  solve+adjoint %7.2f ms  worst residual %.2e  fill %ld\n", name,
                r.factor * 1e3, r.solve * 1e3, r.worst, r.fill);
  };
  show("superlu_gssv COLAMD", best([&] { return run_superlu(d, false); }));
  show("cached COLAMD", best([&] { return run_superlu(d, true, COLAMD); }));
  show("cached MMD_AT_PLUS_A", best([&] { return run_superlu(d, true, MMD_AT_PLUS_A); }));
  show("cached MMD_ATA", best([&] { return run_superlu(d, true, MMD_ATA); }));
  show("cached NATURAL", best([&] { return run_superlu(d, true, NATURAL); }));
  show("eigen_sparselu COLAMD", best([&] { return run_eigen(d); }));
  show("dfm ComplexSparseLU", best([&] { return run_dfm(d); }));
  return 0;
}
