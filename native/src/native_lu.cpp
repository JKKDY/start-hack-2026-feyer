#include "native_lu.hpp"

#include <slu_zdefs.h>

#include <algorithm>
#include <cstddef>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace dfm::native_lu {
namespace {

constexpr Complex kZero(0.0, 0.0);

bool finite(Complex z) { return std::isfinite(z.real()) && std::isfinite(z.imag()); }

struct OrderEntry {
  std::vector<int> colptr, rowind;
  std::shared_ptr<const std::vector<int>> order;
};

}  // namespace

std::shared_ptr<const std::vector<int>> column_order(int n, int nnz, const int* colptr, const int* rowind) {
  static std::mutex mutex;
  static std::vector<OrderEntry> cache;
  {
    std::lock_guard<std::mutex> lock(mutex);
    for (const OrderEntry& e : cache) {
      if (static_cast<int>(e.colptr.size()) == n + 1 && static_cast<int>(e.rowind.size()) == nnz &&
          std::equal(e.colptr.begin(), e.colptr.end(), colptr) &&
          std::equal(e.rowind.begin(), e.rowind.end(), rowind)) {
        return e.order;
      }
    }
  }
  // COLAMD via SuperLU (values are not read).
  std::vector<int> cp(colptr, colptr + n + 1), ri(rowind, rowind + nnz);
  std::vector<Complex> dummy(static_cast<std::size_t>(std::max(nnz, 1)));
  SuperMatrix amat;
  zCreate_CompCol_Matrix(&amat, n, n, nnz, reinterpret_cast<doublecomplex*>(dummy.data()), ri.data(), cp.data(),
                         SLU_NC, SLU_Z, SLU_GE);
  std::vector<int> perm_c(static_cast<std::size_t>(n));
  get_perm_c(COLAMD, &amat, perm_c.data());
  Destroy_SuperMatrix_Store(&amat);
  // perm_c[j] is the new position of column j; invert it to the list of
  // columns in factorisation order.
  auto order = std::make_shared<std::vector<int>>(static_cast<std::size_t>(n));
  for (int j = 0; j < n; ++j) (*order)[static_cast<std::size_t>(perm_c[static_cast<std::size_t>(j)])] = j;
  std::lock_guard<std::mutex> lock(mutex);
  if (cache.size() >= 16) cache.erase(cache.begin());
  cache.push_back(OrderEntry{std::vector<int>(colptr, colptr + n + 1), std::vector<int>(rowind, rowind + nnz), order});
  return order;
}

void factor(int n, const int* colptr, const int* rowind, const Complex* values,
            std::shared_ptr<const std::vector<int>> order, Factors& f) {
  const int nnz = colptr[n];
  f.n = n;
  f.q = std::move(order);
  const int* q = f.q->data();
  f.pinv.assign(static_cast<std::size_t>(n), -1);
  f.lp.assign(static_cast<std::size_t>(n) + 1, 0);
  f.up.assign(static_cast<std::size_t>(n) + 1, 0);
  const std::size_t guess = static_cast<std::size_t>(2) * nnz + n;
  f.li.clear();
  f.lx.clear();
  f.ui.clear();
  f.ux.clear();
  f.li.reserve(guess);
  f.lx.reserve(guess);
  f.ui.reserve(guess);
  f.ux.reserve(guess);
  f.udiag_inv.resize(static_cast<std::size_t>(n));

  std::vector<int>& pinv = f.pinv;
  std::vector<int>& lp = f.lp;
  std::vector<int>& li = f.li;  // original row ids until the end
  std::vector<Complex>& lx = f.lx;
  std::vector<Complex> x(static_cast<std::size_t>(n), kZero);
  std::vector<int> xi(static_cast<std::size_t>(n));  // DFS stack (bottom) and reach output (top..n)
  std::vector<int> pstack(static_cast<std::size_t>(n));
  std::vector<int> mark(static_cast<std::size_t>(n), -1);

  for (int k = 0; k < n; ++k) {
    const int col = q[k];
    // Symbolic: rows reachable from A(:, col) through the L columns so far,
    // in topological order xi[top..n).
    int top = n;
    for (int s = colptr[col]; s < colptr[col + 1]; ++s) {
      const int start = rowind[s];
      if (mark[start] == k) continue;
      int head = 0;
      xi[0] = start;
      while (head >= 0) {
        const int j = xi[head];
        const int jnew = pinv[j];
        if (mark[j] != k) {
          mark[j] = k;
          pstack[head] = jnew < 0 ? 0 : lp[jnew] + 1;
        }
        bool done = true;
        const int end = jnew < 0 ? 0 : lp[jnew + 1];
        for (int p = pstack[head]; p < end; ++p) {
          const int i = li[static_cast<std::size_t>(p)];
          if (mark[i] == k) continue;
          pstack[head] = p + 1;
          xi[++head] = i;
          done = false;
          break;
        }
        if (done) {
          --head;
          xi[--top] = j;
        }
      }
    }

    // Numeric: x = L \ A(:, col) on the reached rows.
    for (int p = top; p < n; ++p) x[xi[p]] = kZero;
    for (int s = colptr[col]; s < colptr[col + 1]; ++s) x[rowind[s]] = values[s];
    for (int px = top; px < n; ++px) {
      const int j = xi[px];
      const int jj = pinv[j];
      if (jj < 0) continue;
      const Complex v = x[j];
      if (v == kZero) continue;
      for (int p = lp[jj] + 1; p < lp[jj + 1]; ++p) x[li[p]] -= lx[p] * v;
    }
    // Pivot, then U(:, k) (diagonal last) and L(:, k) (unit diagonal first).
    int ipiv = -1;
    double best = -1.0;
    for (int p = top; p < n; ++p) {
      const int i = xi[p];
      if (pinv[i] < 0) {
        const double a = pivot_key(x[i]);
        if (a > best) {
          best = a;
          ipiv = i;
        }
      } else {
        f.ui.push_back(pinv[i]);
        f.ux.push_back(x[i]);
      }
    }
    if (ipiv < 0 || !(best > 0.0) || !finite(x[ipiv])) {
      throw std::runtime_error("ComplexSparseLU: matrix is singular (zero pivot at column " +
                               std::to_string(k + 1) + ")");
    }
    if (col != ipiv && pinv[col] < 0 && mark[col] == k && pivot_key(x[col]) >= best) ipiv = col;
    const Complex pivot = x[ipiv];
    f.ui.push_back(k);
    f.ux.push_back(pivot);
    f.up[k + 1] = static_cast<int>(f.ui.size());
    const Complex inv_pivot = Complex(1.0, 0.0) / pivot;
    f.udiag_inv[k] = inv_pivot;
    pinv[ipiv] = k;
    li.push_back(ipiv);
    lx.push_back(Complex(1.0, 0.0));
    for (int p = top; p < n; ++p) {
      const int i = xi[p];
      if (pinv[i] < 0) {
        li.push_back(i);
        lx.push_back(x[i] * inv_pivot);
      }
      x[i] = kZero;
    }
    lp[k + 1] = static_cast<int>(li.size());
  }
  for (int& r : li) r = pinv[r];  // L rows in pivot order
}

void solve(const Factors& f, Complex* x, int nrhs, bool conjugate_transpose, Complex* t) {
  const int n = f.n;
  const int* lp = f.lp.data();
  const int* li = f.li.data();
  const int* up = f.up.data();
  const int* ui = f.ui.data();
  const int* pinv = f.pinv.data();
  const int* q = f.q->data();
  const Complex* lx = f.lx.data();
  const Complex* ux = f.ux.data();
  const Complex* dinv = f.udiag_inv.data();
  for (int c = 0; c < nrhs; ++c) {
    Complex* xc = x + static_cast<std::size_t>(c) * n;
    if (!conjugate_transpose) {
      // L U (Q^T x) = P b
      for (int i = 0; i < n; ++i) t[pinv[i]] = xc[i];
      for (int j = 0; j < n; ++j) {
        const Complex v = t[j];
        if (v == kZero) continue;
        for (int p = lp[j] + 1; p < lp[j + 1]; ++p) t[li[p]] -= lx[p] * v;
      }
      for (int j = n - 1; j >= 0; --j) {
        const int last = up[j + 1] - 1;
        const Complex v = t[j] * dinv[j];
        t[j] = v;
        if (v == kZero) continue;
        for (int p = up[j]; p < last; ++p) t[ui[p]] -= ux[p] * v;
      }
      for (int k = 0; k < n; ++k) xc[q[k]] = t[k];
    } else {
      // A^H = Q U^H L^H P:  U^H w = Q^T b,  L^H v = w,  x = P^T v
      for (int k = 0; k < n; ++k) t[k] = xc[q[k]];
      for (int k = 0; k < n; ++k) {
        const int last = up[k + 1] - 1;
        Complex acc = t[k];
        for (int p = up[k]; p < last; ++p) acc -= std::conj(ux[p]) * t[ui[p]];
        t[k] = acc * std::conj(dinv[k]);
      }
      for (int k = n - 1; k >= 0; --k) {
        Complex acc = t[k];
        for (int p = lp[k] + 1; p < lp[k + 1]; ++p) acc -= std::conj(lx[p]) * t[li[p]];
        t[k] = acc;
      }
      for (int i = 0; i < n; ++i) xc[i] = t[pinv[i]];
    }
  }
}

}  // namespace dfm::native_lu
