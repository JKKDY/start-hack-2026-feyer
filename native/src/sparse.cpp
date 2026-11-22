#include "dfm/sparse.hpp"

#include <algorithm>
#include <numeric>
#include <string>

#include "dfm/errors.hpp"

namespace dfm {

SparsePattern::SparsePattern(int n, std::vector<int> col_ptr, std::vector<int> row_idx)
    : n_(n), col_ptr_(std::move(col_ptr)), row_idx_(std::move(row_idx)) {
  if (n_ < 0) throw ValidationError("SparsePattern: negative size");
  if (static_cast<int>(col_ptr_.size()) != n_ + 1) throw ValidationError("SparsePattern: col_ptr must have n+1 entries");
  if (col_ptr_.front() != 0 || col_ptr_.back() != static_cast<int>(row_idx_.size())) {
    throw ValidationError("SparsePattern: col_ptr must start at 0 and end at nnz");
  }
  for (int c = 0; c < n_; ++c) {
    if (col_ptr_[c] > col_ptr_[c + 1]) throw ValidationError("SparsePattern: col_ptr must be non-decreasing");
    for (int s = col_ptr_[c]; s < col_ptr_[c + 1]; ++s) {
      const int r = row_idx_[s];
      if (r < 0 || r >= n_) throw ValidationError("SparsePattern: row index out of range");
      if (s > col_ptr_[c] && row_idx_[s - 1] >= r) {
        throw ValidationError("SparsePattern: row indices must be sorted and unique within a column");
      }
    }
  }
}

std::pair<SparsePattern, std::vector<int>> SparsePattern::from_coordinates(int n, const std::vector<int>& rows,
                                                                           const std::vector<int>& cols) {
  if (rows.size() != cols.size()) throw ValidationError("from_coordinates: rows and cols differ in length");
  const std::size_t count = rows.size();
  std::vector<std::size_t> order(count);
  std::iota(order.begin(), order.end(), std::size_t{0});
  for (std::size_t i = 0; i < count; ++i) {
    if (rows[i] < 0 || rows[i] >= n || cols[i] < 0 || cols[i] >= n) {
      throw ValidationError("from_coordinates: coordinate out of range");
    }
  }
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    return cols[a] != cols[b] ? cols[a] < cols[b] : rows[a] < rows[b];
  });

  std::vector<int> col_ptr(static_cast<std::size_t>(n) + 1, 0);
  std::vector<int> row_idx;
  std::vector<int> slot_of(count, -1);
  row_idx.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    const std::size_t e = order[i];
    const bool duplicate = i > 0 && rows[order[i - 1]] == rows[e] && cols[order[i - 1]] == cols[e];
    if (!duplicate) {
      row_idx.push_back(rows[e]);
      col_ptr[static_cast<std::size_t>(cols[e]) + 1] += 1;
    }
    slot_of[e] = static_cast<int>(row_idx.size()) - 1;
  }
  for (int c = 0; c < n; ++c) col_ptr[c + 1] += col_ptr[c];
  return {SparsePattern(n, std::move(col_ptr), std::move(row_idx)), std::move(slot_of)};
}

int SparsePattern::find(int row, int col) const noexcept {
  if (col < 0 || col >= n_) return -1;
  const auto begin = row_idx_.begin() + col_ptr_[col];
  const auto end = row_idx_.begin() + col_ptr_[col + 1];
  const auto it = std::lower_bound(begin, end, row);
  return (it != end && *it == row) ? static_cast<int>(it - row_idx_.begin()) : -1;
}

std::vector<int> SparsePattern::slot_columns() const {
  std::vector<int> out(row_idx_.size());
  for (int c = 0; c < n_; ++c) {
    for (int s = col_ptr_[c]; s < col_ptr_[c + 1]; ++s) out[s] = c;
  }
  return out;
}

void sparse_matmul(const SparsePattern& p, const Complex* values, const Complex* x, int m, Complex* y) {
  const int n = p.size();
  const auto& cp = p.col_ptr();
  const auto& ri = p.row_idx();
  for (int c = 0; c < m; ++c) {
    const Complex* xc = x + static_cast<Index>(c) * n;
    Complex* yc = y + static_cast<Index>(c) * n;
    std::fill(yc, yc + n, Complex(0.0, 0.0));
    for (int j = 0; j < n; ++j) {
      const Complex xj = xc[j];
      for (int s = cp[j]; s < cp[j + 1]; ++s) yc[ri[s]] += values[s] * xj;
    }
  }
}

void sparse_matmul_adjoint(const SparsePattern& p, const Complex* values, const Complex* x, int m, Complex* y) {
  const int n = p.size();
  const auto& cp = p.col_ptr();
  const auto& ri = p.row_idx();
  for (int c = 0; c < m; ++c) {
    const Complex* xc = x + static_cast<Index>(c) * n;
    Complex* yc = y + static_cast<Index>(c) * n;
    for (int j = 0; j < n; ++j) {
      Complex acc(0.0, 0.0);
      for (int s = cp[j]; s < cp[j + 1]; ++s) acc += std::conj(values[s]) * xc[ri[s]];
      yc[j] = acc;
    }
  }
}

}  // namespace dfm
