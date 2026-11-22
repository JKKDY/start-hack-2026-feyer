#pragma once

// Sparse matrix storage independent of any optical geometry.
//
// A SparsePattern is the immutable structure (square CSC, sorted unique row
// indices per column). Numerical values live separately as a flat array
// aligned with the pattern's slots, so the same pattern serves every
// frequency, every parameter set and every AD pass. Patterns are shared via
// shared_ptr; the planner creates them once.

#include <memory>
#include <utility>
#include <vector>

#include "dfm/types.hpp"

namespace dfm {

class SparsePattern {
 public:
  SparsePattern() = default;

  /// From CSC arrays. Validates structure (sorted unique rows, bounds).
  SparsePattern(int n, std::vector<int> col_ptr, std::vector<int> row_idx);

  /// From (row, col) coordinates in any order, duplicates allowed. Returns
  /// the pattern and, for every input coordinate, the slot it maps to (equal
  /// coordinates share a slot). This is how assembly instructions are
  /// turned into value slots without geometry-specific code.
  static std::pair<SparsePattern, std::vector<int>> from_coordinates(int n, const std::vector<int>& rows,
                                                                     const std::vector<int>& cols);

  int size() const noexcept { return n_; }
  int nnz() const noexcept { return static_cast<int>(row_idx_.size()); }
  const std::vector<int>& col_ptr() const noexcept { return col_ptr_; }
  const std::vector<int>& row_idx() const noexcept { return row_idx_; }

  /// Slot of (row, col), or -1 if not in the pattern. Binary search.
  int find(int row, int col) const noexcept;

  /// Column of every slot (row index is row_idx()[slot]).
  std::vector<int> slot_columns() const;

 private:
  int n_ = 0;
  std::vector<int> col_ptr_{0};
  std::vector<int> row_idx_;
};

using SparsePatternPtr = std::shared_ptr<const SparsePattern>;

/// y = A x for a pattern-aligned value array (passive). x and y are n x m
/// column-major (column c starts at c * n).
void sparse_matmul(const SparsePattern& pattern, const Complex* values, const Complex* x, int m, Complex* y);

/// y = A^H x (passive), same layout.
void sparse_matmul_adjoint(const SparsePattern& pattern, const Complex* values, const Complex* x, int m,
                           Complex* y);

}  // namespace dfm
