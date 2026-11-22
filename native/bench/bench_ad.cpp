// CoDiPack record vs replay benchmark for the differentiable sparse solve.
//
// Recording (building the tape while running the forward computation) and
// replay (the reverse sweep over the tape) are timed separately, so changes
// to either side are visible on their own. A plain passive solve chain is
// timed as a baseline with no tape at all.
//
// Usage: dfm_bench_ad [n] [chain] [reps]
//   n      matrix size (default 751, the optical fixture's signal matrix size)
//   chain  number of dependent solves per tape (default 8)
//   reps   timed repetitions, median reported (default 5)

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "dfm/ad_sparse_solve.hpp"
#include "dfm/sparse_lu.hpp"

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

// Diagonally dominant complex matrix with a banded structure, similar in
// sparsity to the optical matrices (a few couplings per row).
dfm::SparseMatrixC make_matrix(int n) {
  std::vector<Eigen::Triplet<dfm::Complex>> t;
  t.reserve(static_cast<std::size_t>(5 * n));
  for (int i = 0; i < n; ++i) {
    t.emplace_back(i, i, dfm::Complex(4.0 + 0.001 * i, 0.5));
    if (i + 1 < n) {
      t.emplace_back(i, i + 1, dfm::Complex(-1.0, 0.3));
      t.emplace_back(i + 1, i, dfm::Complex(0.2, -0.7));
    }
    if (i + 7 < n) {
      t.emplace_back(i, i + 7, dfm::Complex(0.0, 0.4));
    }
  }
  dfm::SparseMatrixC a(n, n);
  a.setFromTriplets(t.begin(), t.end());
  a.makeCompressed();
  return a;
}

}  // namespace

int main(int argc, char** argv) {
  const int n = argc > 1 ? std::atoi(argv[1]) : 751;
  const int chain = argc > 2 ? std::atoi(argv[2]) : 8;
  const int reps = argc > 3 ? std::atoi(argv[3]) : 5;

  const dfm::SparseMatrixC a = make_matrix(n);
  const dfm::ComplexSparseLU lu(a);

  dfm::VectorC b0(n);
  for (int i = 0; i < n; ++i) {
    b0(i) = dfm::Complex(1.0 + 0.01 * i, -0.2 * i / n);
  }

  // ---- baseline: passive chain, no tape --------------------------------
  std::vector<double> t_passive;
  double sink = 0.0;
  for (int r = 0; r < reps; ++r) {
    const auto start = Clock::now();
    dfm::VectorC y = b0;
    for (int k = 0; k < chain; ++k) {
      lu.solve_inplace(y);
    }
    sink += y(0).real();
    t_passive.push_back(seconds_since(start));
  }

  // ---- AD: record and replay timed separately ---------------------------
  using ActiveReal = dfm::ActiveReal;
  using Tape = ActiveReal::Tape;
  Tape& tape = ActiveReal::getTape();

  std::vector<double> t_record;
  std::vector<double> t_replay;
  for (int r = 0; r < reps; ++r) {
    tape.reset();

    // Recording: the forward pass runs on the active types and writes the
    // tape. Inputs are the interleaved real parts of b0.
    const auto rec_start = Clock::now();
    tape.setActive();
    std::vector<ActiveReal> inputs(static_cast<std::size_t>(2 * n));
    for (int i = 0; i < n; ++i) {
      inputs[static_cast<std::size_t>(2 * i)] = b0(i).real();
      inputs[static_cast<std::size_t>(2 * i + 1)] = b0(i).imag();
      tape.registerInput(inputs[static_cast<std::size_t>(2 * i)]);
      tape.registerInput(inputs[static_cast<std::size_t>(2 * i + 1)]);
    }
    dfm::ActiveVectorC y(n);
    for (int i = 0; i < n; ++i) {
      y(i) = dfm::ActiveComplex(inputs[static_cast<std::size_t>(2 * i)],
                                inputs[static_cast<std::size_t>(2 * i + 1)]);
    }
    for (int k = 0; k < chain; ++k) {
      y = dfm::solve_recorded(lu, y);
    }
    ActiveReal loss = 0.0;
    for (int i = 0; i < n; ++i) {
      loss += y(i).real() * y(i).real() + y(i).imag() * y(i).imag();
    }
    tape.registerOutput(loss);
    tape.setPassive();
    t_record.push_back(seconds_since(rec_start));

    // Replay: one reverse sweep. Adjoints are cleared outside the timed
    // region, because a real gradient loop does the same.
    loss.setGradient(1.0);
    const auto rep_start = Clock::now();
    tape.evaluate();
    t_replay.push_back(seconds_since(rep_start));
    sink += inputs[0].getGradient();
  }

  std::printf("n=%d chain=%d reps=%d\n", n, chain, reps);
  std::printf("passive solve chain   median %.6f s\n", median(t_passive));
  std::printf("tape record (forward) median %.6f s\n", median(t_record));
  std::printf("tape replay (reverse) median %.6f s\n", median(t_replay));
  // Prevents the optimiser from removing the timed work.
  std::printf("checksum %.6e\n", sink);
  tape.reset();
  return 0;
}
