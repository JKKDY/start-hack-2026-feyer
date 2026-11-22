#pragma once

// Complex forward-mode jet with K real seed directions.
//
// A Jet<K> carries a complex value v and K complex derivatives d_k = dv/dx_k
// with respect to K REAL inputs x_k. Arithmetic is complex, exactly like the
// upstream JAX kernels, which evaluate on complex128 parameters. Because the
// seed directions are real, the non-holomorphic conj() is still valid:
//     d/dx conj(f(x)) = conj(f'(x))   for real x.
//
// One evaluation with Jet<K> yields the full 4 x K Jacobian of a component
// kernel. That Jacobian is then pushed to the CoDiPack tape as one statement
// per real output, so the kernel's intermediate operations are never
// recorded (no scalar tape expansion).
//
// K is a compile-time constant (the kernel arity, at most 7), so every loop
// below has a fixed trip count and is fully unrolled by the compiler. Jets
// live on the stack; nothing allocates.

#include <array>
#include <complex>

#include "dfm/types.hpp"

namespace dfm {

template <int K>
struct Jet {
  Complex v{};
  std::array<Complex, K> d{};

  constexpr Jet() = default;
  constexpr Jet(Complex value) : v(value) {}  // NOLINT: implicit constant lift
  constexpr Jet(double value) : v(value, 0.0) {}  // NOLINT

  /// Input variable k with value x (real seed direction e_k).
  static Jet variable(double x, int k) {
    Jet j(x);
    j.d[static_cast<std::size_t>(k)] = Complex(1.0, 0.0);
    return j;
  }
};

namespace jet_detail {

// Applies the chain rule for a scalar function with value fv and derivative
// fd at the jet's value: result = (fv, fd * d).
template <int K>
inline Jet<K> chain(const Jet<K>& a, const Complex& fv, const Complex& fd) {
  Jet<K> r(fv);
  for (int k = 0; k < K; ++k) r.d[k] = fd * a.d[k];
  return r;
}

}  // namespace jet_detail

// ------------------------------------------------------------ arithmetic
template <int K>
inline Jet<K> operator-(const Jet<K>& a) {
  Jet<K> r(-a.v);
  for (int k = 0; k < K; ++k) r.d[k] = -a.d[k];
  return r;
}

template <int K>
inline Jet<K> operator+(const Jet<K>& a, const Jet<K>& b) {
  Jet<K> r(a.v + b.v);
  for (int k = 0; k < K; ++k) r.d[k] = a.d[k] + b.d[k];
  return r;
}

template <int K>
inline Jet<K> operator-(const Jet<K>& a, const Jet<K>& b) {
  Jet<K> r(a.v - b.v);
  for (int k = 0; k < K; ++k) r.d[k] = a.d[k] - b.d[k];
  return r;
}

template <int K>
inline Jet<K> operator*(const Jet<K>& a, const Jet<K>& b) {
  Jet<K> r(a.v * b.v);
  for (int k = 0; k < K; ++k) r.d[k] = a.d[k] * b.v + a.v * b.d[k];
  return r;
}

template <int K>
inline Jet<K> operator/(const Jet<K>& a, const Jet<K>& b) {
  // The value uses true complex division so that jet and plain-complex
  // evaluations produce bit-identical primals.
  const Complex q = a.v / b.v;
  const Complex inv = Complex(1.0, 0.0) / b.v;
  Jet<K> r(q);
  for (int k = 0; k < K; ++k) r.d[k] = (a.d[k] - q * b.d[k]) * inv;
  return r;
}

// Mixed operations with constants (complex or real) avoid building zero
// derivative arrays.
template <int K>
inline Jet<K> operator+(const Jet<K>& a, const Complex& c) {
  Jet<K> r = a;
  r.v += c;
  return r;
}
template <int K>
inline Jet<K> operator+(const Complex& c, const Jet<K>& a) {
  return a + c;
}
template <int K>
inline Jet<K> operator-(const Jet<K>& a, const Complex& c) {
  Jet<K> r = a;
  r.v -= c;
  return r;
}
template <int K>
inline Jet<K> operator-(const Complex& c, const Jet<K>& a) {
  return -a + c;
}
template <int K>
inline Jet<K> operator*(const Jet<K>& a, const Complex& c) {
  Jet<K> r(a.v * c);
  for (int k = 0; k < K; ++k) r.d[k] = a.d[k] * c;
  return r;
}
template <int K>
inline Jet<K> operator*(const Complex& c, const Jet<K>& a) {
  Jet<K> r(c * a.v);
  for (int k = 0; k < K; ++k) r.d[k] = c * a.d[k];
  return r;
}
template <int K>
inline Jet<K> operator/(const Jet<K>& a, const Complex& c) {
  Jet<K> r(a.v / c);
  for (int k = 0; k < K; ++k) r.d[k] = a.d[k] / c;
  return r;
}
template <int K>
inline Jet<K> operator/(const Complex& c, const Jet<K>& a) {
  return Jet<K>(c) / a;
}

// Real scalars promote to Complex so that `2.0 * x` works for both
// Complex and Jet kernels.
template <int K>
inline Jet<K> operator+(const Jet<K>& a, double c) { return a + Complex(c, 0.0); }
template <int K>
inline Jet<K> operator+(double c, const Jet<K>& a) { return a + Complex(c, 0.0); }
template <int K>
inline Jet<K> operator-(const Jet<K>& a, double c) { return a - Complex(c, 0.0); }
template <int K>
inline Jet<K> operator-(double c, const Jet<K>& a) { return Complex(c, 0.0) - a; }
template <int K>
inline Jet<K> operator*(const Jet<K>& a, double c) { return a * Complex(c, 0.0); }
template <int K>
inline Jet<K> operator*(double c, const Jet<K>& a) { return Complex(c, 0.0) * a; }
template <int K>
inline Jet<K> operator/(const Jet<K>& a, double c) { return a / Complex(c, 0.0); }
template <int K>
inline Jet<K> operator/(double c, const Jet<K>& a) { return Complex(c, 0.0) / a; }

// ------------------------------------------------------------- functions
// Found by ADL from kernels written as `using std::exp; exp(x);`.
template <int K>
inline Jet<K> exp(const Jet<K>& a) {
  const Complex e = std::exp(a.v);
  return jet_detail::chain(a, e, e);
}

template <int K>
inline Jet<K> sqrt(const Jet<K>& a) {
  const Complex s = std::sqrt(a.v);
  return jet_detail::chain(a, s, Complex(0.5, 0.0) / s);
}

template <int K>
inline Jet<K> sin(const Jet<K>& a) {
  return jet_detail::chain(a, std::sin(a.v), std::cos(a.v));
}

template <int K>
inline Jet<K> cos(const Jet<K>& a) {
  return jet_detail::chain(a, std::cos(a.v), -std::sin(a.v));
}

template <int K>
inline Jet<K> sinh(const Jet<K>& a) {
  return jet_detail::chain(a, std::sinh(a.v), std::cosh(a.v));
}

template <int K>
inline Jet<K> cosh(const Jet<K>& a) {
  return jet_detail::chain(a, std::cosh(a.v), std::sinh(a.v));
}

template <int K>
inline Jet<K> asin(const Jet<K>& a) {
  // d/dz asin(z) = 1 / sqrt(1 - z^2), on the same branch as std::asin.
  const Complex one(1.0, 0.0);
  return jet_detail::chain(a, std::asin(a.v), one / std::sqrt(one - a.v * a.v));
}

template <int K>
inline Jet<K> conj(const Jet<K>& a) {
  Jet<K> r(std::conj(a.v));
  for (int k = 0; k < K; ++k) r.d[k] = std::conj(a.d[k]);
  return r;
}

}  // namespace dfm
