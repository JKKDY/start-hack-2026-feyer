#pragma once

// Parameter links: a property whose value is f(other parameter).
//
// Upstream stores links as (source_parameter_name, python_callable) and calls
// the callable inside JAX. The native engine cannot call arbitrary Python in
// its hot path, so links use a closed set of differentiable functions. The
// Python layer maps known callables (numpy/jax/math sqrt, exp, ...) onto this
// enum. Evaluation is a switch, so it inlines and needs no indirection.

#include <cmath>
#include <optional>
#include <string_view>

namespace dfm {

enum class LinkFunction : int {
  Identity = 0,
  Sqrt,
  Square,
  Exp,
  Log,
  Sin,
  Cos,
  Negative,
  Abs,
};

/// Stable lower-case names, used by the bindings and in error messages.
constexpr std::string_view link_function_name(LinkFunction f) noexcept {
  switch (f) {
    case LinkFunction::Identity: return "identity";
    case LinkFunction::Sqrt: return "sqrt";
    case LinkFunction::Square: return "square";
    case LinkFunction::Exp: return "exp";
    case LinkFunction::Log: return "log";
    case LinkFunction::Sin: return "sin";
    case LinkFunction::Cos: return "cos";
    case LinkFunction::Negative: return "negative";
    case LinkFunction::Abs: return "abs";
  }
  return "unknown";
}

constexpr std::optional<LinkFunction> link_function_from_name(std::string_view name) noexcept {
  for (int i = 0; i <= static_cast<int>(LinkFunction::Abs); ++i) {
    const auto f = static_cast<LinkFunction>(i);
    if (link_function_name(f) == name) {
      return f;
    }
  }
  return std::nullopt;
}

/// f(x). Templated so it also runs on CoDiPack active types.
template <typename T>
inline T apply_link(LinkFunction f, const T& x) {
  using std::abs;
  using std::cos;
  using std::exp;
  using std::log;
  using std::sin;
  using std::sqrt;
  switch (f) {
    case LinkFunction::Identity: return x;
    case LinkFunction::Sqrt: return sqrt(x);
    case LinkFunction::Square: return x * x;
    case LinkFunction::Exp: return exp(x);
    case LinkFunction::Log: return log(x);
    case LinkFunction::Sin: return sin(x);
    case LinkFunction::Cos: return cos(x);
    case LinkFunction::Negative: return -x;
    case LinkFunction::Abs: return abs(x);
  }
  return x;
}

}  // namespace dfm
