#pragma once

// Exception types of the native engine. The bindings translate them to the
// matching Python exceptions:
//   ValidationError     -> ValueError        (bad input, same as upstream)
//   NotImplementedError -> NotImplementedError (milestone not reached yet)
//   StateError          -> RuntimeError      (stale or foreign state object)

#include <stdexcept>
#include <string>

namespace dfm {

struct ValidationError : std::invalid_argument {
  using std::invalid_argument::invalid_argument;
};

struct NotImplementedError : std::logic_error {
  using std::logic_error::logic_error;
};

struct StateError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

}  // namespace dfm
