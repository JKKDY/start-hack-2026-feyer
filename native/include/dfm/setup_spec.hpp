#pragma once

// Immutable native snapshot of a Python `Setup`.
//
// The Python `Setup` stays a mutable Python object (its nested dicts are
// mutated directly by user code and by `utils.set_value`). When a plan is
// built, the setup is converted once into this plain C++ value. After that
// the native side never looks at Python objects again.
//
// Order is significant everywhere: node, edge and parameter order drive port
// numbering and coefficient indexing, exactly as in upstream `build.py`.

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "dfm/link.hpp"

namespace dfm {

/// A property linked to another parameter: value = function(source).
/// `source` is a parameter name such as "l0_power" (node "l0", property
/// "power") as written in upstream setups.
struct LinkedValue {
  std::string source;
  LinkFunction function = LinkFunction::Identity;
};

/// A property value is either a number or a link.
using PropertyValue = std::variant<double, LinkedValue>;

/// Ordered (name, value) list. Upstream uses insertion-ordered dicts.
using Properties = std::vector<std::pair<std::string, PropertyValue>>;

struct NodeSpec {
  std::string name;
  std::string component;  ///< key of DEFAULT_PROPERTIES, e.g. "mirror"
  Properties properties;
  std::optional<std::string> target;           ///< node or "source_target" edge
  std::optional<std::string> target_property;  ///< "amplitude" or "frequency"
  std::optional<std::string> port;             ///< left/top/right/bottom
  std::optional<std::string> direction;        ///< in/out
  std::optional<bool> auxiliary;
  std::optional<std::string> detector1;
  std::optional<std::string> detector2;
  std::optional<std::string> sideband;         ///< detectors: "upper" (default) or "lower"
};

struct EdgeSpec {
  std::string source;
  std::string target;
  Properties properties;
  std::string source_port = "right";
  std::string target_port = "left";

  /// Upstream's external edge name, "source_target".
  std::string name() const { return source + "_" + target; }
};

/// (component_identifier, property_name). For edges the identifier is
/// "source_target".
using ParameterPair = std::pair<std::string, std::string>;

struct SetupSpec {
  std::vector<NodeSpec> nodes;
  std::vector<EdgeSpec> edges;
  std::vector<ParameterPair> parameters;

  /// Index lookup; returns nullptr when absent. Linear scans are fine here:
  /// they run once at plan time, never in the simulation loop.
  const NodeSpec* find_node(const std::string& name) const;
  const EdgeSpec* find_edge(const std::string& edge_name) const;

  /// True if (identifier, property) names an existing node or edge property.
  bool has_property(const ParameterPair& pair) const;

  /// Structural checks mirroring upstream `Setup` invariants: unique node
  /// names without underscores, edges between existing nodes, targets and
  /// detector references that resolve. Throws ValidationError.
  void validate() const;
};

using SetupSpecPtr = std::shared_ptr<const SetupSpec>;

}  // namespace dfm
