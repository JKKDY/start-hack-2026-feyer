#include "dfm/setup_spec.hpp"

#include <algorithm>
#include <unordered_set>

#include "dfm/errors.hpp"

namespace dfm {
namespace {

bool has_key(const Properties& props, const std::string& key) {
  return std::any_of(props.begin(), props.end(), [&](const auto& kv) { return kv.first == key; });
}

}  // namespace

const NodeSpec* SetupSpec::find_node(const std::string& name) const {
  for (const NodeSpec& n : nodes) {
    if (n.name == name) {
      return &n;
    }
  }
  return nullptr;
}

const EdgeSpec* SetupSpec::find_edge(const std::string& edge_name) const {
  for (const EdgeSpec& e : edges) {
    if (e.name() == edge_name) {
      return &e;
    }
  }
  return nullptr;
}

bool SetupSpec::has_property(const ParameterPair& pair) const {
  if (const NodeSpec* n = find_node(pair.first)) {
    return has_key(n->properties, pair.second);
  }
  if (const EdgeSpec* e = find_edge(pair.first)) {
    return has_key(e->properties, pair.second);
  }
  return false;
}

void SetupSpec::validate() const {
  std::unordered_set<std::string> names;
  for (const NodeSpec& n : nodes) {
    if (n.name.find('_') != std::string::npos) {
      throw ValidationError("Node name '" + n.name + "' cannot contain underscores. Use '-' instead.");
    }
    if (!names.insert(n.name).second) {
      throw ValidationError("Duplicate node name '" + n.name + "'.");
    }
  }
  for (const EdgeSpec& e : edges) {
    if (names.count(e.source) == 0) {
      throw ValidationError("Source node '" + e.source + "' is not in the setup.");
    }
    if (names.count(e.target) == 0) {
      throw ValidationError("Target node '" + e.target + "' is not in the setup.");
    }
  }
  // Upstream build() writes "<edge>_source" / "<edge>_target" into laser,
  // squeezer and detector targets; setups saved after a build (such as the
  // UIFO fixture) contain them, and upstream accepts them.
  auto is_edge_endpoint = [this](const std::string& t) {
    for (const char* suffix : {"_source", "_target"}) {
      const std::string sfx(suffix);
      if (t.size() > sfx.size() && t.compare(t.size() - sfx.size(), sfx.size(), sfx) == 0 &&
          find_edge(t.substr(0, t.size() - sfx.size())) != nullptr) {
        return true;
      }
    }
    return false;
  };
  for (const NodeSpec& n : nodes) {
    if (n.target && find_node(*n.target) == nullptr && find_edge(*n.target) == nullptr &&
        !is_edge_endpoint(*n.target)) {
      throw ValidationError("Target '" + *n.target + "' is not in the setup.");
    }
    if (n.detector1 && names.count(*n.detector1) == 0) {
      throw ValidationError("Detector1 '" + *n.detector1 + "' is not in the setup.");
    }
    if (n.detector2 && names.count(*n.detector2) == 0) {
      throw ValidationError("Detector2 '" + *n.detector2 + "' is not in the setup.");
    }
  }
  for (const ParameterPair& p : parameters) {
    if (!has_property(p)) {
      throw ValidationError("Parameter ('" + p.first + "', '" + p.second +
                            "') does not name a property of the setup.");
    }
  }
}

}  // namespace dfm
