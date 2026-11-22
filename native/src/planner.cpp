// Runtime planner: native port of upstream build.py (build, set_instructions,
// parameter_linking, pairs_to_arrays, prepare_arrays).
//
// The port follows upstream statement by statement so that port numbering,
// parameter order, instruction order and the resulting matrices match. Where
// upstream behaviour is surprising it is reproduced and marked "upstream:".
// The only intended difference is determinism: upstream iterates a Python
// set for unused ports (hash order); here they are processed sorted.

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "dfm/component_ops.hpp"
#include "dfm/components.hpp"
#include "dfm/errors.hpp"
#include "dfm/plan.hpp"
#include "dfm/plan_internals.hpp"

namespace dfm {

namespace {

using CF = ComponentFunction;
using Row = std::array<int, kComponentInputWidth>;

const std::map<std::string, int> kMatrixSizes = {
    {"mirror", 4}, {"beamsplitter", 8}, {"nothing", 4}, {"directional_beamsplitter", 8}};

// Insertion-ordered like the Python dicts.
const std::map<std::string, std::vector<std::pair<std::string, int>>> kPortDicts = {
    {"mirror", {{"left", 0}, {"right", 2}}},
    {"beamsplitter", {{"left", 0}, {"right", 4}, {"bottom", 6}, {"top", 2}}},
    {"nothing", {{"left", 0}, {"right", 2}}},
    {"directional_beamsplitter", {{"left", 0}, {"right", 4}, {"bottom", 6}, {"top", 2}}},
};

const std::map<std::string, std::vector<std::pair<std::string, double>>> kDefaultProperties = {
    {"frequency", {{"frequency", 1.0}}},
    {"laser", {{"power", 1.0}, {"phase", 0.0}}},
    {"squeezer", {{"db", 0.0}, {"angle", 90.0}}},
    {"mirror", {{"loss", 5e-6}, {"reflectivity", 0.5}, {"tuning", 0.0}}},
    {"beamsplitter", {{"loss", 5e-6}, {"reflectivity", 0.5}, {"tuning", 0.0}, {"alpha", 45.0}}},
    {"free_mass", {{"mass", 40.0}}},
    {"signal", {{"amplitude", 1.0}, {"phase", 0.0}}},
    {"space", {{"length", 0.0}, {"refractive_index", 1.0}}},
    {"detector", {}},
    {"qnoised", {}},
    {"qhd", {{"phase", 180.0}}},
    {"nothing", {}},
    {"directional_beamsplitter", {}},
};

std::optional<int> port_offset(const std::string& component, const std::string& port) {
  const auto it = kPortDicts.find(component);
  if (it == kPortDicts.end()) return std::nullopt;
  for (const auto& [name, offset] : it->second) {
    if (name == port) return offset;
  }
  return std::nullopt;
}

bool is_matrix_component(const std::string& c) { return kMatrixSizes.count(c) != 0; }
bool is_one_of(const std::string& c, std::initializer_list<const char*> names) {
  for (const char* n : names) {
    if (c == n) return true;
  }
  return false;
}

std::string strip_endpoint_suffix(std::string s) {
  // upstream: data["target"].replace("_source", "").replace("_target", "")
  for (const char* suffix : {"_source", "_target"}) {
    const std::string needle(suffix);
    std::size_t pos;
    while ((pos = s.find(needle)) != std::string::npos) s.erase(pos, needle.size());
  }
  return s;
}

// One upstream instruction dict.
struct Instr {
  std::vector<Row> inputs;
  std::vector<CF> functions;
  std::vector<int> out_cols, rows, cols;
  std::optional<std::vector<int>> out_rows;
  std::optional<std::vector<int>> carrier_indices, carrier_rows, carrier_cols;
};

// Python dict with insertion order and replace-in-place on reassignment.
struct InstrDict {
  std::vector<std::pair<std::string, Instr>> items;
  void set(const std::string& key, Instr value) {
    for (auto& [k, v] : items) {
      if (k == key) {
        v = std::move(value);
        return;
      }
    }
    items.emplace_back(key, std::move(value));
  }
  const Instr* find(const std::string& key) const {
    for (const auto& [k, v] : items) {
      if (k == key) return &v;
    }
    return nullptr;
  }
};

// Ordered multimap name -> list (Python defaultdict(list) insertion order).
struct OrderedLists {
  std::vector<std::pair<std::string, std::vector<std::string>>> items;
  void append(const std::string& key, const std::string& value) {
    for (auto& [k, v] : items) {
      if (k == key) {
        v.push_back(value);
        return;
      }
    }
    items.push_back({key, {value}});
  }
  const std::vector<std::string>* find(const std::string& key) const {
    for (const auto& [k, v] : items) {
      if (k == key) return &v;
    }
    return nullptr;
  }
};

Row row(std::initializer_list<int> values) {
  Row r{};
  int k = 0;
  for (int v : values) r[k++] = v;
  return r;
}

// upstream set_instructions with the system_size_for_sidebands expansion.
void set_instructions(InstrDict& dict, const std::string& key, std::vector<Row> inputs, std::vector<int> out_cols,
                      std::vector<int> rows, std::vector<int> cols, std::vector<CF> functions,
                      std::optional<std::vector<int>> out_rows = std::nullopt,
                      std::optional<std::vector<int>> carrier_indices = std::nullopt,
                      std::optional<std::vector<int>> carrier_rows = std::nullopt,
                      std::optional<std::vector<int>> carrier_cols = std::nullopt,
                      std::optional<int> sidebands = std::nullopt) {
  if (out_cols.size() != rows.size()) {
    throw ValidationError("internal: output column indices must match system matrix row indices");
  }
  if (sidebands) {
    const int s = *sidebands;
    auto expand = [s](std::vector<int>& r, std::vector<int>& c, std::vector<std::vector<int>*> payload) {
      const std::size_t n = r.size();
      std::vector<char> only_row(n), only_col(n), both(n), pure(n);
      for (std::size_t i = 0; i < n; ++i) {
        only_row[i] = r[i] < s && c[i] >= s;
        only_col[i] = c[i] < s && r[i] >= s;
        both[i] = r[i] < s && c[i] < s;
        pure[i] = r[i] >= s && c[i] >= s;
      }
      for (std::size_t i = 0; i < n; ++i) {
        if (pure[i]) {
          c[i] += s;
          r[i] += s;
        }
        if (only_row[i]) c[i] += s;
        if (only_col[i]) r[i] += s;
      }
      std::vector<int> nr = r, nc = c;
      std::vector<std::vector<int>> np;
      for (auto* p : payload) np.push_back(*p);
      auto append = [&](const std::vector<char>& mask, int dr, int dc) {
        for (std::size_t i = 0; i < n; ++i) {
          if (!mask[i]) continue;
          nr.push_back(r[i] + dr);
          nc.push_back(c[i] + dc);
          for (std::size_t q = 0; q < payload.size(); ++q) np[q].push_back((*payload[q])[i]);
        }
      };
      append(only_row, s, 0);
      append(only_col, 0, s);
      append(both, s, s);
      r = std::move(nr);
      c = std::move(nc);
      for (std::size_t q = 0; q < payload.size(); ++q) *payload[q] = std::move(np[q]);
    };
    if (out_rows) {
      expand(rows, cols, {&out_cols, &*out_rows});
    } else {
      expand(rows, cols, {&out_cols});
    }
    if (carrier_indices) {
      expand(*carrier_rows, *carrier_cols, {&*carrier_indices});
    }
  }
  Instr instr;
  instr.inputs = std::move(inputs);
  instr.functions = std::move(functions);
  instr.out_cols = std::move(out_cols);
  instr.rows = std::move(rows);
  instr.cols = std::move(cols);
  instr.out_rows = std::move(out_rows);
  instr.carrier_indices = std::move(carrier_indices);
  instr.carrier_rows = std::move(carrier_rows);
  instr.carrier_cols = std::move(carrier_cols);
  dict.set(key, std::move(instr));
}

// Mutable working copy of a node (upstream build mutates the setup dicts).
struct WNode {
  const NodeSpec* spec = nullptr;
  std::string component;
  std::optional<std::string> target, port, direction;
};

// A deferred constant-template write: either a constant or a kernel output.
struct TemplateOp {
  int row = 0;
  int col = 0;  // kRhsColumn for the right-hand side
  Complex constant{};
  bool from_kernel = false;
  CF function = CF::DummyFunction;
  Row inputs{};
  int output = 0;
};

std::uint64_t key_of(int r, int c, int n) {
  return static_cast<std::uint64_t>(r) * static_cast<std::uint64_t>(n + 1) +
         static_cast<std::uint64_t>(c < 0 ? n : c);
}

}  // namespace

// Friend of SimulationPlan; fills its private members.
struct PlanBuilder {
  static SimulationPlanPtr build(SetupSpecPtr setup, PlanRequest request);
};

namespace {

struct Builder {
  const SetupSpec& setup;
  const PlanRequest& request;
  SimulationPlan::Internals& out;
  PortMetadata& meta;

  // build() locals.
  int system_size = 0;
  int signal_size = 0;
  std::map<std::string, int> matrix_positions;
  InstrDict carrier_instructions, signal_instructions, noise_instructions;
  std::vector<double> parameters{0.0, 1.0, 0.0};  // F, DEFAULT_REFRACTIVE_INDEX, 0
  std::vector<std::string> parameter_names{"", "", ""};
  std::vector<std::string> signal_components;
  OrderedLists space_to_signals, laser_to_signals;
  std::map<std::string, int> parameter_positions;
  std::set<std::string> all_ports, used_ports;
  int quantum_detector_number = 0;
  std::vector<std::string> free_masses;
  int signal_frequency_position = 0;
  std::map<std::string, std::map<std::string, int>> surface_index_positions;
  std::vector<int> indices_to_link;
  std::vector<std::string> linked_names;
  std::vector<LinkFunction> link_functions;
  std::vector<TemplateOp> carrier_ops;
  std::vector<std::pair<std::string, WNode>> nodes;  // node order

  Builder(const SetupSpec& s, const PlanRequest& r, SimulationPlan::Internals& o, PortMetadata& m)
      : setup(s), request(r), out(o), meta(m) {
    for (const NodeSpec& n : setup.nodes) {
      WNode w;
      w.spec = &n;
      w.component = n.component;
      w.target = n.target;
      w.port = n.port;
      w.direction = n.direction;
      nodes.emplace_back(n.name, w);
    }
  }

  WNode* node(const std::string& name) {
    for (auto& [k, v] : nodes) {
      if (k == name) return &v;
    }
    return nullptr;
  }

  WNode& require_node(const std::string& name, const char* context) {
    WNode* n = node(name);
    if (!n) throw ValidationError(std::string(context) + ": node '" + name + "' is not in the setup.");
    return *n;
  }

  int position(const std::string& key) const {
    const auto it = matrix_positions.find(key);
    if (it == matrix_positions.end()) {
      throw ValidationError("'" + key + "' has no position in the system (unsupported connection).");
    }
    return it->second;
  }

  // load_defaults + lookup.
  PropertyValue property(const Properties& props, const std::string& component, const std::string& key) const {
    for (const auto& [k, v] : props) {
      if (k == key) return v;
    }
    const auto it = kDefaultProperties.find(component);
    if (it != kDefaultProperties.end()) {
      for (const auto& [k, v] : it->second) {
        if (k == key) return v;
      }
    }
    throw ValidationError("Component '" + component + "' has no property '" + key + "'.");
  }
  PropertyValue node_property(const WNode& n, const std::string& key) const {
    return property(n.spec->properties, n.component, key);
  }

  // upstream parameter_linking
  void link(std::initializer_list<PropertyValue> values) {
    for (const PropertyValue& v : values) {
      if (const double* d = std::get_if<double>(&v)) {
        parameters.push_back(*d);
      } else {
        const LinkedValue& l = std::get<LinkedValue>(v);
        parameters.push_back(0.0);
        link_functions.push_back(l.function);
        indices_to_link.push_back(static_cast<int>(parameters.size()) - 1);
        linked_names.push_back(l.source);
      }
    }
  }

  int last(int back) const { return static_cast<int>(parameters.size()) - back; }

  void run() {
    edges_first_pass();
    nodes_first_pass();
    if (signal_size == 0) signal_size = 1;
    const int s = system_size;
    const int n_signal = 2 * s + signal_size;
    out.system_size = s;
    out.signal_size = signal_size;
    out.noise_selection = MatrixC::Zero(quantum_detector_number, n_signal);
    nodes_second_pass();
    laser_modulation();
    free_mass_instructions();
    edges_second_pass();
    unused_ports();
    ports_to_index();
    finish_links();
    out.parameters = parameters;
    out.parameter_names = parameter_names;
  }

  void edges_first_pass() {
    for (const EdgeSpec& e : setup.edges) {
      const std::string name = e.name();
      WNode& src = require_node(e.source, "edge source");
      WNode& tgt = require_node(e.target, "edge target");
      parameter_positions[name] = static_cast<int>(parameters.size());
      link({property(e.properties, "space", "length"), property(e.properties, "space", "refractive_index")});
      parameter_names.push_back(name + "_length");
      parameter_names.push_back(name + "_refractive_index");
      const int n_position = last(1);
      if (src.component == "mirror") surface_index_positions[e.source][e.source_port] = n_position;
      if (tgt.component == "mirror") surface_index_positions[e.target][e.target_port] = n_position;
      auto map_port = [](const std::string& p) {
        if (p == "right" || p == "bottom") return std::string("right");
        if (p == "left" || p == "top") return std::string("left");
        throw ValidationError("Unknown port '" + p + "'.");
      };
      auto check_beamsplitter = [&](const std::string& bs, const std::string& port_name) {
        const std::string port = map_port(port_name);
        auto& m = surface_index_positions[bs];
        const auto it = m.find(port);
        if (it != m.end() && parameters[static_cast<std::size_t>(it->second)] != parameters.back()) {
          throw ValidationError(
              "Beamsplitter has to have the same refractive index on left and top and right and bottom "
              "respectively.");
        }
        m[port] = n_position;
      };
      if (src.component == "beamsplitter") check_beamsplitter(e.source, e.source_port);
      if (tgt.component == "beamsplitter") check_beamsplitter(e.target, e.target_port);

      if (is_one_of(src.component, {"laser", "squeezer"})) {
        matrix_positions[name + "_source"] = system_size;
        system_size += 2;
        all_ports.insert(name + ".left");
        src.target = name + "_source";
      } else if (is_one_of(src.component, {"detector", "qnoised"})) {
        throw ValidationError("Detectors can only be targets of edges.");
      }
      if (is_one_of(tgt.component, {"detector", "qnoised"})) {
        matrix_positions[name + "_target"] = system_size;
        system_size += 2;
        all_ports.insert(name + ".right");
        tgt.target = name + "_target";
        tgt.direction = "out";
      } else if (is_one_of(tgt.component, {"laser", "squeezer"})) {
        throw ValidationError("Lasers can only be sources of edges.");
      }
    }
  }

  void nodes_first_pass() {
    for (auto& [name, n] : nodes) {
      if (kDefaultProperties.count(n.component) == 0) {
        throw ValidationError("Component '" + n.component + "' is not recognized.");
      }
      if (is_matrix_component(n.component)) {
        const int size = kMatrixSizes.at(n.component);
        matrix_positions[name] = system_size;
        std::vector<std::int64_t>* indices = nullptr;
        std::vector<std::int64_t> discard;
        if (n.component == "mirror") indices = &mirror_indices;
        else if (n.component == "beamsplitter") indices = &beamsplitter_indices;
        else if (n.component == "directional_beamsplitter") indices = &isolator_indices;
        else indices = &discard;
        for (int k = 0; k < size; ++k) indices->push_back(system_size + k);
        system_size += size;
        if (n.component == "mirror") {
          all_ports.insert(name + ".left");
          all_ports.insert(name + ".right");
        }
        if (n.component == "beamsplitter") {
          all_ports.insert(name + ".top");
          all_ports.insert(name + ".bottom");
          all_ports.insert(name + ".left");
          all_ports.insert(name + ".right");
        }
      }
      if (is_one_of(n.component, {"laser", "squeezer", "detector", "qnoised"})) {
        if (!n.port) n.port = "left";
        if (!n.direction) n.direction = "in";
      }
      if (n.component == "squeezer") signal_components.push_back(name);
      if (is_one_of(n.component, {"qnoised", "qhd"})) {
        if (!n.spec->auxiliary.value_or(false)) quantum_detector_number += 1;
      }
      if (n.component == "signal") {
        signal_components.push_back(name);
        matrix_positions[name] = signal_size;
        if (!n.target) throw ValidationError("Signal '" + name + "' needs a target.");
        if (const WNode* t = node(*n.target)) {
          if (t->component == "laser") laser_to_signals.append(*n.target, name);
          // upstream: a signal targeting any other node type is ignored.
        } else {
          space_to_signals.append(*n.target, name);
        }
        signal_size += 1;
      }
      if (n.component == "free_mass") {
        signal_components.push_back(name);
        matrix_positions[name] = signal_size;
        signal_size += 2;
        free_masses.push_back(name);
      }
      if (n.component == "frequency") {
        signal_components.push_back(name);
        signal_frequency_position = static_cast<int>(parameters.size());
        const PropertyValue f = node_property(n, "frequency");
        if (!std::holds_alternative<double>(f)) {
          throw ValidationError("The frequency property cannot be linked.");
        }
        parameters.push_back(std::get<double>(f));
        parameter_names.push_back(name + "_frequency");
      }
    }
  }

  std::vector<std::int64_t> mirror_indices, beamsplitter_indices, isolator_indices, detector_indices;
  int noise_detector_count = 0;

  void block_identity(int start, int size) {
    // upstream assigns a whole block matrix; its diagonal is 1.
    for (int k = 0; k < size; ++k) carrier_ops.push_back({start + k, start + k, Complex(1.0, 0.0)});
  }

  void nodes_second_pass() {
    const int s = system_size;
    const double sqrt2 = std::sqrt(2.0);
    for (auto& [name, n] : nodes) {
      if (n.component == "signal") {
        const int signal_index = position(name);
        link({node_property(n, "amplitude"), node_property(n, "phase")});
        parameter_names.push_back(name + "_amplitude");
        parameter_names.push_back(name + "_phase");
        set_instructions(signal_instructions, name, {row({last(2), last(1)})}, {0}, {2 * s + signal_index},
                         {kRhsColumn}, {CF::SignalFunction});
      }
      if (is_one_of(n.component, {"laser", "detector", "qnoised", "squeezer"})) {
        if (!n.target) throw ValidationError("'" + name + "' needs a target.");
        const std::string& target = *n.target;
        int target_index = 0;
        if (matrix_positions.count(target)) {
          target_index = matrix_positions.at(target);
        } else {
          const std::string suffix = (*n.port == "left") ? "_source" : "_target";
          target_index = position(target + suffix);
        }
        int offset = 0;
        bool target_is_matrix = false;
        if (const WNode* t = node(target)) {
          offset = port_offset(t->component, *n.port).value_or(0);
          target_is_matrix = is_matrix_component(t->component);
        }
        if (*n.direction != "in" && *n.direction != "out") {
          throw ValidationError("Direction has to be either in or out.");
        }
        const int direction_offset = !target_is_matrix ? (*n.direction == "in" ? 1 : 0) : (*n.direction == "in" ? 0 : 1);
        int component_index = target_index + offset + direction_offset;
        matrix_positions[name] = component_index;
        if (n.component == "detector") {
          if (n.spec->sideband.value_or("upper") == "lower") component_index += s;
          detector_indices.push_back(component_index);
        }
        if (n.component == "qnoised" && !n.spec->auxiliary.value_or(false)) {
          out.noise_selection(noise_detector_count, component_index) = sqrt2;
          out.noise_selection(noise_detector_count, component_index + s) = sqrt2;
          noise_detector_count += 1;
        }
        if (n.component == "laser") {
          used_ports.insert(strip_endpoint_suffix(target) + "." + *n.port);
          link({node_property(n, "power"), node_property(n, "phase")});
          parameter_names.push_back(name + "_power");
          parameter_names.push_back(name + "_phase");
          const Row inputs = row({last(2), last(1)});
          carrier_ops.push_back({component_index, kRhsColumn, {}, true, CF::Laser, inputs, 0});
          set_instructions(carrier_instructions, name, {inputs}, {0}, {component_index}, {kRhsColumn}, {CF::Laser});
          set_instructions(noise_instructions, name, {row({0})}, {0}, {component_index}, {component_index},
                           {CF::VacuumQuantumNoise}, std::nullopt, std::nullopt, std::nullopt, std::nullopt, s);
        }
        if (n.component == "squeezer") {
          used_ports.insert(strip_endpoint_suffix(target) + "." + *n.port);
          link({node_property(n, "db"), node_property(n, "angle")});
          parameter_names.push_back(name + "_db");
          parameter_names.push_back(name + "_angle");
          const int c = component_index;
          set_instructions(noise_instructions, name, {row({last(2), last(1)})}, {0, 1, 2, 3}, {c, c, c + s, c + s},
                           {c, c + s, c + s, c}, {CF::Squeezer});
        }
      }
      if (n.component == "qhd") {
        link({node_property(n, "phase")});
        parameter_names.push_back(name + "_phase");
        if (!n.spec->detector1 || !n.spec->detector2) throw ValidationError("qhd '" + name + "' needs detector1 and detector2.");
        const int d1 = position(*n.spec->detector1);
        const int d2 = position(*n.spec->detector2);
        if (noise_detector_count >= out.noise_selection.rows()) {
          throw ValidationError("qhd '" + name + "' has no noise-detector slot (auxiliary qhd is not supported upstream).");
        }
        out.qhd_parameter_indices.push_back(last(1));
        out.qhd_parameter_indices.push_back(last(1));
        out.qhd_placing_indices.push_back({noise_detector_count, 0, d2});
        out.qhd_placing_indices.push_back({noise_detector_count, 0, d2 + s});
        out.noise_selection(noise_detector_count, d1) = sqrt2;
        out.noise_selection(noise_detector_count, d2) = sqrt2;
        out.noise_selection(noise_detector_count, d1 + s) = sqrt2;
        out.noise_selection(noise_detector_count, d2 + s) = sqrt2;
        noise_detector_count += 1;
      }
      if (n.component == "nothing" || n.component == "directional_beamsplitter") {
        const int start = position(name);
        const int size = kMatrixSizes.at(n.component);
        block_identity(start, size);
        // nothing_matrix / directional_beamsplitter_matrix off-diagonal -1 couplings.
        const std::vector<std::pair<int, int>> couplings =
            n.component == "nothing" ? std::vector<std::pair<int, int>>{{1, 2}, {3, 0}}
                                     : std::vector<std::pair<int, int>>{{1, 2}, {3, 6}, {5, 0}, {7, 4}};
        for (auto [r, c] : couplings) carrier_ops.push_back({start + r, start + c, Complex(-1.0, 0.0)});
      }
      if (n.component == "mirror" || n.component == "beamsplitter") {
        surface(name, n);
      }
    }
  }

  void surface(const std::string& name, const WNode& n) {
    const int s = system_size;
    const int ci = position(name);
    parameter_positions[name] = static_cast<int>(parameters.size());
    auto& positions = surface_index_positions[name];
    const int nl = positions.count("left") ? positions.at("left") : 1;
    const int nr = positions.count("right") ? positions.at("right") : 1;
    const int sfp = signal_frequency_position;
    Row inputs, signal_inputs;
    std::vector<int> out_cols, rows, cols, noise_out, noise_idx;
    int loss_position = 0;
    int size = 0;
    if (n.component == "mirror") {
      link({node_property(n, "loss"), node_property(n, "reflectivity"), node_property(n, "tuning")});
      for (const char* p : {"_loss", "_reflectivity", "_tuning"}) parameter_names.push_back(name + p);
      loss_position = last(3);
      inputs = row({last(3), last(2), last(1), 0, nl, nr, 2});
      signal_inputs = row({last(3), last(2), last(1), sfp, nl, nr, 2});
      out_cols = {0, 1, 1, 2};
      for (int d : {1, 1, 3, 3}) rows.push_back(ci + d);
      for (int d : {0, 2, 0, 2}) cols.push_back(ci + d);
      noise_out = {0, 0};
      for (int d : {1, 3}) noise_idx.push_back(ci + d);
      size = 4;
    } else {
      link({node_property(n, "loss"), node_property(n, "reflectivity"), node_property(n, "tuning"),
            node_property(n, "alpha")});
      for (const char* p : {"_loss", "_reflectivity", "_tuning", "_alpha"}) parameter_names.push_back(name + p);
      loss_position = last(4);
      inputs = row({last(4), last(3), last(2), 0, nl, nr, last(1)});
      signal_inputs = row({last(4), last(3), last(2), sfp, nl, nr, last(1)});
      out_cols = {0, 1, 0, 1, 1, 2, 1, 2};
      for (int d : {1, 1, 3, 3, 5, 5, 7, 7}) rows.push_back(ci + d);
      for (int d : {2, 4, 0, 6, 0, 6, 2, 4}) cols.push_back(ci + d);
      noise_out = {0, 0, 0, 0};
      for (int d : {1, 3, 5, 7}) noise_idx.push_back(ci + d);
      size = 8;
    }
    // Template: mirror_matrix / beamsplitter_matrix = identity block + surface outputs.
    block_identity(ci, size);
    for (std::size_t p = 0; p < rows.size(); ++p) {
      carrier_ops.push_back({rows[p], cols[p], {}, true, CF::Surface, inputs, out_cols[p]});
    }
    set_instructions(carrier_instructions, name, {inputs}, out_cols, rows, cols, {CF::Surface});
    set_instructions(signal_instructions, name, {signal_inputs}, out_cols, rows, cols, {CF::Surface}, std::nullopt,
                     std::nullopt, std::nullopt, std::nullopt, s);
    set_instructions(noise_instructions, name, {row({loss_position})}, noise_out, noise_idx, noise_idx,
                     {CF::LossQuantumNoise}, std::nullopt, std::nullopt, std::nullopt, std::nullopt, s);
  }

  void laser_modulation() {
    const int s = system_size;
    for (const auto& [laser, signals] : laser_to_signals.items) {
      const int li = position(laser);
      std::vector<int> cols, functions_int;
      std::vector<CF> functions;
      for (const std::string& sig : signals) {
        const int si = position(sig);
        cols.push_back(2 * s + si);
        cols.push_back(2 * s + si);
        const std::string tp = require_node(sig, "signal").spec->target_property.value_or("");
        if (tp == "amplitude") {
          functions.push_back(CF::LaserAmplitudeModulation);
          functions.push_back(CF::LaserAmplitudeModulation);
        } else if (tp == "frequency") {
          functions.push_back(CF::LaserFrequencyModulation);
          functions.push_back(CF::LaserFrequencyModulationLower);
        } else {
          throw ValidationError("Target property for laser modulation has to be either amplitude or frequency.");
        }
      }
      const int count = static_cast<int>(signals.size());
      std::vector<int> rows, out_cols, out_rows, carrier_indices;
      std::vector<Row> inputs;
      for (int k = 0; k < count; ++k) {
        rows.push_back(li);
        rows.push_back(li + s);
        out_cols.push_back(0);
        out_cols.push_back(0);
        carrier_indices.push_back(li);
        carrier_indices.push_back(li);
        inputs.push_back(row({signal_frequency_position}));
        inputs.push_back(row({signal_frequency_position}));
      }
      for (int k = 0; k < 2 * count; ++k) out_rows.push_back(k);
      set_instructions(signal_instructions, laser, inputs, out_cols, rows, cols, functions, out_rows, carrier_indices,
                       rows, cols);
    }
  }

  void free_mass_instructions() {
    const int s = system_size;
    const int sfp = signal_frequency_position;
    for (const std::string& name : free_masses) {
      WNode& n = require_node(name, "free mass");
      if (!n.target || !matrix_positions.count(*n.target) || !parameter_positions.count(*n.target) ||
          !node(*n.target)) {
        throw ValidationError("Free mass has to be connected to either a mirror or a beamsplitter.");
      }
      const std::string& target = *n.target;
      const int ci = matrix_positions.at(target);
      const int pp = parameter_positions.at(target);
      const std::string tc = node(target)->component;
      auto& positions = surface_index_positions[target];
      const int nl = positions.count("left") ? positions.at("left") : 1;
      const int nr = positions.count("right") ? positions.at("right") : 1;
      const int fmi = position(name) + s;
      if (tc != "mirror" && tc != "beamsplitter") continue;  // upstream: silently ignored
      link({node_property(n, "mass")});
      parameter_names.push_back(name + "_mass");
      const std::vector<CF> functions = {CF::Susceptibility, CF::ForceCalculationLeft, CF::ForceCalculationRight,
                                         CF::CorrectedOptomechanicalPhaseLeft,
                                         CF::CorrectedOptomechanicalPhaseRight};
      std::vector<int> rows, cols, out_rows, carrier_indices;
      std::vector<Row> inputs;
      if (tc == "mirror") {
        rows = {fmi + 1, fmi, fmi, fmi, fmi, ci + 1, ci + 3};
        cols = {fmi, ci, ci + 1, ci + 2, ci + 3, fmi + 1, fmi + 1};
        inputs = {row({sfp, last(1)}), row({2}), row({2, nl, nr}), row({sfp, pp + 2, pp + 1, pp, nl, 2, 0}),
                  row({sfp, pp + 2, pp + 1, pp, nl, nr, 2})};
        out_rows = {0, 1, 1, 2, 2, 3, 4};
        carrier_indices = {ci, ci + 1, ci + 2, ci + 3, ci, ci + 2};
      } else {
        rows = {fmi + 1, fmi, fmi, fmi, fmi, fmi, fmi, fmi, fmi, ci + 1, ci + 3, ci + 5, ci + 7};
        cols = {fmi, ci, ci + 1, ci + 2, ci + 3, ci + 4, ci + 5, ci + 6, ci + 7, fmi + 1, fmi + 1, fmi + 1, fmi + 1};
        inputs = {row({sfp, last(1)}), row({pp + 3}), row({pp + 3, nl, nr}),
                  row({sfp, pp + 2, pp + 1, pp, nl, pp + 3, 0}), row({sfp, pp + 2, pp + 1, pp, nl, nr, pp + 3})};
        out_rows = {0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4};
        carrier_indices = {ci, ci + 1, ci + 2, ci + 3, ci + 4, ci + 5, ci + 6, ci + 7, ci + 2, ci, ci + 6, ci + 4};
      }
      std::vector<int> out_cols(rows.size(), 0);
      std::vector<int> crow(rows.begin() + 1, rows.end()), ccol(cols.begin() + 1, cols.end());
      set_instructions(signal_instructions, name, inputs, out_cols, rows, cols, functions, out_rows, carrier_indices,
                       crow, ccol, s);
    }
  }

  void edges_second_pass() {
    const int s = system_size;
    const int sfp = signal_frequency_position;
    for (const EdgeSpec& e : setup.edges) {
      const std::string name = e.name();
      const WNode& src = *node(e.source);
      const WNode& tgt = *node(e.target);
      if (is_matrix_component(src.component)) used_ports.insert(e.source + "." + e.source_port);
      if (is_matrix_component(tgt.component)) used_ports.insert(e.target + "." + e.target_port);
      const int si = position(e.source);
      const int ti = position(e.target);
      int s_in, s_out, t_in, t_out;
      if (const auto off = port_offset(src.component, e.source_port)) {
        s_in = si + *off;
        s_out = si + *off + 1;
      } else {
        s_in = si - 1;
        s_out = si;
      }
      if (const auto off = port_offset(tgt.component, e.target_port)) {
        t_in = ti + *off;
        t_out = ti + *off + 1;
      } else {
        t_in = ti + 1;
        t_out = ti;
      }
      const int pp = parameter_positions.at(name);
      const Row carrier_inputs = row({0, pp, pp + 1});
      carrier_ops.push_back({s_in, t_out, {}, true, CF::Space, carrier_inputs, 0});
      carrier_ops.push_back({t_in, s_out, {}, true, CF::Space, carrier_inputs, 0});
      set_instructions(carrier_instructions, name, {carrier_inputs}, {0, 0}, {s_in, t_in}, {t_out, s_out}, {CF::Space});
      const Row signal_row = row({sfp, pp, pp + 1});
      set_instructions(signal_instructions, name, {signal_row, signal_row}, {0, 0, 0, 0},
                       {s_in, t_in, s_in + s, t_in + s}, {t_out, s_out, t_out + s, s_out + s},
                       {CF::Space, CF::SpaceLower}, std::vector<int>{0, 0, 1, 1});
      if (const auto* signals = space_to_signals.find(name)) {
        const int count = static_cast<int>(signals->size());
        std::vector<int> rows, cols, out_cols, out_rows, carrier_indices;
        std::vector<Row> inputs;
        std::vector<CF> functions;
        for (int k = 0; k < count; ++k) {
          const int sig = position((*signals)[static_cast<std::size_t>(k)]);
          for (int q = 0; q < 4; ++q) cols.push_back(2 * s + sig);
          for (int r : {s_in, t_in, s_in + s, t_in + s}) rows.push_back(r);
          for (int c : {s_in, t_in, s_in, t_in}) carrier_indices.push_back(c);
          for (int q = 0; q < 4; ++q) out_cols.push_back(0);
          inputs.push_back(signal_row);
          inputs.push_back(signal_row);
          functions.push_back(CF::SpaceModulation);
          functions.push_back(CF::SpaceModulationLower);
        }
        for (int k = 0; k < 2 * count; ++k) {
          out_rows.push_back(k);
          out_rows.push_back(k);
        }
        set_instructions(signal_instructions, name + "_modulation", inputs, out_cols, rows, cols, functions, out_rows,
                         carrier_indices, rows, cols);
      }
    }
  }

  void unused_ports() {
    const int s = system_size;
    std::vector<std::string> unused;
    std::set_difference(all_ports.begin(), all_ports.end(), used_ports.begin(), used_ports.end(),
                        std::back_inserter(unused));
    for (const std::string& np : unused) {
      const std::size_t dot = np.find('.');
      const std::string name = np.substr(0, dot);
      const std::string port = np.substr(dot + 1);
      int offset = 0;
      if (const WNode* n = node(name)) offset = port_offset(n->component, port).value_or(0);
      const int pi = position(name) + offset;
      set_instructions(noise_instructions, np, {row({0})}, {0}, {pi}, {pi}, {CF::VacuumQuantumNoise}, std::nullopt,
                       std::nullopt, std::nullopt, std::nullopt, s);
    }
  }

  void ports_to_index() {
    for (const auto& [name, n] : nodes) {
      const auto it = kPortDicts.find(n.component);
      if (it == kPortDicts.end()) continue;
      const int idx = position(name);
      for (const auto& [port, offset] : it->second) {
        meta.port_to_index.emplace_back(name + "." + port + ".in", idx + offset);
        meta.port_to_index.emplace_back(name + "." + port + ".out", idx + offset + 1);
      }
    }
  }

  void finish_links() {
    out.linked_indices.clear();
    for (const std::string& ln : linked_names) {
      const auto it = std::find(parameter_names.begin(), parameter_names.end(), ln);
      if (it == parameter_names.end()) {
        throw ValidationError("Linked parameter source '" + ln + "' is not a parameter of the setup.");
      }
      out.linked_indices.push_back(static_cast<int>(it - parameter_names.begin()));
    }
    out.indices_to_link = indices_to_link;
    out.link_functions = link_functions;
    if (out.linked_indices.empty()) {  // upstream identity placeholder
      out.linked_indices = {0};
      out.indices_to_link = {0};
      out.link_functions = {LinkFunction::Identity};
    }
  }
};

// ---------------------------------------------------------- pairs_to_arrays
int name_index(const std::vector<std::string>& names, const std::string& name) {
  const auto it = std::find(names.begin(), names.end(), name);
  return it == names.end() ? -1 : static_cast<int>(it - names.begin());
}

void append_to_table(InstructionTable& t, const Instr& in, int output_row_base, bool explicit_rows) {
  for (std::size_t r = 0; r < in.inputs.size(); ++r) {
    t.inputs.push_back(in.inputs[r]);
    t.functions.push_back(in.functions[r]);
  }
  for (std::size_t p = 0; p < in.out_cols.size(); ++p) {
    t.output_cols.push_back(in.out_cols[p]);
    t.output_rows.push_back(output_row_base + (explicit_rows ? (*in.out_rows)[p] : 0));
    t.rows.push_back(in.rows[p]);
    t.cols.push_back(in.cols[p]);
  }
}

void make_dummy(InstructionTable& t) {
  // upstream combine_arrays placeholder: dummy_function writing 1 at (0, 0).
  t = InstructionTable{};
  t.functions = {CF::DummyFunction};
  t.inputs = {Row{}};
  t.output_rows = {0};
  t.output_cols = {0};
  t.rows = {0};
  t.cols = {0};
  t.is_dummy = true;
}

}  // namespace

SimulationPlanPtr PlanBuilder::build(SetupSpecPtr setup_ptr, PlanRequest request) {
  std::shared_ptr<SimulationPlan> plan(new SimulationPlan(setup_ptr, std::move(request)));
  SimulationPlan::Internals& in = *plan->internals_;
  const SetupSpec& setup = *plan->setup_;
  const PlanRequest& req = plan->request_;

  Builder b(setup, req, in, plan->metadata_);
  b.run();

  // ---- pairs_to_arrays ----------------------------------------------------
  // Signal: every instruction, output rows offset by a running counter.
  {
    int counter = 0;
    for (const auto& [key, instr] : b.signal_instructions.items) {
      const bool explicit_rows = instr.out_rows.has_value();
      append_to_table(in.signal, instr, counter, explicit_rows);
      if (explicit_rows) {
        std::set<int> unique(instr.out_rows->begin(), instr.out_rows->end());
        counter += static_cast<int>(unique.size());
      } else {
        counter += 1;
      }
      if (instr.carrier_indices) {
        for (std::size_t c = 0; c < instr.carrier_indices->size(); ++c) {
          in.signal.carrier_indices.push_back((*instr.carrier_indices)[c]);
          in.signal.coupling_rows.push_back((*instr.carrier_rows)[c]);
          in.signal.coupling_cols.push_back((*instr.carrier_cols)[c]);
        }
      }
    }
    if (in.signal.functions.empty()) make_dummy(in.signal);
    if (in.signal.carrier_indices.empty()) {  // upstream default placing [[0], [-1]]
      in.signal.carrier_indices = {0};
      in.signal.coupling_rows = {0};
      in.signal.coupling_cols = {kRhsColumn};
    }
  }
  // Noise: every instruction, one component counter per instruction.
  {
    int counter = 0;
    for (const auto& [key, instr] : b.noise_instructions.items) {
      append_to_table(in.noise, instr, counter, false);
      counter += 1;
    }
    if (in.noise.functions.empty()) make_dummy(in.noise);
  }
  // Carrier: only optimised (non-signal) and carrier-changing components.
  std::vector<std::string> carrier_components;
  auto append_carrier = [&](const std::string& component) {
    if (std::find(carrier_components.begin(), carrier_components.end(), component) != carrier_components.end()) return;
    const Instr* instr = b.carrier_instructions.find(component);
    if (!instr) return;
    carrier_components.push_back(component);
    append_to_table(in.carrier, *instr, static_cast<int>(carrier_components.size()) - 1, false);
  };
  auto is_signal_component = [&](const std::string& c) {
    return std::find(b.signal_components.begin(), b.signal_components.end(), c) != b.signal_components.end();
  };
  const auto& targets = req.optimization_targets;
  for (std::size_t ix = 0; ix < targets.size(); ++ix) {
    std::string last_component;
    for (const ParameterPair& pair : targets[ix]) {
      last_component = pair.first;  // upstream: the loop variable leaks; only the last member counts below
      const int idx = name_index(b.parameter_names, pair.first + "_" + pair.second);
      if (idx >= 0) {
        in.optimized_parameter_indices.push_back(idx);
        in.optimized_value_indices.push_back(static_cast<int>(ix));
      }
    }
    if (!is_signal_component(last_component)) append_carrier(last_component);
  }
  for (const ParameterPair& pair : req.changing_pairs) {
    const int idx = name_index(b.parameter_names, pair.first + "_" + pair.second);
    if (idx < 0) {
      throw ValidationError("'" + pair.first + "_" + pair.second + "' is not in the parameter list.");
    }
    if (is_signal_component(pair.first)) {
      in.signal_changing_indices.push_back(idx);
    } else {
      in.carrier_changing_indices.push_back(idx);
      append_carrier(pair.first);
    }
  }
  if (in.carrier.functions.empty()) make_dummy(in.carrier);

  // ---- prepare_arrays ------------------------------------------------------
  if (in.optimized_parameter_indices.empty()) {
    in.default_optimization = true;
    in.optimized_parameter_indices = {0};
    in.optimized_value_indices = {0};
  }
  if (!in.carrier_changing_indices.empty() && !in.signal_changing_indices.empty()) {
    throw ValidationError("This combination of changing component-parameter pairs is not supported yet.");
  }
  if (in.carrier_changing_indices.empty()) {
    in.carrier_changing_indices = {0};
    in.carrier_changing_values = MatrixR::Constant(1, 1, in.parameters[0]);
  } else {
    in.carrier_changing_values = req.changing_values;
    if (in.carrier_changing_values.rows() != static_cast<Index>(in.carrier_changing_indices.size())) {
      throw ValidationError("The changing_values array needs to have shape (number of changing parameters, number of values).");
    }
  }
  if (in.signal_changing_indices.empty()) {
    in.signal_changing_indices = {0};
    in.signal_changing_values = MatrixR::Constant(1, 1, in.parameters[0]);
  } else {
    in.signal_changing_values = req.changing_values;
  }

  // ---- layouts and templates ----------------------------------------------
  // Resolved base parameters (links applied) evaluate the constant template.
  std::vector<double> base = in.parameters;
  {
    std::vector<double> linked(in.linked_indices.size());
    for (std::size_t i = 0; i < linked.size(); ++i) {
      linked[i] = apply_link(in.link_functions[i], base[static_cast<std::size_t>(in.linked_indices[i])]);
    }
    for (std::size_t i = 0; i < linked.size(); ++i) base[static_cast<std::size_t>(in.indices_to_link[i])] = linked[i];
  }
  const int s = in.system_size;
  const int n_signal = 2 * s + in.signal_size;

  // Carrier template: identity, then every build-time write in order.
  std::unordered_map<std::uint64_t, Complex> carrier_template;
  std::vector<Complex> carrier_rhs(static_cast<std::size_t>(s), Complex(0.0, 0.0));
  for (int i = 0; i < s; ++i) carrier_template[key_of(i, i, s)] = Complex(1.0, 0.0);
  for (const TemplateOp& op : b.carrier_ops) {
    Complex v = op.constant;
    if (op.from_kernel) {
      double p[kComponentInputWidth];
      for (int k = 0; k < kComponentInputWidth; ++k) p[k] = base[static_cast<std::size_t>(op.inputs[k])];
      Complex y[kComponentOutputs];
      evaluate_component(op.function, p, y);
      v = y[op.output];
    }
    if (op.col == kRhsColumn) {
      carrier_rhs[static_cast<std::size_t>(op.row)] = v;
    } else {
      carrier_template[key_of(op.row, op.col, s)] = v;
    }
  }

  auto build_layout = [](MatrixLayout& layout, int n, const std::unordered_map<std::uint64_t, Complex>& tmpl,
                         std::vector<Complex> rhs, const InstructionTable& table) {
    std::vector<int> rows, cols;
    std::vector<Complex> vals;
    for (const auto& [k, v] : tmpl) {
      if (v == Complex(0.0, 0.0)) continue;
      const int r = static_cast<int>(k / static_cast<std::uint64_t>(n + 1));
      const int c = static_cast<int>(k % static_cast<std::uint64_t>(n + 1));
      if (c == n) continue;
      rows.push_back(r);
      cols.push_back(c);
      vals.push_back(v);
    }
    const std::size_t n_template = rows.size();
    for (std::size_t p = 0; p < table.rows.size(); ++p) {
      if (table.cols[p] == kRhsColumn) continue;
      rows.push_back(table.rows[p]);
      cols.push_back(table.cols[p]);
    }
    for (std::size_t c = 0; c < table.coupling_rows.size(); ++c) {
      if (table.coupling_cols[c] == kRhsColumn) continue;
      rows.push_back(table.coupling_rows[c]);
      cols.push_back(table.coupling_cols[c]);
    }
    for (int r : rows) if (r < 0 || r >= n) throw ValidationError("internal: matrix row out of range");
    for (int c : cols) if (c < 0 || c >= n) throw ValidationError("internal: matrix column out of range");
    auto [pattern, slots] = SparsePattern::from_coordinates(n, rows, cols);
    layout.n = n;
    layout.template_values.assign(static_cast<std::size_t>(pattern.nnz()), Complex(0.0, 0.0));
    for (std::size_t i = 0; i < n_template; ++i) layout.template_values[static_cast<std::size_t>(slots[i])] = vals[i];
    layout.template_rhs = std::move(rhs);
    std::size_t cursor = n_template;
    layout.placement_slots.clear();
    for (std::size_t p = 0; p < table.rows.size(); ++p) {
      layout.placement_slots.push_back(table.cols[p] == kRhsColumn ? -1 : slots[cursor++]);
    }
    layout.coupling_slots.clear();
    for (std::size_t c = 0; c < table.coupling_rows.size(); ++c) {
      layout.coupling_slots.push_back(table.coupling_cols[c] == kRhsColumn ? -1 : slots[cursor++]);
    }
    for (auto& v : layout.rows_by_function) v.clear();
    for (std::size_t r = 0; r < table.functions.size(); ++r) {
      layout.rows_by_function[static_cast<std::size_t>(table.functions[r])].push_back(static_cast<int>(r));
    }
    layout.pattern = std::make_shared<const SparsePattern>(std::move(pattern));
  };

  build_layout(in.carrier_layout, s, carrier_template, carrier_rhs, in.carrier);

  // Signal template: identity, both sideband blocks copy the carrier template.
  std::unordered_map<std::uint64_t, Complex> signal_template;
  for (int i = 0; i < n_signal; ++i) signal_template[key_of(i, i, n_signal)] = Complex(1.0, 0.0);
  for (const auto& [k, v] : carrier_template) {
    const int r = static_cast<int>(k / static_cast<std::uint64_t>(s + 1));
    const int c = static_cast<int>(k % static_cast<std::uint64_t>(s + 1));
    if (c == s) continue;
    signal_template[key_of(r, c, n_signal)] = v;
    signal_template[key_of(r + s, c + s, n_signal)] = v;
  }
  build_layout(in.signal_layout, n_signal, signal_template,
               std::vector<Complex>(static_cast<std::size_t>(n_signal), Complex(0.0, 0.0)), in.signal);
  {
    const SparsePattern& p = *in.signal_layout.pattern;
    const auto slot_cols = p.slot_columns();
    for (int slot = 0; slot < p.nnz(); ++slot) {
      const int r = p.row_idx()[static_cast<std::size_t>(slot)];
      const int c = slot_cols[static_cast<std::size_t>(slot)];
      const bool lower_block = r >= s && r < 2 * s && c >= s && c < 2 * s;
      const bool signal_to_upper = r >= 2 * s && c < s;
      const bool lower_to_signal = r >= s && r < 2 * s && c >= 2 * s;
      if (lower_block || signal_to_upper || lower_to_signal) in.signal_layout.conjugate_slots.push_back(slot);
    }
    // The right-hand side column (index N >= 2S) of rows S..2S is conjugated too.
    for (int r = s; r < 2 * s; ++r) in.signal_layout.conjugate_rhs_rows.push_back(r);
  }

  build_layout(in.noise_layout, n_signal, {}, std::vector<Complex>(static_cast<std::size_t>(n_signal), Complex(0.0, 0.0)),
               in.noise);

  // ---- public metadata and dimensions -------------------------------------
  auto to_vector = [](const std::vector<std::int64_t>& v) {
    VectorI out(static_cast<Index>(v.size()));
    for (std::size_t i = 0; i < v.size(); ++i) out(static_cast<Index>(i)) = v[i];
    return out;
  };
  plan->metadata_.detector_indices = to_vector(b.detector_indices);
  plan->metadata_.mirror_indices = to_vector(b.mirror_indices);
  plan->metadata_.beamsplitter_indices = to_vector(b.beamsplitter_indices);
  plan->metadata_.isolator_indices = to_vector(b.isolator_indices);
  in.port_to_index.clear();
  for (const auto& [k, v] : plan->metadata_.port_to_index) in.port_to_index.emplace_back(k, static_cast<int>(v));

  Dimensions& d = plan->dimensions_;
  d.parameters = static_cast<Index>(in.parameters.size());
  d.optimized_values = in.default_optimization ? 1 : static_cast<Index>(req.optimization_targets.size());
  d.carrier_ports = s;
  d.carrier_values = in.carrier_changing_values.cols();
  d.signal_variables = n_signal;
  d.signal_values = std::max(in.carrier_changing_values.cols(), in.signal_changing_values.cols());
  d.noise_detectors = in.noise_selection.rows();
  return plan;
}

SimulationPlanPtr build_plan(SetupSpecPtr setup, PlanRequest request) {
  if (!setup) throw ValidationError("build_plan: setup must not be null.");
  setup->validate();
  validate_request(*setup, request);
  return PlanBuilder::build(std::move(setup), std::move(request));
}

}  // namespace dfm
