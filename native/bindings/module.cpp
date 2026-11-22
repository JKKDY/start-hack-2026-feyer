// nanobind bindings: differometor_native._native
//
// The only Python-aware code. It converts plain Python/NumPy data into the
// C++ value types of dfm_core and back. Crossings are coarse (one call per
// plan, forward, or backward), inputs are viewed without copying where the
// dtype and layout already match, and outputs are moved into NumPy.
//
// Submodules:
//   _native.core        SetupSpec, SimulationPlan, Simulator, SimulationState
//   _native.detectors   amplitude/power/signal detectors
//   _native.linalg      ComplexSparseLU
//   _native.diagnostics CoDiPack/Eigen smoke check

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <complex>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "dfm/ad_eigen.hpp"
#include "dfm/assembly.hpp"
#include "dfm/component_ad.hpp"
#include "dfm/component_ops.hpp"
#include "dfm/components.hpp"
#include "dfm/constants.hpp"
#include "dfm/sparse.hpp"
#include "dfm/sparse_ad.hpp"
#include "dfm/detectors.hpp"
#include "dfm/errors.hpp"
#include "dfm/link.hpp"
#include "dfm/plan.hpp"
#include "dfm/plan_internals.hpp"
#include "dfm/setup_spec.hpp"
#include "dfm/simulator.hpp"
#include "dfm/sparse_lu.hpp"
#include "dfm/version.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

using dfm::Complex;
using dfm::Index;

// ------------------------------------------------------------ array types
using CArray1 = nb::ndarray<const Complex, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using CArray2 = nb::ndarray<const Complex, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using CArrayN = nb::ndarray<const Complex, nb::c_contig, nb::device::cpu>;
using RArray1 = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using RArray2 = nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using IArray1 = nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

template <typename T>
using NpArray = nb::ndarray<nb::numpy, T>;

// Moves an Eigen object to the heap and exposes its storage to NumPy. The
// capsule frees it when the array dies. No element copy happens.
template <typename Owner, typename T>
NpArray<T> expose(Owner&& value, std::initializer_list<size_t> shape) {
  auto* holder = new std::decay_t<Owner>(std::forward<Owner>(value));
  nb::capsule owner(holder, [](void* p) noexcept { delete static_cast<std::decay_t<Owner>*>(p); });
  return NpArray<T>(holder->data(), shape, owner);
}

NpArray<Complex> to_numpy(dfm::MatrixC&& m) {
  const size_t r = static_cast<size_t>(m.rows());
  const size_t c = static_cast<size_t>(m.cols());
  return expose<dfm::MatrixC, Complex>(std::move(m), {r, c});
}
NpArray<double> to_numpy(dfm::MatrixR&& m) {
  const size_t r = static_cast<size_t>(m.rows());
  const size_t c = static_cast<size_t>(m.cols());
  return expose<dfm::MatrixR, double>(std::move(m), {r, c});
}
NpArray<double> to_numpy(dfm::VectorR&& v) {
  const size_t n = static_cast<size_t>(v.size());
  return expose<dfm::VectorR, double>(std::move(v), {n});
}
NpArray<std::int64_t> to_numpy(const dfm::VectorI& v) {
  dfm::VectorI copy = v;  // plan metadata is shared; hand out an owned copy
  const size_t n = static_cast<size_t>(copy.size());
  return expose<dfm::VectorI, std::int64_t>(std::move(copy), {n});
}
NpArray<std::int64_t> to_numpy(dfm::MatrixI&& m) {
  const size_t r = static_cast<size_t>(m.rows());
  const size_t c = static_cast<size_t>(m.cols());
  return expose<dfm::MatrixI, std::int64_t>(std::move(m), {r, c});
}
NpArray<std::int64_t> to_numpy(dfm::VectorI&& v) {
  const size_t n = static_cast<size_t>(v.size());
  return expose<dfm::VectorI, std::int64_t>(std::move(v), {n});
}

NpArray<Complex> to_numpy(dfm::VectorC&& v) {
  const size_t n = static_cast<size_t>(v.size());
  return expose<dfm::VectorC, Complex>(std::move(v), {n});
}

Eigen::Map<const dfm::MatrixC> view(const CArray2& a) {
  return {a.data(), static_cast<Index>(a.shape(0)), static_cast<Index>(a.shape(1))};
}
Eigen::Map<const dfm::MatrixR> view(const RArray2& a) {
  return {a.data(), static_cast<Index>(a.shape(0)), static_cast<Index>(a.shape(1))};
}
Eigen::Map<const dfm::VectorR> view(const RArray1& a) {
  return {a.data(), static_cast<Index>(a.shape(0))};
}

// ------------------------------------------------------- setup conversion
// The Python layer normalises `Setup.to_data()` into these plain tuples.
// Property values are a float or a link (source_parameter, function_name).
using PyPropertyValue = std::variant<double, std::pair<std::string, std::string>>;
using PyProperties = std::vector<std::pair<std::string, PyPropertyValue>>;
using PyNode = std::tuple<std::string,                 // name
                          std::string,                 // component
                          PyProperties,                // properties
                          std::optional<std::string>,  // target
                          std::optional<std::string>,  // target_property
                          std::optional<std::string>,  // port
                          std::optional<std::string>,  // direction
                          std::optional<bool>,         // auxiliary
                          std::optional<std::string>,  // detector1
                          std::optional<std::string>,  // detector2
                          std::optional<std::string>>; // sideband
using PyEdge = std::tuple<std::string, std::string, PyProperties, std::string, std::string>;

dfm::Properties convert_properties(const PyProperties& in) {
  dfm::Properties out;
  out.reserve(in.size());
  for (const auto& [key, value] : in) {
    if (const double* number = std::get_if<double>(&value)) {
      out.emplace_back(key, *number);
    } else {
      const auto& [source, function_name] = std::get<std::pair<std::string, std::string>>(value);
      const auto f = dfm::link_function_from_name(function_name);
      if (!f) {
        throw dfm::ValidationError("Unsupported link function '" + function_name + "' for property '" +
                                   key + "'.");
      }
      out.emplace_back(key, dfm::LinkedValue{source, *f});
    }
  }
  return out;
}

std::shared_ptr<const dfm::SetupSpec> make_setup_spec(const std::vector<PyNode>& nodes,
                                                      const std::vector<PyEdge>& edges,
                                                      const std::vector<dfm::ParameterPair>& parameters) {
  auto spec = std::make_shared<dfm::SetupSpec>();
  spec->nodes.reserve(nodes.size());
  for (const PyNode& n : nodes) {
    dfm::NodeSpec node;
    node.name = std::get<0>(n);
    node.component = std::get<1>(n);
    node.properties = convert_properties(std::get<2>(n));
    node.target = std::get<3>(n);
    node.target_property = std::get<4>(n);
    node.port = std::get<5>(n);
    node.direction = std::get<6>(n);
    node.auxiliary = std::get<7>(n);
    node.detector1 = std::get<8>(n);
    node.detector2 = std::get<9>(n);
    node.sideband = std::get<10>(n);
    spec->nodes.push_back(std::move(node));
  }
  spec->edges.reserve(edges.size());
  for (const PyEdge& e : edges) {
    dfm::EdgeSpec edge;
    edge.source = std::get<0>(e);
    edge.target = std::get<1>(e);
    edge.properties = convert_properties(std::get<2>(e));
    edge.source_port = std::get<3>(e);
    edge.target_port = std::get<4>(e);
    spec->edges.push_back(std::move(edge));
  }
  spec->parameters = parameters;
  spec->validate();
  return spec;
}

dfm::SignalSolver parse_solver(const std::string& name) {
  if (name == "packed") return dfm::SignalSolver::Packed;
  if (name == "dense") return dfm::SignalSolver::Dense;
  throw dfm::ValidationError("signal_solver must be 'packed' or 'dense', got '" + name + "'.");
}

nb::tuple outputs_to_python(dfm::ForwardOutputs&& out) {
  return nb::make_tuple(to_numpy(std::move(out.carrier)), to_numpy(std::move(out.signal)),
                        to_numpy(std::move(out.noise)));
}

nb::dict dimensions_dict(const dfm::Dimensions& d) {
  nb::dict out;
  out["parameters"] = d.parameters;
  out["optimized_values"] = d.optimized_values;
  out["carrier_ports"] = d.carrier_ports;
  out["carrier_values"] = d.carrier_values;
  out["signal_variables"] = d.signal_variables;
  out["signal_values"] = d.signal_values;
  out["noise_detectors"] = d.noise_detectors;
  return out;
}

}  // namespace

NB_MODULE(_native, m) {
  m.doc() = "Native C++ Differometor engine (planner, simulator, detectors).";
  // Plans and simulators referenced by long-lived caches (e.g. JAX's jit cache
  // holding static NativePlan handles) are still alive at interpreter exit;
  // that is expected, not a reference-counting bug.
  nb::set_leak_warnings(false);

  nb::register_exception_translator([](const std::exception_ptr& p, void*) {
    try {
      std::rethrow_exception(p);
    } catch (const dfm::ValidationError& e) {
      PyErr_SetString(PyExc_ValueError, e.what());
    } catch (const dfm::NotImplementedError& e) {
      PyErr_SetString(PyExc_NotImplementedError, e.what());
    } catch (const dfm::StateError& e) {
      PyErr_SetString(PyExc_RuntimeError, e.what());
    }
  });

  m.def("version", &dfm::version, "Version string of the native engine.");
  m.def(
      "link_functions",
      [] {
        std::vector<std::string> names;
        for (int i = 0; i <= static_cast<int>(dfm::LinkFunction::Abs); ++i) {
          names.emplace_back(dfm::link_function_name(static_cast<dfm::LinkFunction>(i)));
        }
        return names;
      },
      "Names of the parameter-link functions the native engine supports.");

  // ------------------------------------------------------------------ core
  nb::module_ core = m.def_submodule("core", "Setup snapshot, runtime planner and simulator.");

  nb::class_<dfm::SetupSpec>(core, "SetupSpec",
                             "Immutable native snapshot of a Setup. Build it with "
                             "differometor_native.simulate.setup_spec(setup).")
      .def_static("from_normalized", &make_setup_spec, "nodes"_a, "edges"_a, "parameters"_a,
                  "Create from the normalised tuples produced by the Python layer.")
      .def_prop_ro("num_nodes", [](const dfm::SetupSpec& s) { return s.nodes.size(); })
      .def_prop_ro("num_edges", [](const dfm::SetupSpec& s) { return s.edges.size(); })
      .def_prop_ro("parameters", [](const dfm::SetupSpec& s) { return s.parameters; })
      .def_prop_ro("node_names",
                   [](const dfm::SetupSpec& s) {
                     std::vector<std::string> out;
                     for (const auto& n : s.nodes) out.push_back(n.name);
                     return out;
                   })
      .def_prop_ro("edge_names",
                   [](const dfm::SetupSpec& s) {
                     std::vector<std::string> out;
                     for (const auto& e : s.edges) out.push_back(e.name());
                     return out;
                   })
      .def("has_property", [](const dfm::SetupSpec& s, const std::string& name, const std::string& prop) {
        return s.has_property({name, prop});
      });

  nb::class_<dfm::SimulationPlan>(core, "SimulationPlan",
                                  "Immutable plan for one setup and sweep/optimisation request.")
      .def_prop_ro("dimensions", [](const dfm::SimulationPlan& p) { return dimensions_dict(p.dimensions()); })
      .def_prop_ro("metadata",
                   [](const dfm::SimulationPlan& p) {
                     const auto& md = p.metadata();
                     nb::dict port_to_index;
                     for (const auto& [k, v] : md.port_to_index) port_to_index[k.c_str()] = v;
                     return nb::make_tuple(to_numpy(md.detector_indices), to_numpy(md.mirror_indices),
                                           to_numpy(md.beamsplitter_indices), to_numpy(md.isolator_indices),
                                           port_to_index);
                   })
      .def("export_simulation_arrays",
           [](const dfm::SimulationPlan& p) {
             nb::dict out;
             for (dfm::NamedArray& a : p.export_simulation_arrays()) {
               nb::object value = std::visit(
                   [](auto&& v) -> nb::object {
                     using V = std::decay_t<decltype(v)>;
                     if constexpr (std::is_same_v<V, std::monostate>) {
                       return nb::none();
                     } else {
                       return nb::cast(to_numpy(std::move(v)));
                     }
                   },
                   a.value);
               if (!a.shape.empty()) {
                 nb::list shape;
                 for (dfm::Index d : a.shape) shape.append(d);
                 value = value.attr("reshape")(nb::tuple(shape));
               }
               out[a.name.c_str()] = value;
             }
             return out;
           })
      .def("assemble",
           [](const dfm::SimulationPlan& p, int kind, const RArray1& x, std::optional<CArray2> carrier) {
             const dfm::VectorR resolved = dfm::resolve_parameters(p, view(x));
             dfm::AssembledSystem sys;
             if (kind == 0) {
               sys = dfm::assemble_carrier(p, resolved);
             } else if (kind == 1) {
               if (!carrier) throw dfm::ValidationError("signal assembly needs carrier fields");
               sys = dfm::assemble_signal(p, resolved, view(*carrier));
             } else if (kind == 2) {
               sys = dfm::assemble_noise(p, resolved);
             } else {
               throw dfm::ValidationError("kind must be 0 (carrier), 1 (signal) or 2 (noise)");
             }
             dfm::VectorI indptr(sys.pattern->size() + 1), indices(sys.pattern->nnz());
             for (int k = 0; k <= sys.pattern->size(); ++k) indptr(k) = sys.pattern->col_ptr()[static_cast<std::size_t>(k)];
             for (int k = 0; k < sys.pattern->nnz(); ++k) indices(k) = sys.pattern->row_idx()[static_cast<std::size_t>(k)];
             return nb::make_tuple(to_numpy(std::move(indptr)), to_numpy(std::move(indices)),
                                   to_numpy(std::move(sys.values)), to_numpy(std::move(sys.rhs)));
           },
           "kind"_a, "optimized_parameters"_a, "carrier"_a.none() = nb::none(),
           "Assemble kind 0 carrier / 1 signal / 2 noise. Returns (indptr, indices, values (batch, nnz), "
           "rhs (batch, n)); CSC with a shared pattern.")
      .def("assembly_vjp",
           [](const dfm::SimulationPlan& p, int kind, const RArray1& x, std::optional<CArray2> carrier,
              const CArray2& values_cot, const CArray2& rhs_cot) {
             if (kind < 0 || kind > 2) throw dfm::ValidationError("kind must be 0, 1 or 2");
             dfm::MatrixC fields = carrier ? dfm::MatrixC(view(*carrier)) : dfm::MatrixC(0, 0);
             auto g = dfm::assembly_vjp(p, static_cast<dfm::SystemKind>(kind), view(x), fields, view(values_cot),
                                        view(rhs_cot));
             nb::object carrier_grad = nb::none();
             if (kind == 1) carrier_grad = nb::cast(to_numpy(std::move(g.carrier)));
             return nb::make_tuple(to_numpy(std::move(g.optimized)), carrier_grad);
           },
           "kind"_a, "optimized_parameters"_a, "carrier"_a.none(), "values_cotangent"_a, "rhs_cotangent"_a,
           "Reverse-mode gradient of sum Re(conj(cot) * entry) over values and rhs, through the CoDiPack tape. "
           "Returns (d/d optimized_parameters, d/d carrier or None).")
      .def_prop_ro("parameter_names", [](const dfm::SimulationPlan& p) {
        return p.internals().parameter_names;
      })
      .def_prop_ro("parameters", [](const dfm::SimulationPlan& p) -> nb::object {
        // Returned as a generic object: a fresh owned array (no reference_internal policy).
        const auto& v = p.internals().parameters;
        return nb::cast(to_numpy(dfm::VectorR(Eigen::Map<const dfm::VectorR>(v.data(), static_cast<dfm::Index>(v.size())))));
      });

  core.def(
      "build_plan",
      [](std::shared_ptr<const dfm::SetupSpec> setup, const std::vector<dfm::ParameterPair>& changing_pairs,
         std::optional<RArray2> changing_values, const std::vector<dfm::OptimizationTarget>& optimization_targets,
         const std::string& signal_solver, std::optional<bool> pack_noise) {
        dfm::PlanRequest request;
        request.changing_pairs = changing_pairs;
        if (changing_values) {
          request.changing_values = view(*changing_values);
        }
        request.optimization_targets = optimization_targets;
        request.signal_solver = parse_solver(signal_solver);
        request.pack_noise = pack_noise;
        return dfm::build_plan(std::move(setup), std::move(request));
      },
      "setup"_a, "changing_pairs"_a, "changing_values"_a.none(), "optimization_targets"_a,
      "signal_solver"_a = "packed", "pack_noise"_a.none() = nb::none(),
      "Validate the request and build an immutable SimulationPlan.");

  core.def(
      "validate_request",
      [](const dfm::SetupSpec& setup, const std::vector<dfm::ParameterPair>& changing_pairs,
         std::optional<RArray2> changing_values, const std::vector<dfm::OptimizationTarget>& optimization_targets) {
        dfm::PlanRequest request;
        request.changing_pairs = changing_pairs;
        if (changing_values) {
          request.changing_values = view(*changing_values);
        }
        request.optimization_targets = optimization_targets;
        dfm::validate_request(setup, request);
      },
      "setup"_a, "changing_pairs"_a, "changing_values"_a.none(), "optimization_targets"_a);

  nb::class_<dfm::SimulationState>(core, "SimulationState",
                                   "Opaque forward state needed by Simulator.backward. Keeps its plan alive.")
      .def_prop_ro("is_current", &dfm::SimulationState::is_current);

  nb::class_<dfm::Simulator>(core, "Simulator",
                             "Native forward/reverse engine for one plan. Not thread-safe: use one "
                             "simulator per thread.")
      .def(nb::init<std::shared_ptr<const dfm::SimulationPlan>>(), "plan"_a)
      .def_prop_ro("plan", [](const dfm::Simulator& s) { return s.plan_ptr(); })
      .def(
          "forward",
          [](dfm::Simulator& s, const RArray1& p) {
            dfm::ForwardOutputs out;
            {
              nb::gil_scoped_release release;
              out = s.forward(view(p));
            }
            return outputs_to_python(std::move(out));
          },
          "optimized_parameters"_a, "Returns (carrier, signal, noise) with noise 2-D (unsqueezed).")
      .def(
          "forward_with_state",
          [](dfm::Simulator& s, const RArray1& p) {
            std::pair<dfm::ForwardOutputs, std::unique_ptr<dfm::SimulationState>> result;
            {
              nb::gil_scoped_release release;
              result = s.forward_with_state(view(p));
            }
            return nb::make_tuple(outputs_to_python(std::move(result.first)), std::move(result.second));
          },
          "optimized_parameters"_a, "Returns ((carrier, signal, noise), state).")
      .def(
          "backward",
          [](dfm::Simulator& s, const dfm::SimulationState& state, std::optional<CArray2> carrier,
             std::optional<CArray2> signal, std::optional<RArray2> noise) {
            dfm::OutputCotangents ct;
            if (carrier) ct.carrier = view(*carrier);
            if (signal) ct.signal = view(*signal);
            if (noise) ct.noise = view(*noise);
            dfm::VectorR grad;
            {
              nb::gil_scoped_release release;
              grad = s.backward(state, ct);
            }
            return to_numpy(std::move(grad));
          },
          "state"_a, "carrier"_a.none() = nb::none(), "signal"_a.none() = nb::none(),
          "noise"_a.none() = nb::none(),
          "Vector-Jacobian product with respect to the optimized parameters. Missing "
          "cotangents count as zero.");

  // ------------------------------------------------------------- detectors
  nb::module_ det = m.def_submodule("detectors", "Detector readouts.");
  det.def(
      "power_detector",
      [](const CArrayN& x) {
        auto* out = new dfm::VectorR(static_cast<Index>(x.size()));
        dfm::power_detector(x.data(), out->data(), static_cast<Index>(x.size()));
        std::vector<size_t> shape(x.ndim());
        for (size_t i = 0; i < x.ndim(); ++i) shape[i] = x.shape(i);
        nb::capsule owner(out, [](void* p) noexcept { delete static_cast<dfm::VectorR*>(p); });
        return NpArray<double>(out->data(), shape.size(), shape.data(), owner);
      },
      "solution"_a);
  det.def(
      "amplitude_detector",
      [](const CArrayN& x) {
        auto* out = new dfm::VectorR(static_cast<Index>(x.size()));
        dfm::amplitude_detector(x.data(), out->data(), static_cast<Index>(x.size()));
        std::vector<size_t> shape(x.ndim());
        for (size_t i = 0; i < x.ndim(); ++i) shape[i] = x.shape(i);
        nb::capsule owner(out, [](void* p) noexcept { delete static_cast<dfm::VectorR*>(p); });
        return NpArray<double>(out->data(), shape.size(), shape.data(), owner);
      },
      "solution"_a);
  det.def(
      "signal_detector_2d",
      [](const CArray2& carrier, const CArray2& signal) {
        return to_numpy(dfm::signal_detector(view(carrier), view(signal)));
      },
      "carrier"_a, "signal"_a, "2-D kernel: carrier (N, Kc), signal (>=2N, Ks) -> (N, K).");

  // ----------------------------------------------------------------- linalg
  nb::module_ linalg = m.def_submodule("linalg", "Sparse linear algebra.");
  nb::class_<dfm::ComplexSparseLU>(linalg, "ComplexSparseLU",
                                   "Sparse LU of a complex square CSC matrix. Factorise once, then "
                                   "solve A x = b or A^H x = b.")
      .def(
          "__init__",
          [](dfm::ComplexSparseLU* self, const IArray1& indptr, const IArray1& indices, const CArray1& data) {
            if (indptr.shape(0) < 1) throw dfm::ValidationError("indptr must have at least one entry");
            const int n = static_cast<int>(indptr.shape(0)) - 1;
            const int nnz = static_cast<int>(indices.shape(0));
            if (static_cast<int>(data.shape(0)) != nnz)
              throw dfm::ValidationError("indices and data must have the same length");
            if (indptr.data()[0] != 0 || indptr.data()[n] != nnz)
              throw dfm::ValidationError("indptr must start at 0 and end at len(indices)");
            for (int c = 0; c < n; ++c)
              if (indptr.data()[c] > indptr.data()[c + 1])
                throw dfm::ValidationError("indptr must be non-decreasing");
            for (int k = 0; k < nnz; ++k)
              if (indices.data()[k] < 0 || indices.data()[k] >= n)
                throw dfm::ValidationError("row index out of range");
            new (self) dfm::ComplexSparseLU(
                Eigen::Map<const dfm::SparseMatrixC>(n, n, nnz, indptr.data(), indices.data(), data.data()));
          },
          "indptr"_a, "indices"_a, "data"_a)
      .def_prop_ro("size", &dfm::ComplexSparseLU::size)
      .def(
          "solve",
          [](const dfm::ComplexSparseLU& s, const CArray1& rhs) {
            if (static_cast<int>(rhs.shape(0)) != s.size())
              throw dfm::ValidationError("right-hand side has the wrong length");
            return to_numpy(s.solve(Eigen::Map<const dfm::VectorC>(rhs.data(), s.size())));
          },
          "rhs"_a)
      .def(
          "solve_adjoint",
          [](const dfm::ComplexSparseLU& s, const CArray1& rhs) {
            if (static_cast<int>(rhs.shape(0)) != s.size())
              throw dfm::ValidationError("right-hand side has the wrong length");
            return to_numpy(s.solve_conjugate_transpose(Eigen::Map<const dfm::VectorC>(rhs.data(), s.size())));
          },
          "rhs"_a);

  // Tape round trips for testing the external operations end to end:
  // register A and B on the CoDiPack tape, record the operation, seed the
  // output cotangent, evaluate, and read Abar / Bbar back.
  struct TapeRoundTrip {
    static void seed_and_evaluate(std::vector<dfm::ActiveReal>& outputs, const Complex* cot) {
      auto& tape = dfm::ActiveReal::getTape();
      for (std::size_t k = 0; k < outputs.size() / 2; ++k) {
        tape.registerOutput(outputs[2 * k]);
        tape.registerOutput(outputs[2 * k + 1]);
      }
      tape.setPassive();
      for (std::size_t k = 0; k < outputs.size() / 2; ++k) {
        outputs[2 * k].setGradient(cot[k].real());
        outputs[2 * k + 1].setGradient(cot[k].imag());
      }
      tape.evaluate();
    }
    static dfm::VectorC read_complex_gradient(const std::vector<dfm::ActiveReal>& v, std::size_t begin,
                                              std::size_t count) {
      dfm::VectorC out(static_cast<Index>(count));
      for (std::size_t k = 0; k < count; ++k) {
        out(static_cast<Index>(k)) = Complex(v[begin + 2 * k].getGradient(), v[begin + 2 * k + 1].getGradient());
      }
      return out;
    }
  };

  auto make_pattern = [](const IArray1& indptr, const IArray1& indices) {
    const int n = static_cast<int>(indptr.shape(0)) - 1;
    return std::make_shared<const dfm::SparsePattern>(
        n, std::vector<int>(indptr.data(), indptr.data() + indptr.shape(0)),
        std::vector<int>(indices.data(), indices.data() + indices.shape(0)));
  };

  // Runs `op` on active copies of (data, b) and returns
  // (outputs, data_bar, b_bar), all complex, b and outputs column-major (n, m)
  // exposed as NumPy (m, n) transposed back by the Python caller.
  auto tape_vjp = [](int nnz, const CArray1& data, const CArray2& b_cm, const CArray2& cot_cm, auto&& op) {
    auto& tape = dfm::ActiveReal::getTape();
    dfm::invalidate_recorded_states();
    tape.reset();
    tape.setActive();
    const std::size_t m = b_cm.shape(0);
    const std::size_t n = b_cm.shape(1);
    std::vector<dfm::ActiveReal> inputs(2 * static_cast<std::size_t>(nnz) + 2 * n * m);
    for (int s = 0; s < nnz; ++s) {
      inputs[2 * s] = data.data()[s].real();
      inputs[2 * s + 1] = data.data()[s].imag();
    }
    for (std::size_t k = 0; k < n * m; ++k) {
      inputs[2 * nnz + 2 * k] = b_cm.data()[k].real();
      inputs[2 * nnz + 2 * k + 1] = b_cm.data()[k].imag();
    }
    for (auto& v : inputs) tape.registerInput(v);
    std::vector<dfm::ActiveReal> outputs(2 * n * m);
    op(inputs.data(), inputs.data() + 2 * nnz, static_cast<int>(m), outputs.data());
    TapeRoundTrip::seed_and_evaluate(outputs, cot_cm.data());

    dfm::MatrixC y(static_cast<Index>(m), static_cast<Index>(n));
    for (std::size_t k = 0; k < n * m; ++k) {
      y.data()[k] = Complex(outputs[2 * k].getValue(), outputs[2 * k + 1].getValue());
    }
    dfm::VectorC a_bar = TapeRoundTrip::read_complex_gradient(inputs, 0, static_cast<std::size_t>(nnz));
    dfm::VectorC b_flat = TapeRoundTrip::read_complex_gradient(inputs, 2 * nnz, n * m);
    dfm::MatrixC b_bar = Eigen::Map<dfm::MatrixC>(b_flat.data(), static_cast<Index>(m), static_cast<Index>(n));
    tape.reset();
    return nb::make_tuple(to_numpy(std::move(y)), to_numpy(std::move(a_bar)), to_numpy(std::move(b_bar)));
  };

  linalg.def(
      "solve_vjp",
      [make_pattern, tape_vjp](const IArray1& indptr, const IArray1& indices, const CArray1& data,
                               const CArray2& b_t, const CArray2& cot_t, bool conjugate_transpose) {
        auto pattern = make_pattern(indptr, indices);
        if (static_cast<int>(data.shape(0)) != pattern->nnz() || static_cast<int>(b_t.shape(1)) != pattern->size() ||
            b_t.shape(0) != cot_t.shape(0) || b_t.shape(1) != cot_t.shape(1)) {
          throw dfm::ValidationError("solve_vjp: inconsistent shapes");
        }
        const dfm::Op op = conjugate_transpose ? dfm::Op::ConjugateTranspose : dfm::Op::Normal;
        return tape_vjp(pattern->nnz(), data, b_t, cot_t,
                        [&](const dfm::ActiveReal* a, const dfm::ActiveReal* b, int m, dfm::ActiveReal* x) {
                          dfm::RecordedLinearSystem system(pattern, a);
                          system.solve(b, m, op, x);
                        });
      },
      "indptr"_a, "indices"_a, "data"_a, "b"_a, "cotangent"_a, "conjugate_transpose"_a = false,
      "Testing hook. b and cotangent are (m, n): row c is right-hand side c. Records "
      "X = op(A)^{-1} B on the CoDiPack tape and returns (X (m, n), Abar (nnz,), Bbar (m, n)) for "
      "cotangent Xbar = dL/dRe X + 1j dL/dIm X.");

  linalg.def(
      "matmul_vjp",
      [make_pattern, tape_vjp](const IArray1& indptr, const IArray1& indices, const CArray1& data,
                               const CArray2& b_t, const CArray2& cot_t, bool conjugate_transpose) {
        auto pattern = make_pattern(indptr, indices);
        if (static_cast<int>(data.shape(0)) != pattern->nnz() || static_cast<int>(b_t.shape(1)) != pattern->size() ||
            b_t.shape(0) != cot_t.shape(0) || b_t.shape(1) != cot_t.shape(1)) {
          throw dfm::ValidationError("matmul_vjp: inconsistent shapes");
        }
        const dfm::Op op = conjugate_transpose ? dfm::Op::ConjugateTranspose : dfm::Op::Normal;
        return tape_vjp(pattern->nnz(), data, b_t, cot_t,
                        [&](const dfm::ActiveReal* a, const dfm::ActiveReal* b, int m, dfm::ActiveReal* y) {
                          dfm::sparse_matmul_recorded(pattern, a, b, m, op, y);
                        });
      },
      "indptr"_a, "indices"_a, "data"_a, "b"_a, "cotangent"_a, "conjugate_transpose"_a = false,
      "Testing hook, as solve_vjp but for Y = op(A) B.");

  // ---------------------------------------------------------------- kernels
  nb::module_ kern = m.def_submodule("kernels", "Component kernels (upstream FUNCTIONS order).");
  kern.attr("NUM_FUNCTIONS") = dfm::kNumComponentFunctions;
  kern.attr("INPUT_WIDTH") = dfm::kComponentInputWidth;
  kern.def("names", [] {
    std::vector<std::string> out;
    for (int i = 0; i < dfm::kNumComponentFunctions; ++i) {
      out.emplace_back(dfm::name(static_cast<dfm::ComponentFunction>(i)));
    }
    return out;
  });
  kern.def("arity", [](int index) { return dfm::arity(dfm::component_function_from_index(index)); }, "index"_a);

  auto check_width = [](dfm::ComponentFunction f, size_t width) {
    if (static_cast<int>(width) < dfm::arity(f)) {
      throw dfm::ValidationError("parameters has " + std::to_string(width) + " entries, but " +
                                 std::string(dfm::name(f)) + " reads " + std::to_string(dfm::arity(f)) + ".");
    }
  };

  kern.def(
      "evaluate",
      [check_width](int index, const RArray1& params) {
        const auto f = dfm::component_function_from_index(index);
        check_width(f, params.shape(0));
        dfm::VectorC out(dfm::kComponentOutputs);
        dfm::evaluate_component(f, params.data(), out.data());
        return to_numpy(std::move(out));
      },
      "index"_a, "parameters"_a, "Kernel output, complex128 of length 4.");
  kern.def(
      "evaluate_batch",
      [check_width](int index, const RArray2& params) {
        const auto f = dfm::component_function_from_index(index);
        check_width(f, params.shape(1));
        dfm::MatrixC out(static_cast<Index>(params.shape(0)), dfm::kComponentOutputs);
        dfm::evaluate_components(f, static_cast<Index>(params.shape(0)), params.data(),
                                 static_cast<Index>(params.shape(1)), out.data());
        return to_numpy(std::move(out));
      },
      "index"_a, "parameters"_a, "Rows of parameters -> (rows, 4) outputs.");
  kern.def(
      "jacobian",
      [check_width](int index, const RArray1& params) {
        const auto f = dfm::component_function_from_index(index);
        check_width(f, params.shape(0));
        const int k = dfm::arity(f);
        dfm::VectorC out(dfm::kComponentOutputs);
        dfm::MatrixC jac(dfm::kComponentOutputs, k);
        dfm::component_jacobian(f, params.data(), out.data(), jac.data());
        return to_numpy(std::move(jac));
      },
      "index"_a, "parameters"_a, "d output_j / d parameter_k, complex (4, arity).");
  kern.def(
      "vjp",
      [check_width](int index, const RArray1& params, const CArray1& cot) {
        const auto f = dfm::component_function_from_index(index);
        check_width(f, params.shape(0));
        if (cot.shape(0) != dfm::kComponentOutputs) throw dfm::ValidationError("cotangent must have length 4");
        dfm::VectorR grad = dfm::VectorR::Zero(static_cast<Index>(params.shape(0)));
        dfm::component_vjp(f, params.data(), cot.data(), grad.data());
        return to_numpy(std::move(grad));
      },
      "index"_a, "parameters"_a, "cotangent"_a,
      "dL/dparameters (zero beyond the kernel's arity) for cotangent dL/dRe y + 1j dL/dIm y.");
  kern.def(
      "vjp_recorded",
      [check_width](int index, const RArray2& params, const CArray2& cot) {
        // Same as vjp, but through the CoDiPack tape (record_components):
        // checks the pushed statements against the jet Jacobian.
        const auto f = dfm::component_function_from_index(index);
        check_width(f, params.shape(1));
        const std::size_t rows = params.shape(0);
        const std::size_t width = params.shape(1);
        if (cot.shape(0) != rows || cot.shape(1) != dfm::kComponentOutputs) {
          throw dfm::ValidationError("cotangent must have shape (rows, 4)");
        }
        auto& tape = dfm::ActiveReal::getTape();
        dfm::invalidate_recorded_states();
        tape.reset();
        tape.setActive();
        std::vector<dfm::ActiveReal> in(rows * width);
        for (std::size_t k = 0; k < in.size(); ++k) {
          in[k] = params.data()[k];
          tape.registerInput(in[k]);
        }
        std::vector<dfm::ActiveReal> out(rows * 2 * dfm::kComponentOutputs);
        dfm::record_components(f, static_cast<Index>(rows), in.data(), static_cast<Index>(width), out.data());
        TapeRoundTrip::seed_and_evaluate(out, cot.data());
        dfm::MatrixR grad(static_cast<Index>(rows), static_cast<Index>(width));
        for (std::size_t k = 0; k < in.size(); ++k) grad.data()[k] = in[k].getGradient();
        tape.reset();
        return to_numpy(std::move(grad));
      },
      "index"_a, "parameters"_a, "cotangent"_a, "Batch VJP through the CoDiPack tape.");

  // ------------------------------------------------------------ diagnostics
  nb::module_ diag = m.def_submodule("diagnostics", "Toolchain smoke checks.");
  diag.def(
      "complex_gradient_smoke",
      [](const std::vector<double>& p) {
        dfm::GradientSmokeResult r = dfm::complex_gradient_smoke(p);
        dfm::VectorR g = Eigen::Map<const dfm::VectorR>(r.gradient.data(), static_cast<Index>(r.gradient.size()));
        return nb::make_tuple(r.value, to_numpy(std::move(g)), r.max_abs_error);
      },
      "p"_a);

  // -------------------------------------------------------------- constants
  nb::module_ consts = m.def_submodule("constants", "Physical constants (match components.py).");
  consts.attr("C_LIGHT") = dfm::constants::C_LIGHT;
  consts.attr("LAMBDA") = dfm::constants::LAMBDA;
  consts.attr("F0") = dfm::constants::F0;
  consts.attr("H_PLANCK") = dfm::constants::H_PLANCK;
  consts.attr("EPSILON_0C") = dfm::constants::EPSILON_0C;
}
