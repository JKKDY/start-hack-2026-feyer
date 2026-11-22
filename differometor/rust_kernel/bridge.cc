#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
using ComplexMatrix = ffi::BufferR2<ffi::C128>;
using IndexMatrix = ffi::BufferR2<ffi::S32>;
using ResultMatrix = ffi::ResultBufferR2<ffi::C128>;

extern "C" size_t feyer_function_count();
extern "C" void run_coefficients(bool, size_t, size_t, size_t, size_t,
                                 const void*, const int32_t*, const int32_t*, const void*, void*);

static bool valid_plan(ComplexMatrix x, IndexMatrix plan, IndexMatrix writes) {
  auto inputs = x.dimensions(), groups = plan.dimensions(), slots = writes.dimensions();
  if (inputs[0] <= 0 || inputs[1] <= 0 || groups[0] <= 0 || groups[1] != 11 ||
      slots[0] <= 0 || slots[1] != 2) return false;
  const auto functions = static_cast<int64_t>(feyer_function_count());
  for (int64_t i = 0; i < groups[0]; ++i) {
    const auto* row = plan.typed_data() + i * 11;
    if (row[0] < 0 || row[0] >= functions || row[1] < 0 || row[1] >= 128 ||
        row[2] < 0 || row[3] < row[2] || row[3] > slots[0]) return false;
    for (int col = 4; col < 11; ++col)
      if (row[col] < 0 || row[col] >= inputs[1]) return false;
  }
  for (int64_t i = 0; i < slots[0]; ++i) {
    const auto* slot = writes.typed_data() + i * 2;
    if (slot[0] < 0 || slot[0] >= 4 || slot[1] < 0 || slot[1] >= slots[0]) return false;
  }
  return true;
}

ffi::Error forward(ComplexMatrix x, IndexMatrix plan, IndexMatrix writes, ResultMatrix out) {
  auto d = x.dimensions(), w = writes.dimensions(), y = out->dimensions();
  if (!valid_plan(x, plan, writes) || y[0] != d[0] || y[1] != w[0])
    return ffi::Error::InvalidArgument("Invalid optical coefficient layout");
  run_coefficients(false, d[0], d[1], plan.dimensions()[0], w[0], x.typed_data(),
                   plan.typed_data(), writes.typed_data(), nullptr, out->typed_data());
  return ffi::Error::Success();
}

ffi::Error backward(ComplexMatrix x, IndexMatrix plan, IndexMatrix writes,
                    ComplexMatrix cotangent, ResultMatrix out) {
  auto d = x.dimensions(), w = writes.dimensions(), g = cotangent.dimensions(), y = out->dimensions();
  if (!valid_plan(x, plan, writes) || g[0] != d[0] || g[1] != w[0] ||
      y[0] != d[0] || y[1] != d[1])
    return ffi::Error::InvalidArgument("Invalid optical coefficient transpose layout");
  run_coefficients(true, d[0], d[1], plan.dimensions()[0], w[0], x.typed_data(),
                   plan.typed_data(), writes.typed_data(), cotangent.typed_data(), out->typed_data());
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(feyer_coeff_f, forward,
    ffi::Ffi::Bind().Arg<ComplexMatrix>().Arg<IndexMatrix>().Arg<IndexMatrix>().Ret<ComplexMatrix>());
XLA_FFI_DEFINE_HANDLER_SYMBOL(feyer_coeff_b, backward,
    ffi::Ffi::Bind().Arg<ComplexMatrix>().Arg<IndexMatrix>().Arg<IndexMatrix>()
                    .Arg<ComplexMatrix>().Ret<ComplexMatrix>());
