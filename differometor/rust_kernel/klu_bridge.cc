#include <klu.h>
#include <new>

// Each sweep owns its symbolic analysis. Pivot and scale each new complex matrix.
struct Sweep {
    klu_common common{};
    klu_symbolic* symbolic{};
    klu_numeric* numeric{};
    int size;
};

extern "C" void* optical_prepare(int n, int, double*, int* rows, int* ptr) {
    auto* sweep = new (std::nothrow) Sweep{};
    if (!sweep) return nullptr;
    sweep->size = n;
    klu_defaults(&sweep->common);
    sweep->common.tol = 1.0;  // Select a maximum-magnitude pivot within each block.
    sweep->symbolic = klu_analyze(n, ptr, rows, &sweep->common);
    if (!sweep->symbolic) {
        delete sweep;
        return nullptr;
    }
    return sweep;
}

extern "C" void optical_finish(Sweep* sweep) {
    if (!sweep) return;
    klu_free_symbolic(&sweep->symbolic, &sweep->common);
    delete sweep;
}

extern "C" void* optical_factor(int, int, double* values, int* rows, int* ptr, Sweep* sweep) {
    sweep->numeric = klu_z_factor(ptr, rows, values, sweep->symbolic, &sweep->common);
    return sweep->numeric ? sweep : nullptr;
}

extern "C" int optical_solve(Sweep* sweep, int mode, int count, double* rhs) {
    int ok = mode == 0
        ? klu_z_solve(sweep->symbolic, sweep->numeric, sweep->size, count, rhs, &sweep->common)
        : klu_z_tsolve(sweep->symbolic, sweep->numeric, sweep->size, count, rhs, mode == 2, &sweep->common);
    return ok ? 0 : 1;
}

extern "C" void optical_release(Sweep* sweep) {
    klu_z_free_numeric(&sweep->numeric, &sweep->common);
}
