#include <slu_zdefs.h>
#include <algorithm>
#include <cstdint>
#include <vector>

static_assert(sizeof(int_t) == sizeof(int32_t), "Requires a 32-bit SuperLU index ABI");

// Rust owns the CSC arrays and the dense RHS workspace. This object owns
// SuperLU's descriptors and numeric factors for exactly one frequency.
struct Factors {
    SuperMatrix a{}, b{}, l{}, u{};
    SuperLUStat_t stats{};
    doublecomplex placeholder{};
    std::vector<int> columns, rows;
    explicit Factors(int n) : columns(n), rows(n) { StatInit(&stats); }
    ~Factors() {
        if (a.Store) Destroy_SuperMatrix_Store(&a);
        if (b.Store) Destroy_SuperMatrix_Store(&b);
        if (l.Store) Destroy_SuperNode_Matrix(&l);
        if (u.Store) Destroy_CompCol_Matrix(&u);
        StatFree(&stats);
    }
};

extern "C" void* optical_factor(int n, int nz, doublecomplex* data, int* rows,
                                int* ptr, int* ordering, bool prepare) {
    auto* factors = new Factors(n);
    zCreate_CompCol_Matrix(&factors->a, n, n, nz, data, rows, ptr, SLU_NC, SLU_Z, SLU_GE);
    superlu_options_t options;
    set_default_options(&options);
    if (prepare) get_perm_c(COLAMD, &factors->a, ordering);
    // SuperLU may postorder its permutation. Preserve the original for the next frequency.
    std::copy(ordering, ordering + n, factors->columns.begin());
    options.ColPerm = MY_PERMC;
    zCreate_Dense_Matrix(&factors->b, n, 0, &factors->placeholder, n, SLU_DN, SLU_Z, SLU_GE);
    int info = 0;
    zgssv(&options, &factors->a, factors->columns.data(), factors->rows.data(),
          &factors->l, &factors->u, &factors->b, &factors->stats, &info);
    if (info) {
        delete factors;
        return nullptr;
    }
    return factors;
}

extern "C" int optical_solve(Factors* factors, int operation, int nrhs, doublecomplex* values) {
    factors->b.ncol = nrhs;
    static_cast<DNformat*>(factors->b.Store)->nzval = values;
    int info = 0;
    const trans_t mode = operation == 0 ? NOTRANS : operation == 1 ? TRANS : CONJ;
    zgstrs(mode, &factors->l, &factors->u, factors->columns.data(), factors->rows.data(),
           &factors->b, &factors->stats, &info);
    return info;
}

extern "C" void optical_release(Factors* factors) { delete factors; }
