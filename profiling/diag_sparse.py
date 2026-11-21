"""Diagnose sparse-vs-dense discrepancies.

Captures the coefficients/pattern/RHS that simulate() hands to the sparse solver (eager run,
so the arrays are concrete) and compares SuperLU with dense LU: solutions, backward
residuals and conditioning, per frequency.

    python profiling/diag_sparse.py voyager_space
"""
import os
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from cases import CASES  # noqa: E402
import differometor as df  # noqa: E402
from scipy.sparse import csc_matrix  # noqa: E402
from scipy.sparse.linalg import splu  # noqa: E402

sim = sys.modules["differometor.simulate"]
cap = {}
orig = sim.solve_signal_and_adjoint


def spy(coef, pattern, srhs, drhs):
    cap.update(coef=np.asarray(coef), pattern=[np.asarray(p) for p in pattern],
               srhs=np.asarray(srhs), drhs=np.asarray(drhs))
    return orig(coef, pattern, srhs, drhs)


sim.solve_signal_and_adjoint = spy

name = sys.argv[1]
c = CASES[name]()
A, ports, *_ = df.run_build_step(c["setup"], c["changing_pairs"], jnp.asarray(c["changing_values"]), c["pairs"],
                                 signal_solver="sparse")
A["optimized_parameters"] = jnp.asarray(c["initial"])
df.simulate(**A)  # eager

rows, cols, ptr = cap["pattern"]
F, n = cap["srhs"].shape
D = cap["drhs"].shape[-1]
print(f"{name}: F={F} n={n} detectors={D} nnz={len(rows)}")


def bres(M, x, b):
    return np.linalg.norm(M @ x - b) / (np.linalg.norm(M, 1) * np.linalg.norm(x) + np.linalg.norm(b))


worst = []
for k in range(F):
    M = csc_matrix((cap["coef"][k], rows, ptr), shape=(n, n))
    Md = M.toarray()
    lu = splu(M)
    d = cap["drhs"][k]
    w_sp = lu.solve(d, trans="H")
    w_de = np.linalg.solve(Md.conj().T, d)
    x_sp = lu.solve(cap["srhs"][k])
    x_de = np.linalg.solve(Md, cap["srhs"][k])
    rel_w = np.abs(w_sp - w_de).max() / np.abs(w_de).max()
    rel_x = np.abs(x_sp - x_de).max() / np.abs(x_de).max()
    worst.append((rel_w, k, rel_x, bres(Md.conj().T, w_sp, d), bres(Md.conj().T, w_de, d)))
worst.sort(reverse=True)
print("worst frequencies: rel diff w (adjoint), freq idx, rel diff x (signal), backward residual w sparse / dense")
for rel_w, k, rel_x, r_sp, r_de in worst[:3]:
    print(f"  w {rel_w:.1e}  k={k:3d}  x {rel_x:.1e}  res sparse {r_sp:.1e}  dense {r_de:.1e}")
k = worst[0][1]
Md = csc_matrix((cap["coef"][k], rows, ptr), shape=(n, n)).toarray()
s = np.linalg.svd(Md, compute_uv=False)
print(f"  freq {k}: cond2 {s[0] / s[-1]:.1e}; smallest singular values {np.array2string(s[-4:], precision=2)}")

# --- which unknowns span the near-null space? ------------------------------------
U, s, Vh = np.linalg.svd(Md)
gap = s[:-1] / s[1:]
null = np.where(s < s[0] * 1e-25)[0]
carrier_n = n_carrier = A["carrier_matrix"].shape[0]
def label(i):
    return ("upper" if i < n_carrier else "lower" if i < 2 * n_carrier else "sig/mech") + f"[{i}]"
for j in null:
    r, l = np.abs(Vh[j].conj()), np.abs(U[:, j])
    print(f"  sigma={s[j]:.1e}: right null vector on {[label(i) for i in np.argsort(r)[::-1][:4]]} "
          f"(weights {np.sort(r)[::-1][:4].round(3)})")
    print(f"               left  null vector on {[label(i) for i in np.argsort(l)[::-1][:4]]} "
          f"(weights {np.sort(l)[::-1][:4].round(3)})")
    for i in np.argsort(r)[::-1][:2]:
        nzr = np.nonzero(Md[i])[0]; nzc = np.nonzero(Md[:, i])[0]
        print(f"     unknown {label(i)}: row entries {[(label(j2), complex(np.round(Md[i, j2], 4))) for j2 in nzr]}")
        print(f"                       column entries {[(label(j2), complex(np.round(Md[j2, i], 4))) for j2 in nzc]}")
d0 = cap["drhs"][k][:, 0]
print(f"  detector vector overlap with left null vectors: {[f'{abs(U[:, j].conj() @ d0) / np.linalg.norm(d0):.1e}' for j in null]}")
