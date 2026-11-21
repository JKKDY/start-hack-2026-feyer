"""H1/H4 structure analysis: what the index arrays and the assembled matrices look like.

Replicates the matrix-assembly part of differometor.simulate.simulate (no solves) so we
can inspect the actual carrier / signal / noise matrices. Read-only w.r.t. differometor/.
"""
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__file__))
from common import build_scored  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax import lax, vmap  # noqa: E402
from differometor.components import FUNCTIONS  # noqa: E402
from differometor.simulate import expand_parameters, update, solve  # noqa: E402
from differometor.build import LINKING_FUNCTIONS  # noqa: E402

case = sys.argv[1] if len(sys.argv) > 1 else "uifo_800_3000.json"
ref = sys.argv[2] if len(sys.argv) > 2 else "uifo_800_3000_test_data.json"
arrays, ports, parameters, freqs, expected, pairs = build_scored(case, ref)
A = {k: v for k, v in arrays.items()}
names = [f.__name__ for f in FUNCTIONS]

print("=== index / data arrays passed to simulate (all become jit ARGUMENTS in benchmark.py) ===")
total_bytes = 0
for k, v in A.items():
    leaves = v if isinstance(v, tuple) else (v,)
    for i, leaf in enumerate(leaves):
        leaf = np.asarray(leaf)
        total_bytes += leaf.nbytes
        print(f"  {k + ('[%d]' % i if isinstance(v, tuple) else ''):40s} {str(leaf.dtype):10s} {str(leaf.shape):16s}")
print(f"  total {total_bytes/1e6:.1f} MB")

print("\n=== H1: function usage per matrix (entries evaluated via vmap(lax.switch)) ===")
for which in ("carrier", "signal", "noise"):
    fi = np.asarray(A[f"{which}_function_indices"])
    c = Counter(fi.tolist())
    print(f"  {which}: {fi.size} function evaluations, {len(c)} distinct of {len(FUNCTIONS)} functions:")
    print("     " + ", ".join(f"{names[i]}x{n}" for i, n in sorted(c.items())))
    print(f"     input block shape {np.asarray(A[f'{which}_function_input_indices']).shape}, "
          f"outputs written {np.asarray(A[f'{which}_output_indices'][0]).size}")

# ---- replicate assembly -------------------------------------------------
P = A["parameters"].at[[0], A["optimized_parameter_indices"]].set(parameters[A["optimized_value_indices"]])
inputs_to_link = P[0, A["linked_indices"]]
outs = vmap(lambda i, x: lax.switch(i, LINKING_FUNCTIONS, x))(A["linking_function_indices"], inputs_to_link)
P = P.at[[0], A["indices_to_link"]].set(outs)
P = expand_parameters(P, A["carrier_changing_parameter_indices"], A["carrier_changing_values"])
cm = vmap(update, in_axes=(0, None, None, None, None, None))(
    P, A["carrier_matrix"], A["carrier_function_input_indices"], A["carrier_output_indices"],
    A["carrier_matrix_indices"], A["carrier_function_indices"])
carrier = vmap(solve)(cm[:, :, :-1], cm[:, :, -1])
P = expand_parameters(P, A["signal_changing_parameter_indices"], A["signal_changing_values"])
sm = vmap(update, in_axes=(0, None, None, None, None, None))(
    P, A["signal_matrix"], A["signal_function_input_indices"], A["signal_output_indices"],
    A["signal_matrix_indices"], A["signal_function_indices"])
nm = vmap(update, in_axes=(0, None, None, None, None, None))(
    P, A["noise_matrix"], A["noise_function_input_indices"], A["noise_output_indices"],
    A["noise_system_matrix_indices"], A["noise_function_indices"])
cm, sm, nm = np.asarray(cm), np.asarray(sm), np.asarray(nm)
N = carrier.shape[1]


def pattern(base, idx):
    pat = np.asarray(base) != 0
    r, c = np.asarray(idx[0]), np.asarray(idx[1])
    pat[r, c] = True
    return pat


print("\n=== H4: sparsity ===")
for name, mats, base, idx in (("carrier", cm[:, :, :-1], A["carrier_matrix"], A["carrier_matrix_indices"]),
                              ("signal", sm[:, :, :-1], A["signal_matrix"], A["signal_matrix_indices"]),
                              ("noise", nm, A["noise_matrix"], A["noise_system_matrix_indices"])):
    n = mats.shape[-1]
    numeric = (mats != 0).any(axis=0)
    struct = pattern(base, idx)[:, :n] if name != "noise" else pattern(base, idx)
    print(f"  {name}: n={n}  numeric nnz={numeric.sum()} ({100*numeric.mean():.2f}%)  "
          f"structural nnz={struct.sum()} ({100*struct.mean():.2f}%)  batch={mats.shape[0]}")

# ---- signal matrix row classification -----------------------------------
S = sm[:, :, :-1]
n = S.shape[-1]
struct = pattern(A["signal_matrix"], A["signal_matrix_indices"])[:, :n] | (S != 0).any(axis=0)
diag = np.diagonal(S, axis1=1, axis2=2)
offdeg = struct.sum(axis=1) - struct[np.arange(n), np.arange(n)]
coldeg = struct.sum(axis=0) - struct[np.arange(n), np.arange(n)]
unit_diag = np.all(diag == 1, axis=0)
rhs = sm[:, :, -1]
print("\n=== signal matrix rows by block ===")
blocks = [("upper sideband", 0, N), ("lower sideband", N, 2 * N), ("signal+mech", 2 * N, n)]
for label, a, b in blocks:
    deg = Counter(offdeg[a:b].tolist())
    print(f"  {label:15s} rows {a}-{b-1}: unit diag {unit_diag[a:b].sum()}/{b-a}; off-diag count histogram "
          + str(dict(sorted(deg.items()))))
print(f"  rows with nonzero RHS: {(np.abs(rhs) > 0).any(axis=0).sum()}")
print(f"  max column degree {coldeg.max()}, max row degree {offdeg.max()}")

# ---- greedy exact elimination (structural) ------------------------------
# Eliminate unknown i via its own row when row i has a unit diagonal and <= 1 off-diagonal
# (x_i = b_i - a_ij x_j). Substitution merges column i into column j: never increases nnz.
# Then also allow eliminating rows with <= k off-diagonals and report fill-in.


def eliminate(struct, unitdiag, max_off, max_fill=0):
    st = struct.copy()
    alive = np.ones(n, bool)
    order = []
    changed = True
    while changed:
        changed = False
        for i in np.where(alive & unitdiag)[0]:
            row = st[i] & alive
            row[i] = False
            js = np.where(row)[0]
            if len(js) > max_off:
                continue
            col = st[:, i] & alive
            col[i] = False
            ks = np.where(col)[0]
            # fill-in: new entries (k, j) for k in ks, j in js
            fill = sum((~st[k, js]).sum() for k in ks)
            if fill > max_fill:
                continue
            for k in ks:
                st[k, js] = True
            alive[i] = False
            order.append(i)
            changed = True
    red = st[np.ix_(alive, alive)]
    return alive.sum(), red.sum(), order


for label, mo, mf in (("unit-diag, <=1 off-diag, zero fill (space-type rows)", 1, 0),
                      ("unit-diag, <=2 off-diag, zero fill", 2, 0),
                      ("unit-diag, <=4 off-diag, fill <= 4", 4, 4)):
    m, nnz, order = eliminate(struct, unit_diag, mo, mf)
    by_block = Counter("upper" if i < N else "lower" if i < 2 * N else "sig/mech" for i in order)
    print(f"\n  elimination [{label}]: n {n} -> {m}  (eliminated {dict(by_block)}); "
          f"reduced nnz {nnz} ({100*nnz/m/m:.1f}% dense); dense-LU cost ratio (n/m)^3 = {(n/m)**3:.1f}x")

# carrier too
C = cm[:, :, :-1]
cs = pattern(A["carrier_matrix"], A["carrier_matrix_indices"])[:, :N] | (C != 0).any(axis=0)
cud = np.all(np.diagonal(C, axis1=1, axis2=2) == 1, axis=0)
n_save = n
n = N
m, nnz, _ = eliminate(cs, cud, 1, 0)
print(f"\n  carrier elimination [<=1 off-diag, zero fill]: {N} -> {m}")
n = n_save

# condition numbers
print("\n=== conditioning (signal matrix, a few frequencies) ===")
def equilibrate(M, iters=20):
    M = M.copy()
    for _ in range(iters):   # alternate row/column max-scaling (like LAPACK zgeequ, iterated)
        r = np.abs(M).max(axis=1); r[r == 0] = 1; M /= r[:, None]
        c = np.abs(M).max(axis=0); c[c == 0] = 1; M /= c[None, :]
    return M


for k in (0, len(S) // 2, len(S) - 1):
    print(f"  f={float(freqs[k]):7.1f} Hz  cond_2 = {np.linalg.cond(S[k]):.2e}   "
          f"after row/col equilibration = {np.linalg.cond(equilibrate(S[k])):.2e}")
mags = np.abs(S[0][S[0] != 0])
print(f"  |entries| range {mags.min():.1e} .. {mags.max():.1e}")
print(f"  carrier cond_2 = {np.linalg.cond(C[0]):.2e}")
