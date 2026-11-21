"""Which noise formula is right on an ill-conditioned signal matrix? Ground truth in 40-digit arithmetic.

original (upstream):   w = M^-H s,  c = M^-1 (V w),  S = Re(s^H c)      (two solves)
sparse branch:         w = M^-H s,                   S = Re(w^H V w)    (identity s^H M^-1 = w^H)

Both are identical in exact arithmetic; they differ only in how float64 round-off enters.
Usage: python profiling/diag_noise_truth.py <case> [freq_index ...]   (needs mpmath on sys.path)
"""
import os
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
for p in os.environ.get("EXTRA_PATH", "").split(":"):
    if p:
        sys.path.insert(0, p)
import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax import lax, vmap  # noqa: E402

jax.config.update("jax_enable_x64", True)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tests"))
from cases import CASES  # noqa: E402
import differometor as df  # noqa: E402
from differometor.build import LINKING_FUNCTIONS  # noqa: E402
from differometor.static import unwrap_static  # noqa: E402
from scipy.sparse import csc_matrix  # noqa: E402

sim = sys.modules["differometor.simulate"]
cap = {}
orig = sim.solve_signal_and_adjoint


def spy(coef, pattern, srhs, drhs):
    cap.update(coef=np.asarray(coef), pattern=[np.asarray(p) for p in pattern], drhs=np.asarray(drhs))
    return orig(coef, pattern, srhs, drhs)


sim.solve_signal_and_adjoint = spy
name = sys.argv[1]
freqs = [int(a) for a in sys.argv[2:]] or [0]
c = CASES[name]()
A, ports, *_ = df.run_build_step(c["setup"], c["changing_pairs"], jnp.asarray(c["changing_values"]), c["pairs"],
                                 signal_solver="sparse")
A["optimized_parameters"] = jnp.asarray(c["initial"])
_, _, noise_sparse = df.simulate(**A)
noise_sparse = np.asarray(noise_sparse).reshape(-1, cap["coef"].shape[0])

# noise-source matrix V per frequency, replicated from simulate()
U = unwrap_static(A)
P = U["parameters"].at[[0], U["optimized_parameter_indices"]].set(A["optimized_parameters"][U["optimized_value_indices"]])
P = P.at[[0], U["indices_to_link"]].set(
    vmap(lambda i, x: lax.switch(i, LINKING_FUNCTIONS, x))(U["linking_function_indices"], P[0, U["linked_indices"]]))
P = sim.expand_parameters(P, U["carrier_changing_parameter_indices"], U["carrier_changing_values"])
P = sim.expand_parameters(P, U["signal_changing_parameter_indices"], U["signal_changing_values"])
V_all = np.asarray(vmap(sim.update, in_axes=(0, None, None, None, None, None))(
    P, U["noise_matrix"], U["noise_function_input_indices"], U["noise_output_indices"],
    U["noise_system_matrix_indices"], U["noise_function_indices"]))

ref = np.load(os.path.join(HERE, "..", "tests", "reference", f"{name}.npz"))
noise_ref = np.asarray(ref["noise"]).reshape(-1, cap["coef"].shape[0])
rows, cols, ptr = cap["pattern"]
n = cap["drhs"].shape[1]
mp.mp.dps = 40
to_mp = lambda a: mp.matrix([[mp.mpc(complex(x)) for x in row] for row in np.atleast_2d(a)])  # noqa: E731

print(f"{name}: n={n}; |log10 noise| differences, 0.5*|ln S - ln S_true| (= error in ln noise)")
for k in freqs:
    M = csc_matrix((cap["coef"][k], rows, ptr), shape=(n, n)).toarray()
    V = V_all[k]
    s = cap["drhs"][k][:, 0]
    w = np.linalg.solve(M.conj().T, s)
    S_orig = np.real(s.conj() @ np.linalg.solve(M, V @ w))
    S_branch = np.real(w.conj() @ V @ w)
    # sanity: these must reproduce what the harness compared (up to the ASD constant)
    const = np.asarray(noise_ref[0, k]) ** 2 / S_orig
    print(f"  k={k}: capture check  orig formula vs reference noise ratio {const:.6e}; "
          f"branch formula vs branch output ratio {noise_sparse[0, k] ** 2 / S_branch:.6e}")
    Mm, Vm, sm = to_mp(M), to_mp(V), to_mp(s).T
    wm = mp.lu_solve(Mm.H, sm)
    cm = mp.lu_solve(Mm, Vm * wm)
    S_true = mp.re((sm.H * cm)[0])
    S_true_alt = mp.re((wm.H * Vm * wm)[0])
    t = float(S_true)
    print(f"        S_true={t:.12e} (alt form agrees to {float(abs(S_true - S_true_alt) / abs(S_true)):.1e})")
    for lab, val in (("original 2-solve", S_orig), ("branch  w^H V w ", S_branch)):
        err = 0.5 * abs(np.log(val / t)) if val > 0 else float("inf")
        print(f"        {lab}: S={val:.12e}  error in ln(noise) = {err:.2e}")
