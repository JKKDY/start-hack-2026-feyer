"""Componentwise accuracy of SuperLU vs dense LU on the adjoint (noise-weight) solve, against 40-digit truth,
and the effect of standard remedies (SuperLU equilibration, fixed-precision iterative refinement).

    EXTRA_PATH=<dir with mpmath> python profiling/diag_superlu_accuracy.py voyager_space
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
from scipy.sparse.linalg import splu  # noqa: E402

sim = sys.modules["differometor.simulate"]
cap = {}
orig = sim.solve_signal_and_adjoint


def spy(coef, pattern, srhs, drhs):
    cap.update(coef=np.asarray(coef), pattern=[np.asarray(p) for p in pattern], drhs=np.asarray(drhs))
    return orig(coef, pattern, srhs, drhs)


sim.solve_signal_and_adjoint = spy
name = sys.argv[1]
c = CASES[name]()
A, *_ = df.run_build_step(c["setup"], c["changing_pairs"], jnp.asarray(c["changing_values"]), c["pairs"],
                          signal_solver="sparse")
A["optimized_parameters"] = jnp.asarray(c["initial"])
df.simulate(**A)
U = unwrap_static(A)
P = U["parameters"].at[[0], U["optimized_parameter_indices"]].set(A["optimized_parameters"][U["optimized_value_indices"]])
P = P.at[[0], U["indices_to_link"]].set(
    vmap(lambda i, x: lax.switch(i, LINKING_FUNCTIONS, x))(U["linking_function_indices"], P[0, U["linked_indices"]]))
P = sim.expand_parameters(P, U["carrier_changing_parameter_indices"], U["carrier_changing_values"])
P = sim.expand_parameters(P, U["signal_changing_parameter_indices"], U["signal_changing_values"])
V_all = np.asarray(vmap(sim.update, in_axes=(0, None, None, None, None, None))(
    P, U["noise_matrix"], U["noise_function_input_indices"], U["noise_output_indices"],
    U["noise_system_matrix_indices"], U["noise_function_indices"]))
rows, cols, ptr = cap["pattern"]
F, n, D = cap["drhs"].shape


def solvers(Ms):
    """Ways to compute w = M^-H d. Each returns a function d -> w."""
    Md = Ms.toarray()
    out = {"dense LAPACK LU": lambda d: np.linalg.solve(Md.conj().T, d)}
    lu = splu(Ms)
    out["SuperLU default (branch)"] = lambda d: lu.solve(d, trans="H")
    lu_eq = splu(Ms, options=dict(Equil=True))
    out["SuperLU Equil=True"] = lambda d: lu_eq.solve(d, trans="H")

    def refined(steps):
        def f(d):
            w = lu.solve(d, trans="H")
            for _ in range(steps):
                w = w + lu.solve(d - Ms.conj().T @ w, trans="H")   # fixed-precision iterative refinement
            return w
        return f
    out["SuperLU + 1 refinement step"] = refined(1)
    out["SuperLU + 2 refinement steps"] = refined(2)
    return out


# per-frequency S error of each method vs dense LU (dense was validated against 40-digit truth)
methods = None
errs = {}
for k in range(F):
    Ms = csc_matrix((cap["coef"][k], rows, ptr), shape=(n, n))
    d = cap["drhs"][k][:, 0]
    sv = solvers(Ms)
    S = {lab: np.real(np.vdot(f(d), V_all[k] @ f(d))) for lab, f in sv.items()}
    for lab, val in S.items():
        errs.setdefault(lab, []).append(0.5 * abs(np.log(val / S["dense LAPACK LU"])))
worst_k = int(np.argmax(errs["SuperLU default (branch)"]))
print(f"{name}: max error in ln(noise) over {F} frequencies, relative to dense LU:")
for lab, e in errs.items():
    print(f"  {lab:30s} {max(e):.1e}")

if os.environ.get("NO_TRUTH"):
    sys.exit(0)
# 40-digit truth at the worst frequency
k = worst_k
Ms = csc_matrix((cap["coef"][k], rows, ptr), shape=(n, n))
Md = Ms.toarray()
d = cap["drhs"][k][:, 0]
mp.mp.dps = 40
Mm = mp.matrix([[mp.mpc(complex(x)) for x in r] for r in Md])
dm = mp.matrix([mp.mpc(complex(x)) for x in d])
wt = mp.lu_solve(Mm.H, dm)
w_true = np.array([complex(wt[i]) for i in range(n)])
Vm = mp.matrix([[mp.mpc(complex(x)) for x in r] for r in V_all[k]])
S_true = float(mp.re((wt.H * Vm * wt)[0]))
print(f"\nworst frequency k={k}, against 40-digit truth:")
print(f"  {'method':30s} {'err ln(noise)':>14s} {'max rel err (normwise)':>24s} {'worst componentwise rel err':>28s}")
for lab, f in solvers(Ms).items():
    w = f(d)
    S = np.real(np.vdot(w, V_all[k] @ w))
    comp = np.abs(w - w_true) / np.maximum(np.abs(w_true), 1e-300)
    print(f"  {lab:30s} {0.5 * abs(np.log(S / S_true)):14.1e} {np.abs(w - w_true).max() / np.abs(w_true).max():24.1e} "
          f"{comp[np.abs(w_true) > 0].max():28.1e}")
mag = np.abs(w_true[w_true != 0])
print(f"  |w| dynamic range: {mag.min():.1e} .. {mag.max():.1e}")
