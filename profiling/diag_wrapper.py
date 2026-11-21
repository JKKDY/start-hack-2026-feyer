"""Compare what solve_signal_and_adjoint() returns with a direct dense solve of the same inputs."""
import os, sys
os.environ["JAX_PLATFORMS"] = "cpu"
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
from cases import CASES
import differometor as df
from scipy.sparse import csc_matrix
sim = sys.modules["differometor.simulate"]
cap = {}
orig = sim.solve_signal_and_adjoint
def spy(coef, pattern, srhs, drhs):
    out = orig(coef, pattern, srhs, drhs)
    cap.update(coef=np.asarray(coef), pattern=[np.asarray(p) for p in pattern], srhs=np.asarray(srhs),
               drhs=np.asarray(drhs), x=np.asarray(out[0]), w=np.asarray(out[1]))
    return out
sim.solve_signal_and_adjoint = spy
c = CASES[sys.argv[1]]()
A, *_ = df.run_build_step(c["setup"], c["changing_pairs"], jnp.asarray(c["changing_values"]), c["pairs"], signal_solver="sparse")
A["optimized_parameters"] = jnp.asarray(c["initial"])
df.simulate(**A)
rows, cols, ptr = cap["pattern"]; F, n = cap["srhs"].shape
print("shapes: coef", cap["coef"].shape, "srhs", cap["srhs"].shape, "drhs", cap["drhs"].shape, "-> x", cap["x"].shape, "w", cap["w"].shape)
ex, ew = [], []
for k in range(F):
    M = csc_matrix((cap["coef"][k], rows, ptr), shape=(n, n)).toarray()
    x = np.linalg.solve(M, cap["srhs"][k]); w = np.linalg.solve(M.conj().T, cap["drhs"][k])
    ex.append(np.abs(cap["x"][k] - x).max() / np.abs(x).max()); ew.append(np.abs(cap["w"][k] - w).max() / np.abs(w).max())
ex, ew = np.array(ex), np.array(ew)
print(f"wrapper x vs dense: max {ex.max():.1e}   wrapper w vs dense: max {ew.max():.1e} at k={ew.argmax()}; per k (first 8): {np.array2string(ew[:8], precision=1)}")
