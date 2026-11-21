"""Re-evaluate the branch's final noise contraction from verified inputs (M, s, w, V) in JAX and NumPy."""
import os, sys
os.environ["JAX_PLATFORMS"] = "cpu"
import numpy as np, jax, jax.numpy as jnp
from jax import lax, vmap
jax.config.update("jax_enable_x64", True)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tests"))
from cases import CASES
import differometor as df
from differometor.build import LINKING_FUNCTIONS
from differometor.static import unwrap_static
from differometor.components import UNIT_VACUUM, H_PLANCK, F0
sim = sys.modules["differometor.simulate"]
cap = {}
orig = sim.solve_signal_and_adjoint
def spy(coef, pattern, srhs, drhs):
    out = orig(coef, pattern, srhs, drhs); cap.update(w=np.asarray(out[1]), drhs=np.asarray(drhs)); return out
sim.solve_signal_and_adjoint = spy
name = sys.argv[1]
c = CASES[name]()
A, *_ = df.run_build_step(c["setup"], c["changing_pairs"], jnp.asarray(c["changing_values"]), c["pairs"], signal_solver="sparse")
A["optimized_parameters"] = jnp.asarray(c["initial"])
_, _, noise_branch = df.simulate(**A)
noise_branch = np.asarray(noise_branch).reshape(-1)
U = unwrap_static(A)
P = U["parameters"].at[[0], U["optimized_parameter_indices"]].set(A["optimized_parameters"][U["optimized_value_indices"]])
P = P.at[[0], U["indices_to_link"]].set(vmap(lambda i, x: lax.switch(i, LINKING_FUNCTIONS, x))(U["linking_function_indices"], P[0, U["linked_indices"]]))
P = sim.expand_parameters(P, U["carrier_changing_parameter_indices"], U["carrier_changing_values"])
P = sim.expand_parameters(P, U["signal_changing_parameter_indices"], U["signal_changing_values"])
V = vmap(sim.update, in_axes=(0, None, None, None, None, None))(P, U["noise_matrix"], U["noise_function_input_indices"],
        U["noise_output_indices"], U["noise_system_matrix_indices"], U["noise_function_indices"])
w = jnp.transpose(jnp.asarray(cap["w"]), (2, 0, 1))                      # (D, F, n), as in simulate
const = 2 * UNIT_VACUUM * H_PLANCK * F0 * 0.25
jax_var = jnp.real(jnp.sum(w.conj() * jnp.einsum("bij,dbj->dbi", V, w), axis=-1))   # branch lines, verbatim
Vn, wn = np.asarray(V), np.asarray(w)[0]
np_var = np.array([np.real(np.vdot(wn[k], Vn[k] @ wn[k])) for k in range(len(Vn))])
terms = np.array([np.abs(wn[k].conj() * (Vn[k] @ wn[k])).sum() for k in range(len(Vn))])
for lab, var in (("JAX re-evaluation", np.asarray(jax_var)[0]), ("NumPy re-evaluation", np_var)):
    print(f"{lab:22s} max |ln noise - branch output| = {np.abs(0.5 * np.log(var * const) - np.log(noise_branch)).max():.1e}")
print(f"cancellation: sum|terms| / |sum| = max {np.max(terms / np.abs(np_var)):.1e}")
ref = np.load(os.path.join(HERE, "..", "tests", "reference", f"{name}.npz"))["noise"].reshape(-1)
print(f"branch output vs reference: max |ln| diff {np.abs(np.log(noise_branch) - np.log(ref)).max():.1e}; "
      f"NumPy w^H V w vs reference {np.abs(0.5 * np.log(np_var * const) - np.log(ref)).max():.1e}")
