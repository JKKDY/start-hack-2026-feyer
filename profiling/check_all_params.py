"""Outputs + gradient with EVERY setup parameter optimised (incl. alpha, loss, refractive index, phase),
to exercise the traced fallback of the constant-parameter path. Writes an .npz per case."""
import os, sys
os.environ["JAX_PLATFORMS"] = "cpu"
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:1])
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import differometor as df
from differometor.setups import aligo, uifo
out = sys.argv[1]
res = {}
for name, (S, _) in (("aligo", aligo()), ("uifo2", uifo(2, random=True, random_seed=11))):
    pairs = [list(p) for p in S.parameters]
    data = S.to_data()
    vals = np.array([float(data["nodes" if n in data["nodes"] else "edges"][n]["properties"][k]) for n, k in pairs])
    arrays, ports, *_ = df.run_build_step(S, [("f", "frequency")], jnp.logspace(1, 3.5, 9), pairs, signal_solver="packed", pack_noise=True)
    ports = jnp.asarray(ports)
    def loss(p):
        c, s, n = df.simulate(**dict(arrays, optimized_parameters=p))
        sig = df.signal_detector(c, s)[ports]
        det = sig[0] if sig.shape[0] == 1 else sig[0] - sig[1]
        return jnp.mean(jnp.log10(n.reshape(-1) / jnp.abs(det))), (c, s, n)
    fwd = jax.jit(lambda p: jax.vjp(loss, p, has_aux=True))
    l, pb, (c, s, n) = fwd(jnp.asarray(vals))
    g = jax.jit(lambda pb: pb(jnp.asarray(1.0))[0])(pb)
    kinds = sorted({k for _, k in pairs})
    res.update({f"{name}_{k}": np.asarray(v) for k, v in (("carrier", c), ("signal", s), ("noise", n), ("grad", g))})
    res[f"{name}_kinds"] = np.array(kinds)
    print(f"{name}: {len(pairs)} optimised parameters of kinds {kinds}; loss {float(l):.6f}")
np.savez(out, **res)
