"""H2: cost of the dense batched solves in isolation (1 CPU).

Compares what simulate() does now (3 independent jnp.linalg.solve per frequency) with
one LU factorisation reused for all three solves, and shows the n^3 scaling that makes
reducing n (H4) attractive.
"""
import json
import sys
import time

sys.path.insert(0, __import__("os").path.dirname(__file__))
from common import pin_one_cpu  # noqa: E402

pin_one_cpu()
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.scipy.linalg import lu_factor, lu_solve  # noqa: E402

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_enable_compilation_cache", False)
B = 101


def bench(name, fn, *args, reps=2):
    t = time.perf_counter()
    c = jax.jit(fn).lower(*args).compile()
    tc = time.perf_counter() - t
    jax.block_until_ready(c(*args))  # warm
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        jax.block_until_ready(c(*args))
        ts.append(time.perf_counter() - t)
    r = {"case": name, "compile": round(tc, 3), "exec": round(min(ts), 3)}
    print("RESULT " + json.dumps(r), flush=True)
    return r


rng = np.random.default_rng(0)
for n in [int(a) for a in (sys.argv[1:] or ["751", "612", "264"])]:
    M = jnp.asarray(np.eye(n) + 0.3 * (rng.standard_normal((B, n, n)) + 1j * rng.standard_normal((B, n, n))) / np.sqrt(n))
    b = jnp.asarray(rng.standard_normal((B, n)) + 1j * rng.standard_normal((B, n)))

    def three_solves(M, b):   # current simulate(): solve(M), solve(M^H), solve(M)
        x = jax.vmap(jnp.linalg.solve)(M, b)
        w = jax.vmap(jnp.linalg.solve)(jnp.conj(jnp.swapaxes(M, 1, 2)), b)
        c = jax.vmap(jnp.linalg.solve)(M, w)
        return x, w, c

    def lu_reuse(M, b):       # one factorisation, three triangular solve pairs
        lu, piv = jax.vmap(lu_factor)(M)
        x = jax.vmap(lambda l, p, r: lu_solve((l, p), r))(lu, piv, b)
        w = jax.vmap(lambda l, p, r: lu_solve((l, p), r, trans=2))(lu, piv, b)
        c = jax.vmap(lambda l, p, r: lu_solve((l, p), r))(lu, piv, w)
        return x, w, c

    def one_solve(M, b):
        return jax.vmap(jnp.linalg.solve)(M, b)

    r1 = bench(f"n={n} 1x solve", one_solve, M, b)
    r3 = bench(f"n={n} 3x solve (current)", three_solves, M, b)
    rl = bench(f"n={n} lu_factor + 3x lu_solve", lu_reuse, M, b)
    x3, xl = three_solves(M, b), lu_reuse(M, b)
    print(f"   agreement max rel diff {max(float(jnp.max(jnp.abs(a - c)) / jnp.max(jnp.abs(a))) for a, c in zip(x3, xl)):.1e}")
    flops = B * (8 / 3) * n**3
    print(f"   effective rate of one batched LU: {flops / r1['exec'] / 1e9:.1f} GFLOP/s (complex LU = 8/3 n^3 real flops)")
