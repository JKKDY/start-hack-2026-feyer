"""Snapshot (or compare) run_build_step outputs for several setups, leaf by leaf, to prove a build refactor is exact.
    python profiling/snapshot_build.py save <file.pkl>  |  python profiling/snapshot_build.py compare <file.pkl>"""
import os, pickle, sys
os.environ["JAX_PLATFORMS"] = "cpu"
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
from cases import CASES
import differometor as df
from differometor.static import StaticArray

def leaves(tree):
    flat, _ = jax.tree_util.tree_flatten_with_path(tree, is_leaf=lambda x: isinstance(x, StaticArray) or x is None)
    out = {}
    for path, leaf in flat:
        key = jax.tree_util.keystr(path)
        if isinstance(leaf, StaticArray):
            out[key] = ("static", np.asarray(leaf.array))
        elif leaf is None:
            out[key] = ("none", None)
        else:
            out[key] = (type(leaf).__name__ if not isinstance(leaf, jax.Array) else "jax", np.asarray(leaf))
    return out

snap = {}
for name in sys.argv[3:] or CASES:
    c = CASES[name]()
    for solver in ("dense", "packed"):
        res = df.run_build_step(c["setup"], c["changing_pairs"], jnp.asarray(c["changing_values"]), c["pairs"],
                                signal_solver=solver, pack_noise=solver == "packed")
        snap[(name, solver)] = leaves(res)
if sys.argv[1] == "save":
    pickle.dump(snap, open(sys.argv[2], "wb")); print(f"saved {len(snap)} builds")
else:
    ref = pickle.load(open(sys.argv[2], "rb")); bad = 0
    for k, leaves_new in snap.items():
        leaves_old = ref[k]
        if leaves_old.keys() != leaves_new.keys():
            print(k, "tree structure differs"); bad += 1; continue
        for p, (kind_old, a) in leaves_old.items():
            kind_new, b = leaves_new[p]
            same = (a is None and b is None) or (a is not None and b is not None and a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b, equal_nan=True))
            if not same or kind_old != kind_new:
                bad += 1
                diff = "" if a is None or b is None or a.shape != b.shape else f" maxdiff={np.max(np.abs(a.astype(complex) - b.astype(complex))):.1e}"
                print(f"  DIFF {k} {p}: {kind_old}/{a.dtype if a is not None else None}{getattr(a,'shape','')} -> {kind_new}/{b.dtype if b is not None else None}{getattr(b,'shape','')}{diff}")
    print("IDENTICAL" if not bad else f"{bad} differences")
