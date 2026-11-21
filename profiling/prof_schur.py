"""H4 stability probe: solve the signal system via a Schur complement over the structurally
eliminable unknowns and compare with the full pivoted LU solve. Uses prof_structure's state."""
import sys
sys.argv = sys.argv[:1]
exec(open(__file__.replace("prof_schur.py", "prof_structure.py")).read().split("# carrier too")[0])

rhs_all = sm[:, :, -1]
for label, mo, mf in (("zero fill", 1, 0), ("fill<=4", 4, 4)):
    m, nnz, order = eliminate(struct, unit_diag, mo, mf)
    E = np.array(sorted(order)); K = np.setdiff1d(np.arange(n), E)
    worst = 0.0
    for k in (0, 50, 100):
        M, b = S[k], rhs_all[k]
        x_ref = np.linalg.solve(M, b)
        A, B, C, D = M[np.ix_(K, K)], M[np.ix_(K, E)], M[np.ix_(E, K)], M[np.ix_(E, E)]
        Dinv_C, Dinv_g = np.linalg.solve(D, C), np.linalg.solve(D, b[E])
        xk = np.linalg.solve(A - B @ Dinv_C, b[K] - B @ Dinv_g)
        x = np.empty(n, complex); x[K] = xk; x[E] = Dinv_g - Dinv_C @ xk
        # also an adjoint-type solve with a detector-like RHS
        s = np.zeros(n, complex); s[:2 * N] = rng_vec = np.exp(1j * np.arange(2 * N))
        w_ref = np.linalg.solve(M.conj().T, s)
        Mh = M.conj().T
        Ah, Bh, Ch, Dh = Mh[np.ix_(K, K)], Mh[np.ix_(K, E)], Mh[np.ix_(E, K)], Mh[np.ix_(E, E)]
        wk = np.linalg.solve(Ah - Bh @ np.linalg.solve(Dh, Ch), s[K] - Bh @ np.linalg.solve(Dh, s[E]))
        e1 = np.linalg.norm(x - x_ref) / np.linalg.norm(x_ref)
        e2 = np.linalg.norm(wk - w_ref[K]) / np.linalg.norm(w_ref[K])
        print(f"  [{label}] f-index {k}: n_reduced={len(K)} cond(D)={np.linalg.cond(D):.1e} "
              f"rel err signal x={e1:.1e}  adjoint w={e2:.1e}")
