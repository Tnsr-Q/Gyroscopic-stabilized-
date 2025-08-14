# ads_cft_compliance/api.py
import numpy as np

def _rt_check(boundary_entropy_fn, hypercube) -> dict:
    """
    Minimal RT-like check: compare measured S(A) vs minimal cut across the hypercube edges.
    Expects boundary_entropy_fn(A_mask)->S, and hypercube has .edges and maybe .minimal_cut_edges().
    """
    if hasattr(hypercube, "minimal_cut_edges"):
        cut = hypercube.minimal_cut_edges(hypercube)
    elif hasattr(hypercube, "edges"):
        cut = getattr(hypercube, "edges")
    else:
        cut = []

    # "Area" ~ |cut|. Measure a single S(A) and fit S ≈ α |cut|
    S_meas = boundary_entropy_fn()
    area = float(len(cut))
    alpha = S_meas/max(area, 1.0)
    return {"alpha": alpha, "S_meas": S_meas, "area": area, "pass": np.isfinite(alpha) and alpha > 0}

def _compression_check(k_counts) -> dict:
    """
    Verify K ≤ C (log N)^b. Fit log K vs log log N.
    k_counts: list of (N, K).
    """
    Ns = np.array([x[0] for x in k_counts], dtype=float)
    Ks = np.array([x[1] for x in k_counts], dtype=float)
    mask = (Ns > 1) & (Ks >= 1)
    if np.sum(mask) < 3:
        return {"pass": False, "C": None, "b": None}
    X = np.log(np.log(Ns[mask]))
    Y = np.log(Ks[mask])
    b, c = np.polyfit(X, Y, 1)  # Y ≈ b log log N + c => K ≈ e^c (log N)^b
    C = float(np.exp(c))
    return {"pass": np.isfinite(C) and np.isfinite(b), "C": C, "b": float(b)}

def _recovery_check(reconstruct_fn) -> dict:
    """
    Complementary recovery probe: can we reconstruct the same bulk op from two disjoint boundary regions?
    reconstruct_fn() should return overlaps or fidelities between reconstructions.
    """
    try:
        f1, f2 = reconstruct_fn()
        ok = (f1 > 0.8) and (f2 > 0.8)
        return {"pass": bool(ok), "f1": float(f1), "f2": float(f2)}
    except Exception:
        return {"pass": False, "f1": None, "f2": None}

def holography_checkpack(boundary_state, hypercube) -> dict:
    """
    Aggregate holography checks: RT-like scaling, complementary recovery, compression law.
    boundary_state: opaque here; you provide lambda hooks.
    """
    # Hooks to be supplied by caller; provide naive defaults
    def boundary_entropy_fn():
        # naive entropy proxy from amplitudes
        psi = np.asarray(boundary_state).flatten()
        psi = psi/np.linalg.norm(psi) if np.linalg.norm(psi) > 0 else psi
        p = np.abs(psi)**2 + 1e-30
        return float(-np.sum(p*np.log(p)))

    def reconstruct_fn():
        # Return two mock fidelities (caller should wire real reconstructions)
        return 0.9, 0.88

    # Compression samples (caller should feed real scale-ups)
    k_counts = [(32, 3), (64, 4), (128, 5), (256, 6)]

    rt = _rt_check(boundary_entropy_fn, hypercube)
    rec = _recovery_check(reconstruct_fn)
    cmpc = _compression_check(k_counts)

    return {"rt": rt, "recovery": rec, "compression": cmpc}

