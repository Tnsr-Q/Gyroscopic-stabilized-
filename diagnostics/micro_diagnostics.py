import numpy as np
from typing import Callable, Optional, Tuple, List

def _shannon_entropy_from_patch(patch: np.ndarray) -> float:
    """Entropy of normalized |patch|^2 as a cheap proxy."""
    x = np.abs(patch.astype(np.complex128))**2
    p = x / (x.sum() + 1e-12)
    nz = p[p > 1e-15]
    return float(-(nz * np.log(nz)).sum())

def area_volume_exponent(field: np.ndarray, max_L: Optional[int] = None) -> Tuple[float, List[float]]:
    """
    Returns exponent alpha in S(L) ~ L^alpha (alpha≈1 area-law, alpha≈2 volume-law),
    using entropy proxy on L×L square patches anchored at (0,0).
    """
    H, W = field.shape[:2]
    max_L = max_L or min(H, W, 32)
    Ls = np.arange(2, max_L+1)
    Svals = []
    for L in Ls:
        patch = field[:L, :L]
        Svals.append(_shannon_entropy_from_patch(patch))
    Lfit = np.log(Ls[2:]); Sfit = np.log(np.maximum(Svals[2:], 1e-12))
    slope = float(np.polyfit(Lfit, Sfit, 1)[0])
    return slope, Svals

def mutual_information_decay(field: np.ndarray,
                             box: int = 6,
                             max_sep: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Semiquantitative MI proxy between two box regions separated along x.
    Uses Shannon proxy on |patch|^2. Returns (separations, MI).
    """
    H, W = field.shape[:2]
    max_sep = max_sep or min(W//2 - box - 2, 24)
    def S_box(x0, y0):
        return _shannon_entropy_from_patch(field[y0:y0+box, x0:x0+box])
    y0 = max(0, H//2 - box//2)
    xA = max(0, W//4 - box//2)
    seps = []
    MI = []
    for dx in range(1, max_sep+1):
        xB = xA + box + dx
        if xB + box >= W: break
        S_A = S_box(xA, y0)
        S_B = S_box(xB, y0)
        S_AB = _shannon_entropy_from_patch(
            np.block([[field[y0:y0+box, xA:xA+box], field[y0:y0+box, xB:xB+box]]])
        )
        I = S_A + S_B - S_AB
        seps.append(dx)
        MI.append(I)
    return np.array(seps), np.array(MI)

def fhs_chern_number(states_fn: Callable[[float, float], np.ndarray],
                     Nk: int = 17) -> Optional[int]:
    """
    Fukui-Hatsugai-Suzuki Chern on a discrete k-grid.
    Requires states_fn(kx, ky) -> normalized eigenvector (complex 1D array).
    Returns None if states_fn is None; else an integer Chern.
    """
    if states_fn is None: return None
    ks = np.linspace(-np.pi, np.pi, Nk, endpoint=False)
    F = 0.0
    def U(u, v):  # normalized inner product
        return np.vdot(u, v) / (np.linalg.norm(u)*np.linalg.norm(v) + 1e-12)
    for i, kx in enumerate(ks):
        kx_n = ks[(i+1) % Nk]
        for j, ky in enumerate(ks):
            ky_n = ks[(j+1) % Nk]
            u = states_fn(kx, ky)
            ux = states_fn(kx_n, ky)
            uy = states_fn(kx, ky_n)
            uxy = states_fn(kx_n, ky_n)
            # plaquette Berry phase
            W = U(u, ux) * U(ux, uxy) * U(uxy, uy) * U(uy, u)
            F += np.angle(W)
    C = int(np.rint(F / (2*np.pi)))
    return C