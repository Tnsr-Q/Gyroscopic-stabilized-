# veff_service/api.py
from dataclasses import dataclass
import numpy as np

@dataclass
class VeffOptions:
    scheme: str = "MSbar"
    loops: int = 1
    curvature_R: float = 0.0
    Bfield: float = 0.0
    non_minimal: bool = True
    einstein_frame: bool = True
    kappa: float = 1.0/(2.435e18**2)  # 1/Mpl^2 (reduced Mpl)

def _masses_SM(h, running):
    """Field-dependent masses at tree-level: W, Z, top, Higgs/Goldstone (rough)."""
    g1, g2, g3 = running["g"].T  # (n,3)
    yt = running["yt"]
    lam = running["lambda"]

    # pick closest μ ~ h for RG-improvement-by-choice-of-scale
    mu = running["mu"]
    idx = np.clip(np.searchsorted(mu, np.maximum(h, mu[0]))-1, 0, len(mu)-1)
    g1h, g2h, yth, lamh = g1[idx], g2[idx], yt[idx], lam[idx]

    mW = 0.5*g2h*h
    mZ = 0.5*np.sqrt(g1h**2 + g2h**2)*h
    mt = yth*h/np.sqrt(2.0)
    # crude Higgs/Goldstones (large field): mh^2 ~ 3λh^2, mG^2 ~ λ h^2
    mH = np.sqrt(np.maximum(3.0*lamh, 0.0))*h
    mG = np.sqrt(np.maximum(lamh, 0.0))*h
    return mW, mZ, mt, mH, mG

def _cw_one_loop(h, running):
    """Coleman-Weinberg 1-loop contribution (MSbar; c_i = 5/6 for gauge, 3/2 otherwise)."""
    mW, mZ, mt, mH, mG = _masses_SM(h, running)
    mu = np.clip(h, running["mu"][0], running["mu"][-1])  # μ ~ h

    def F(m, c):
        x = np.maximum(m**2, 1e-32)
        return x**2*(np.log(x/mu**2) - c)

    # d.o.f.: W:6, Z:3, top: -12 (fermion minus sign), H:1, Goldstones:3
    term = (6*F(mW, 5/6) + 3*F(mZ, 5/6) - 12*F(mt, 3/2) + 1*F(mH, 3/2) + 3*F(mG, 3/2))
    return term/(64*np.pi**2)

def einstein_frame_potential(h, running, opts: VeffOptions):
    """Return U(h) = V(h)/Ω(h)^4 and optional canonical χ(h) mapping."""
    lam = np.interp(h, running["mu"], running["lambda"])
    Vtree = 0.25*lam*h**4  # large-field regime
    V1 = _cw_one_loop(h, running) if opts.loops >= 1 else 0.0

    V = Vtree + V1
    if opts.non_minimal:
        xi = np.interp(h, running["mu"], running["xi"])
        Omega2 = 1.0 + opts.kappa*xi*h**2
        U = V / (Omega2**2)
        # canonical field χ: (dχ/dh)^2 = K(h)
        K = (1.0 + opts.kappa*xi*(1.0+6.0*xi)*h**2) / (Omega2**2)
        return U, K
    else:
        return V, np.ones_like(h)

def higgs_veff(h: np.ndarray, running: dict, options: dict) -> dict:
    """
    Compute RG-improved Higgs potential with optional non-minimal coupling (Einstein frame).
    """
    opts = VeffOptions(**options) if options else VeffOptions()
    h = np.asarray(h, dtype=float)

    if opts.einstein_frame:
        U, K = einstein_frame_potential(h, running, opts)
        res = {"h": h, "Veff": U, "K": K}
    else:
        lam = np.interp(h, running["mu"], running["lambda"])
        Vtree = 0.25*lam*h**4
        V1 = _cw_one_loop(h, running) if opts.loops >= 1 else 0.0
        res = {"h": h, "Veff": Vtree + V1}

    # Simple instability finder: μ where λ crosses 0
    lam_arr = running["lambda"]
    mu_arr = running["mu"]
    sgn = np.sign(lam_arr + 1e-20)
    zc = np.where(sgn[:-1]*sgn[1:] < 0)[0]
    meta_scale = mu_arr[zc[0]+1] if zc.size > 0 else None
    res["meta_scale"] = meta_scale
    res["lambda_at"] = np.interp(h, mu_arr, lam_arr)
    return res
