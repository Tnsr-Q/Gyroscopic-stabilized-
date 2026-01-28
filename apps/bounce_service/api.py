# bounce_service/api.py
import numpy as np

def bounce_action(V, include_gravity: bool = False, Mpl: float = 2.435e18, r_max: float = 1e6) -> dict:
    """
    O(4)-symmetric bounce (thin shooting, no gravity by default).
    h'' + (3/r) h' = dV/dh, with h'(0)=0, h(∞)=false vacuum.
    V: callable V(h); we approximate dV/dh numerically.
    Returns S4 and a crude rate estimate.
    """
    # Find false vacuum (min near h=0); crude scan
    hs = np.logspace(-6, 6, 2000)
    Vs = np.array([V(x) for x in hs])
    idx_min = np.argmin(Vs)
    h_f = hs[idx_min]

    def dVdh(h):
        eps = 1e-6*(1+abs(h))
        return (V(h+eps)-V(h-eps))/(2*eps)

    # Shooting on initial value h(0) ≈ h0 (near true vacuum side)
    # crude bracket: try a ladder of h0 values > h_f
    h0_grid = np.geomspace(1e-3+abs(h_f), 1e3+abs(h_f), 40)
    best = None
    for h0 in h0_grid:
        r = 0.0
        hr = h0
        hp = 0.0   # h'(r)
        dr = 1e-3
        for _ in range(200000):
            # RK2
            k1 = hp
            l1 = dVdh(hr) - (3.0/max(r,1e-12))*hp
            h_mid = hr + 0.5*dr*k1
            hp_mid = hp + 0.5*dr*l1
            k2 = hp_mid
            l2 = dVdh(h_mid) - (3.0/max(r+0.5*dr,1e-12))*hp_mid
            hr += dr*k2
            hp += dr*l2
            r += dr
            # stop when near false vacuum
            if abs(hr - h_f) < 1e-6 or r > r_max:
                break
        # boundary mismatch metric
        mismatch = abs(hr - h_f)
        if best is None or mismatch < best[0]:
            best = (mismatch, h0, r, hr)

    # With best trajectory, compute S4
    # S4 = 2π^2 ∫ r^3 [ 1/2 h'^2 + V(h) - V(h_f) ] dr
    h0 = best[1]
    r = 0.0
    hr = h0
    hp = 0.0
    dr = 1e-3
    S4 = 0.0
    for _ in range(200000):
        k1 = hp
        l1 = dVdh(hr) - (3.0/max(r,1e-12))*hp
        h_mid = hr + 0.5*dr*k1
        hp_mid = hp + 0.5*dr*l1
        e_density = 0.5*hp**2 + V(hr) - V(h_f)
        S4 += (2*np.pi**2) * (r**3) * e_density * dr
        k2 = hp_mid
        l2 = dVdh(h_mid) - (3.0/max(r+0.5*dr,1e-12))*hp_mid
        hr += dr*k2
        hp += dr*l2
        r += dr
        if abs(hr - h_f) < 1e-6 or r > r_max:
            break

    rate = None if S4 is None else f"~ μ^4 * exp(-{S4:.2f})"
    return {"S4": float(S4), "profile": None, "rate": rate, "h_false": float(h_f)}
