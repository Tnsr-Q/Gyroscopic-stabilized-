# rge_service/api.py
from dataclasses import dataclass
import numpy as np
from .sm_beta import beta_lambda_sm_1l, beta_yt_sm_1l, beta_g_sm_1l, beta_xi_sm_1l

@dataclass
class Scenario:
    model: str = "SM"          # "SM" | "MSSM" | "SM+GUT" (MSSM/GUT hooks TODO)
    loops: int = 1
    inputs: dict = None        # {"mt":..., "mh":..., "alpha_s":..., "xi_0":...}
    thresholds: list = None    # [{"scale":..., "match":"..."}, ...]

def _rk4_step(y, f, t, h):
    k1 = f(t, y)
    k2 = f(t + 0.5*h, y + 0.5*h*k1)
    k3 = f(t + 0.5*h, y + 0.5*h*k2)
    k4 = f(t + h, y + h*k3)
    return y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def run_rge(scenario: dict, mu_grid: np.ndarray) -> dict:
    """
    Run minimal 1-loop SM RGEs for (λ, y_t, g1, g2, g3, ξ).
    Returns piecewise arrays on the provided mu_grid.
    """
    sc = Scenario(**scenario)
    mu = np.array(mu_grid, dtype=float)
    t = np.log(mu)  # t = ln μ
    n = len(t)

    # EW-ish boundary values (very rough defaults if not supplied)
    inp = {"mt": 172.5, "mh": 125.10, "alpha_s": 0.1181, "xi_0": 0.1}
    if sc.inputs: inp.update(sc.inputs)

    # Convert to MS-bar couplings at μ0 ~ m_t (rough)
    mu0 = mu[0]
    g3_0 = np.sqrt(4*np.pi*inp["alpha_s"])
    # EW couplings (tree-level-ish)
    v = 246.0
    yt_0 = np.sqrt(2)*inp["mt"]/v
    lam_0 = (inp["mh"]**2)/(2*v**2)
    g2_0 = 0.65
    g1_0 = 0.36
    xi_0 = inp.get("xi_0", 0.1)

    y = np.array([lam_0, yt_0, g1_0, g2_0, g3_0, xi_0], dtype=float)
    out = np.zeros((n, y.size))
    out[0] = y

    def flow(tloc, yloc):
        lam, yt, g1, g2, g3, xi = yloc
        beta_l = beta_lambda_sm_1l(lam, yt, g1, g2, g3)
        beta_y = beta_yt_sm_1l(yt, g1, g2, g3)
        beta_g1, beta_g2, beta_g3 = beta_g_sm_1l(g1, g2, g3)
        beta_x = beta_xi_sm_1l(xi, lam, yt, g1, g2)
        return np.array([beta_l, beta_y, beta_g1, beta_g2, beta_g3, beta_x], dtype=float) / (16*np.pi**2)

    for i in range(n-1):
        h = t[i+1]-t[i]
        y = _rk4_step(y, flow, t[i], h)
        # crude guards to avoid runaway/NaNs
        y = np.nan_to_num(y, nan=0.0, posinf=1e3, neginf=-1e3)
        out[i+1] = y

    return {
        "mu": mu,
        "lambda": out[:,0],
        "yt": out[:,1],
        "g": np.stack([out[:,2], out[:,3], out[:,4]], axis=-1),
        "xi": out[:,5]
    }
