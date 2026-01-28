# inflation_service/api.py
import numpy as np

def _slowroll(U, K, h):
    """
    U(h), K(h)=(dχ/dh)^2. Compute ε(h), η(h) in terms of h:
    ε = (1/2)(U_χ/U)^2 = (1/2)(U_h / (U * sqrt(K)))^2
    η ≈ (U_χχ / U) with U_χχ = (U_hh/K) - (K_h U_h)/(2 K^2)
    """
    dh = np.gradient(h)
    Uh = np.gradient(U, h, edge_order=2)
    Uhh = np.gradient(Uh, h, edge_order=2)
    Kh = np.gradient(K, h, edge_order=2)

    eps = 0.5 * (Uh / (U*np.sqrt(K) + 1e-30))**2
    eta = (Uhh/K - (Kh*Uh)/(2*K**2 + 1e-30)) / (U + 1e-30)
    return eps, eta

def _N_of_h(U, K, h, h_end):
    """
    N(h) = ∫_{h_end}^{h} (U / (U_h)) sqrt(K) dh
    """
    Uh = np.gradient(U, h, edge_order=2)
    integrand = (U / (Uh + 1e-30)) * np.sqrt(K)
    # integrate from h_end to each h
    idx_end = np.argmin(np.abs(h - h_end))
    N = np.zeros_like(h)
    if idx_end < len(h)-1:
        N[idx_end:] = np.cumsum(integrand[idx_end:] * np.gradient(h[idx_end:]))
    if idx_end > 0:
        N[:idx_end] = -np.cumsum(integrand[:idx_end][::-1] * np.gradient(h[:idx_end])[::-1])[::-1]
    return N

def higgs_inflation_observables(running: dict, N_star=50, kappa=1.0/(2.435e18**2)) -> dict:
    """
    Take running + precomputed Einstein-frame U(h),K(h) as arrays if provided.
    If not, approximate U(h) at large field via λ(h) h^4 / (4 Ω^4).
    Outputs ns, r at N=N_star (naive).
    """
    mu = running["mu"]
    lam = running["lambda"]
    xi = running["xi"]

    h = np.logspace(2, 19, 1200)
    lam_h = np.interp(h, mu, lam)
    xi_h = np.interp(h, mu, xi)
    Omega2 = 1.0 + kappa*xi_h*h**2
    U = (0.25*lam_h*h**4)/(Omega2**2)
    K = (1.0 + kappa*xi_h*(1.0+6.0*xi_h)*h**2) / (Omega2**2)

    eps, eta = _slowroll(U, K, h)
    # end of inflation: ε = 1
    mask = eps < 1.0
    if not np.any(mask):
        return {"ns": None, "r": None, "A_s": None, "h_star": None, "h_end": None}
    idx_end = np.where(~mask)[0]
    h_end = h[idx_end[0]] if idx_end.size > 0 else h[-1]

    N = _N_of_h(U, K, h, h_end)
    # find h* with N ≈ N_star
    idx_star = np.argmin(np.abs(N - N_star))
    h_star = h[idx_star]
    eps_star, eta_star = eps[idx_star], eta[idx_star]

    ns = 1.0 - 6.0*eps_star + 2.0*eta_star
    r = 16.0*eps_star

    # crude A_s normalization (requires H^2 ~ U/3κ^-1 and A_s ~ U/(24π^2 ε Mpl^4)); drop constants
    As = U[idx_star]/(24*np.pi**2*eps_star/(kappa**2) + 1e-30)

    return {"ns": float(ns), "r": float(r), "A_s": float(As), "h_star": float(h_star), "h_end": float(h_end)}
