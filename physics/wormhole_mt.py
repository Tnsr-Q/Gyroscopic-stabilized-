# physics/wormhole_mt.py
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

# Physical constants (natural units by default)
c = 1.0  # set to 299792458.0 if you're running SI
G = 1.0

@dataclass
class MTWormhole:
    r0: float = 0.5                         # throat radius
    Phi: Callable[[float], float] = lambda r: 0.0   # redshift fn Φ(r); 0 => no horizon
    # Shape function b(r). Default: b(r) = r0^3 / r^2 (your expression)
    b: Callable[[float], float] = None

    def __post_init__(self):
        if self.b is None:
            self.b = lambda r: self.r0 * (self.r0 / max(r, 1e-9))**2

    def flare_out_ok(self, eps: float = 1e-6) -> bool:
        """Flare-out condition b(r0)=r0 and b'(r0) < 1."""
        b_r0 = self.b(self.r0)
        # numerical derivative
        rp, rm = self.r0*(1+eps), max(self.r0*(1-eps), 1e-9)
        bprime = (self.b(rp) - self.b(rm)) / (rp - rm)
        return (abs(b_r0 - self.r0) < 1e-6) and (bprime < 1.0)

    def metric_factors(self, r: float) -> Tuple[float, float]:
        """Return g_tt = e^{2Φ(r)}, g_rr = 1 / (1 - b(r)/r)."""
        one_minus = 1.0 - self.b(r)/max(r, 1e-9)
        g_tt = math.exp(2.0 * self.Phi(r))
        g_rr = 1.0 / max(one_minus, 1e-12)
        return g_tt, g_rr

    def V_eff_sq(self, r: float, Lz: float, m: float = 1.0) -> float:
        """
        Effective potential squared for timelike geodesics (equatorial).
        V^2 = (1 - b/r) * (m^2 c^4 e^{2Φ} + Lz^2 c^2 / r^2)
        With c=1, m=1 by default.
        """
        fac = max(1.0 - self.b(r)/max(r, 1e-9), 0.0) * math.exp(2.0*self.Phi(r))
        return fac * ((m*c*c)**2 + (Lz**2 * c*c) / (max(r, 1e-9)**2))

    def radial_equation(self, r: float, E: float, Lz: float, m: float = 1.0) -> float:
        """
        (dr/dτ)^2 = (E^2 - V_eff^2) / g_rr, clipped to >=0
        """
        g_tt, g_rr = self.metric_factors(r)
        V2 = self.V_eff_sq(r, Lz, m)
        num = max(E*E - V2, 0.0)
        return num / g_rr