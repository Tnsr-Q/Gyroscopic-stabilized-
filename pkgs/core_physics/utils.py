"""
Mathematical utilities and helper functions for quantum physics computations.

Contains finite difference operators, embedding functions, and RG planning utilities.
"""
import torch
import numpy as np
from typing import Tuple, Optional


def _ddx(x: torch.Tensor) -> torch.Tensor:
    """Finite difference in x direction with safe padding."""
    # pad=(left,right,top,bottom) for 2D tensors: pad last dim here
    return x - torch.nn.functional.pad(x, (1, 0, 0, 0), mode="replicate")[..., :, :-1]


def _ddy(x: torch.Tensor) -> torch.Tensor:
    """Finite difference in y direction with safe padding."""
    return x - torch.nn.functional.pad(x, (0, 0, 1, 0), mode="replicate")[..., :-1, :]


def _hypercube_bits(i: int) -> Tuple[int, ...]:
    """Extract 4D hypercube coordinate bits from vertex index."""
    return ((i>>0)&1, (i>>1)&1, (i>>2)&1, (i>>3)&1)


def _embed_4d_to_2d(bits: Tuple, H: int, W: int) -> Tuple[int, int]:
    """Embed 4D hypercube coordinates into 2D grid."""
    return (min(H-1, bits[0]*(H//2)+bits[2]*(H//4)), 
            min(W-1, bits[1]*(W//2)+bits[3]*(W//4)))


class RGPlanner:
    """Renormalization Group Planner for boundary-driven scaling and Planck factor"""
    
    def __init__(self, mu0=1.0, L_rg=1.0, d_boundary=2, eps=1e-6):
        self.mu0, self.L_rg, self.d = mu0, L_rg, d_boundary
        self.eps = eps
        self.c_ref = None  # set on first call
        self.prev_c_rel = 1.0  # for monotonicity enforcement

    def mu_of_r(self, r: float) -> float:
        """Renormalization scale as function of radius"""
        return self.mu0 * np.exp(-r / max(self.L_rg, self.eps))

    def c_hat(self, r: float, dH_est: Optional[float], S_slope: Optional[float]=None) -> float:
        """Relative speed parameter based on entanglement or spectral dimension"""
        # Simple proxy: prefer entanglement slope if available; else d_H
        if S_slope is not None:
            c_raw = max(S_slope, self.eps)
        elif dH_est is not None:
            c_raw = max(dH_est, self.eps)
        else:
            c_raw = 1.0
            
        # Set reference on first call
        if self.c_ref is None: 
            self.c_ref = c_raw
            
        # Compute relative speed and enforce monotonicity
        c_rel = c_raw / max(self.c_ref, self.eps)
        c_rel = min(c_rel, self.prev_c_rel)  # Enforce non-increasing
        self.prev_c_rel = c_rel  # Update for next call
        return c_rel

    def planck_scale_factor(self, r: float, c_rel: float) -> float:
        """Compute Planck scale factor: M_Pl / M_Pl0 = (c_rel)^{1/(d-1)}"""
        exponent = 1.0 / max(1, self.d - 1)
        return float((c_rel + self.eps) ** exponent)