# gauge/proper_time_field.py
from __future__ import annotations
import torch
from typing import Dict, Tuple

Tensor = torch.Tensor

class ProperTimeGauge:
    """
    Discretize τ(x) on a lattice; compute links A_μ = ∂_μ τ and curvature F_μν.
    Flattening tries to find χ with τ -> τ - χ such that curvature lowers (gauge cooling).
    """
    def __init__(self, shape: Tuple[int,int], device="cpu"):
        self.shape = shape
        self.device = device
        self.tau = torch.zeros(shape, dtype=torch.float64, device=device)

    def set_tau(self, tau: Tensor):
        assert tau.shape == self.tau.shape
        self.tau = tau.clone()

    def links(self) -> Dict[str, Tensor]:
        # Forward differences with periodic boundary optional
        dx = torch.roll(self.tau, -1, dims=1) - self.tau
        dy = torch.roll(self.tau, -1, dims=0) - self.tau
        return {"Ax": dx, "Ay": dy}

    def curvature(self) -> Tensor:
        # F_xy = ∂_x Ay - ∂_y Ax (abelian)
        Ax, Ay = self.links()["Ax"], self.links()["Ay"]
        dAx_dy = torch.roll(Ax, -1, dims=0) - Ax
        dAy_dx = torch.roll(Ay, -1, dims=1) - Ay
        F = dAy_dx - dAx_dy
        return F

    def flatten(self, iters: int = 20, alpha: float = 0.2):
        """
        Gradient-like solve for χ: minimize ||F||^2 by adjusting τ -> τ - χ.
        (In abelian case, true curvature from gravity cannot be gauged away,
         but *relative phase* for the internal clock along selected paths can be.)
        """
        for _ in range(iters):
            F = self.curvature()
            # Laplacian step on τ as proxy to reduce local curl
            lap = (torch.roll(self.tau, -1, 0) + torch.roll(self.tau, 1, 0) +
                   torch.roll(self.tau, -1, 1) + torch.roll(self.tau, 1, 1) - 4*self.tau)
            self.tau = self.tau - alpha * lap
        return self.tau