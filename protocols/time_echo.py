# protocols/time_echo.py
from __future__ import annotations
import torch
import numpy as np
from typing import Tuple, Dict
from timeops.three_time_ops import ThreeTimeClock

Tensor = torch.Tensor

class TimeRecompressionProtocol:
    def __init__(self, clock: ThreeTimeClock):
        self.clock = clock

    def estimate_delta_tau(self, psi0: Tensor, U1: Tensor, U2: Tensor) -> Tuple[float,float,float]:
        """
        Estimate vector Δτ by phase tomography:
        For small Δ, the phase φ ≈ ⟨K⟩·Δτ. We probe along each axis by toggling Ki.
        (Works as fast, local estimator; more exact fits are possible.)
        """
        with torch.no_grad():
            # Expectation values <K_i> in state ψ0
            Ek1 = torch.real((psi0.conj()* (self.clock.K1 @ psi0)).sum())
            Ek2 = torch.real((psi0.conj()* (self.clock.K2 @ psi0)).sum())
            Ek3 = torch.real((psi0.conj()* (self.clock.K3 @ psi0)).sum())
            # Net phase from overlap U1† U2
            ov = (psi0.conj() @ (U1.conj().T @ (U2 @ psi0))).item()
            phi = torch.tensor([np.angle(ov)], dtype=torch.float64).item()
            # Distribute φ across components by relative weights (regularized)
            w = torch.tensor([abs(Ek1), abs(Ek2), abs(Ek3)], dtype=torch.float64)
            w = w / (w.sum() + 1e-12)
            Δ = (phi*w[0].item(), phi*w[1].item(), phi*w[2].item())
            return Δ

    def run_echo(self, psi0: Tensor, lambdas1, lambdas2) -> Dict[str,float]:
        """
        Prepare two path evolutions U^γ1, U^γ2, measure visibility,
        estimate Δτ, apply R^Δ on path 2, and report improvement.
        """
        U1 = self.clock.U_path(lambdas1)
        U2 = self.clock.U_path(lambdas2)

        vis_before = self.clock.visibility(psi0, U1, U2)
        Δ = self.estimate_delta_tau(psi0, U1, U2)
        RΔ = self.clock.R_echo(Δ)

        vis_after = self.clock.visibility(psi0, U1, RΔ @ U2)
        return {"vis_before": vis_before, "vis_after": vis_after,
                "d_tau_1": Δ[0], "d_tau_2": Δ[1], "d_tau_3": Δ[2]}