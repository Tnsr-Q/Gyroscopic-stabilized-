# timeops/three_time_ops.py
from __future__ import annotations
import numpy as np
import torch
from dataclasses import dataclass
from typing import Tuple

Tensor = torch.Tensor

@dataclass
class ThreeTimeClock:
    """
    Minimal 3D-time clock: three canonical pairs (τ_i, K_i).
    We represent τ_i as POVM-style phase coordinates and K_i as
    Hermitian generators acting on the internal clock Hilbert space H_clock.
    """
    d: int = 64                 # clock Hilbert dimension per axis
    hbar: float = 1.0
    device: str = "cpu"

    def __post_init__(self):
        self._build_algebra()

    def _build_algebra(self):
        # Discrete phase basis for each axis (Fourier pair)
        n = torch.arange(self.d, device=self.device, dtype=torch.float64)
        # Momentum-like spectrum for K (periodic clock)
        k_spec = 2.0*np.pi*(n - self.d//2)/self.d
        self.K1 = torch.diag(k_spec).to(torch.complex128)        # Hermitian
        self.K2 = torch.diag(k_spec).to(torch.complex128)
        self.K3 = torch.diag(k_spec).to(torch.complex128)

        # τ operators as conjugates via discrete Fourier transform F
        F = torch.fft.fft(torch.eye(self.d, dtype=torch.complex128, device=self.device), dim=0)/np.sqrt(self.d)
        tau_spec = torch.linspace(-np.pi, np.pi, self.d, dtype=torch.float64, device=self.device)
        Tau_diag = torch.diag(tau_spec).to(torch.complex128)
        self.Tau1 = (F.conj().T @ Tau_diag @ F)  # POVM-like angle operator
        self.Tau2 = (F.conj().T @ Tau_diag @ F)
        self.Tau3 = (F.conj().T @ Tau_diag @ F)

    # --- Unitary evolution and echo gates ---
    def U_path(self, lambdas: Tuple[float,float,float]) -> Tensor:
        """U^γ = exp(-i (λ1 K1 + λ2 K2 + λ3 K3)) acting on H_clock."""
        lam1, lam2, lam3 = [torch.tensor(x, dtype=torch.float64, device=self.device) for x in lambdas]
        H = lam1*self.K1 + lam2*self.K2 + lam3*self.K3
        return torch.linalg.matrix_exp(-1j*H)

    def R_echo(self, delta_taus: Tuple[float,float,float]) -> Tensor:
        """
        Time-recompression gate R^Δ = exp(+ i (Δτ_1 K1 + Δτ_2 K2 + Δτ_3 K3)).
        Applying this cancels the relative proper-time phase.
        """
        d1, d2, d3 = [torch.tensor(x, dtype=torch.float64, device=self.device) for x in delta_taus]
        H = d1*self.K1 + d2*self.K2 + d3*self.K3
        return torch.linalg.matrix_exp(+1j*H)

    def visibility(self, psi0: Tensor, U1: Tensor, U2: Tensor) -> float:
        """
        Interference visibility = |⟨ψ| U1† U2 |ψ⟩|.
        Low visibility ⇒ decoherence from Δτ; after echo, should go back to ~1.
        """
        ov = (psi0.conj() @ (U1.conj().T @ (U2 @ psi0))).item()
        return float(abs(ov))