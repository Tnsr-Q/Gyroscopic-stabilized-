"""Field theory components: gauge fields, regulators, and field modifications."""

import numpy as np
import torch
from typing import Dict, Tuple
import sys
import os
# Add the quantum-core directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from pkgs.core_physics.utils import _ddx, _ddy, _hypercube_bits, _embed_4d_to_2d
from pkgs.core_physics.lattice import BondGraph, Edge

# Physical constants
c = 299792458.0
DTYPE = torch.float32

class ProperTimeGaugeField:
    """Gauge field for proper time evolution."""
    
    def __init__(self, grid_size: int = 32, device: str = "cpu"):
        self.grid_size = grid_size
        self.device = device
        self.A_mu = torch.zeros(4, grid_size, grid_size, device=device, dtype=DTYPE)
        self.upsilon_tensor = torch.zeros(4, 4, grid_size, grid_size, device=device, dtype=DTYPE)
        
    def set_from_wormhole_metric(self, r: float, r0: float):
        """Set gauge field from wormhole metric."""
        self.A_mu[0].fill_(-0.5 * np.log(1 + r/max(r0, 1e-9)) / c**2)
        self.A_mu[1].fill_(np.sqrt((r0**2/r)/max(r, r0)) if r > r0 else 0.0)
        
    def set_alena_modification(self, upsilon_tensor: torch.Tensor):
        """Set Alena soul modification tensor."""
        self.upsilon_tensor = upsilon_tensor.to(self.device)
        
    def compute_field_strength(self) -> torch.Tensor:
        """Compute electromagnetic field strength tensor."""
        F = torch.zeros_like(self.upsilon_tensor)
        # Compute time-space components (0,1) and (1,0) of antisymmetric 2-form
        F[0,1] = _ddy(self.A_mu[1]) - _ddx(self.A_mu[0])
        F[1,0] = -F[0,1]
        return F + self.upsilon_tensor
        
    def is_flat(self, tolerance: float = 1e-6) -> bool:
        """Check if field configuration is flat."""
        return torch.max(torch.abs(self.compute_field_strength())) < tolerance

    def zero_fields(self):
        """Set EM potentials and Υ to zero."""
        self.A_mu.zero_()
        self.upsilon_tensor.zero_()

class AlenaSoul:
    """Alena soul field modification system."""
    
    def __init__(self, kappa: float = 0.015):
        self.kappa = kappa
        
    def upsilon_from_A(self, A: torch.Tensor) -> torch.Tensor:
        """Compute upsilon tensor from gauge potential."""
        U = torch.zeros((4, 4, *A.shape[-2:]), dtype=DTYPE, device=A.device)
        # Only set (0,1) and (1,0) components (time-space components of antisymmetric 2-form)
        U[0,1] = self.kappa * (_ddy(A[1]) - _ddx(A[0]))
        U[1,0] = -U[0,1]
        return U
    
    def phase_imprint(self, g: BondGraph, u: torch.Tensor, L: float) -> Dict[Edge, float]:
        """Imprint phases on graph edges based on field magnitude."""
        mag = u.abs().amax(dim=(0,1))
        phi = {}
        scale = float(mag.mean().clamp_min(1e-6))
        
        for e in g.edges:
            xu, yu = _embed_4d_to_2d(_hypercube_bits(e.u), *mag.shape)
            xv, yv = _embed_4d_to_2d(_hypercube_bits(e.v), *mag.shape)
            phi[e] = float(L * 0.5 * np.pi * ((mag[xu,yu] + mag[xv,yv]).item() / (2.0 * scale)))
        return phi

class UVIRRegulator:
    """UV/IR regulator for bond dimension management."""
    
    def __init__(self, d_H: float = 3.12, C: float = 91.64, chi0: int = 2, 
                 chi_min: int = 2, chi_max: int = 16, eta: float = 0.12, 
                 lambda_phi: float = 0.35):
        self.d_H = d_H
        self.C = C
        self.chi0 = chi0
        self.chi_min = chi_min
        self.chi_max = chi_max
        self.eta = eta
        self.lambda_phi = lambda_phi
        
    def _K_bulk(self, N: int) -> float:
        """Compute bulk capacity factor."""
        Nb = max(4, int(N))
        return Nb / (self.C * (np.log(Nb + 1e-6)**self.d_H))
        
    def update_bonds(self, g: BondGraph, ss: Dict, pb: Dict, N: int) -> Tuple[Dict, Dict]:
        """Update bond dimensions and phases."""
        Kb = self._K_bulk(N)
        chi, phi = {}, {}
        
        for e in g.edges:
            chi_val = int(np.clip(
                self.chi0 * (1.0 + self.eta * Kb * float(ss.get(e, 0.0))), 
                self.chi_min, 
                self.chi_max
            ))
            chi[e] = chi_val
            phi[e] = float(pb.get(e, 0.0))
            
        return chi, phi