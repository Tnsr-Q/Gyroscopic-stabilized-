"""Control systems: gyroscope feeler, signature normalization, and Jacobian hypercube."""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple
import sys
import os
# Add the quantum-core directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from pkgs.core_physics.lattice import BondGraph, Edge

class GyroscopeFeeler:
    """Gyroscopic sensing system for bond graph signatures."""
    
    def __init__(self, graph: BondGraph):
        self.graph = graph
        
    def gluewalk_signature(self, chi: Dict[Edge, int], phi: Dict[Edge, float], b_chi: int) -> torch.Tensor:
        """Compute gluon walk signature from bond dimensions and phases."""
        E = len(self.graph.edges)
        if E == 0:
            return torch.zeros(8, dtype=torch.float32)
            
        chi_a = np.fromiter((max(1, chi.get(e, b_chi)) for e in self.graph.edges), dtype=np.float64, count=E)
        phi_a = np.fromiter((float(phi.get(e, 0.0)) for e in self.graph.edges), dtype=np.float64, count=E)
        
        dlog = np.log(chi_a) - np.log(b_chi)
        c_phi, s_phi = np.cos(phi_a), np.sin(phi_a)
        
        s1 = dlog.mean()
        s5, s6 = c_phi.mean(), s_phi.mean()
        s2 = dlog.var() if E > 1 else 0.0
        s3, s4 = dlog.max(), (dlog**2).sum()
        
        chunks = np.array_split(dlog, min(8, E))
        contrast = [float(np.clip(c.sum(), -10, 10)) for c in chunks]
        s7 = np.mean(contrast) if contrast else 0.0
        
        w = np.exp(dlog - dlog.max())
        ws = w.sum()
        p = w / (ws + 1e-12) if ws > 0 else np.zeros_like(w)
        s8 = -np.sum(p * np.log(p + 1e-12))
        
        return torch.tensor([s1, s2, s3, s4, s5, s6, s7, s8], dtype=torch.float32)

class SignatureNormalizer:
    """Exponential moving average normalization for signatures."""
    
    def __init__(self, momentum: float = 0.99, eps: float = 1e-6):
        self.m = None
        self.v = None
        self.mom = momentum
        self.eps = eps
        
    def __call__(self, s: torch.Tensor) -> torch.Tensor:
        """Normalize signature using EMA statistics."""
        x = s.detach().to(torch.float32)
        
        if self.m is None:
            self.m = x.clone()
            self.v = torch.zeros_like(x)
        else:
            self.m = self.m * self.mom + x * (1 - self.mom)
            self.v = self.v * self.mom + (x - self.m)**2 * (1 - self.mom)
            
        return (x - self.m) / (torch.sqrt(self.v + self.eps))

class JacobianHypercube(nn.Module):
    """Neural network for hypercube control actions."""
    
    def __init__(self, in_dim: int = 8, out_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64), 
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize network weights."""
        with torch.no_grad():
            for m in self.net:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, a=np.sqrt(5))
                    nn.init.zeros_(m.bias)
                    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through network."""
        return self.net(x)
    
    @torch.no_grad()
    def act(self, s: torch.Tensor) -> Tuple[Dict[str, float], torch.Tensor]:
        """Compute control action and Jacobian."""
        device = next(self.parameters()).device
        s_vec = s.view(1, -1).to(device)
        
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type=="cuda")):
            u = self.net(s_vec).view(-1)
    
        action = {
            "delta_tau": float(torch.clamp(u[0], -0.02, 0.02)),
            "delta_vr":  float(torch.clamp(u[1], -1e-3, 1e-3)),
            "damping":   float(torch.clamp(u[2], 0.0, 1.0))
        }
    
        eps = 1e-4 if device.type == "cpu" else 1e-3  # Larger eps for CUDA under autocast
        eye = torch.eye(s_vec.numel(), device=device, dtype=s_vec.dtype)
        J = torch.zeros((3, s_vec.numel()), dtype=torch.float32, device=device)
        
        # Disable autocast for finite differences to maintain precision
        with torch.cuda.amp.autocast(enabled=False):
            for k in range(s_vec.numel()):
                sp = s_vec + eye[k] * eps
                up = self.net(sp.float()).view(-1)
                J[:, k] = (up - u.float()) / eps
            
        return action, J