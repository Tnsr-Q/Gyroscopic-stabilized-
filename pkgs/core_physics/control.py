"""
Control system components for signature normalization and Jacobian computation.

Contains SignatureNormalizer and JacobianHypercube classes for control 
operations and neural network-based Jacobian computation.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple

# Device and dtype from original constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32


class SignatureNormalizer:
    """Exponential moving average normalization for signatures."""
    
    def __init__(self, momentum: float = 0.99, eps: float = 1e-6):
        self.m, self.v, self.mom, self.eps = None, None, momentum, eps
        
    def __call__(self, s: torch.Tensor) -> torch.Tensor:
        """Normalize signature tensor using running statistics."""
        x = s.detach().to(torch.float32)
        if self.m is None: 
            self.m, self.v = x.clone(), torch.zeros_like(x)
        else: 
            self.m = self.m * self.mom + x * (1 - self.mom)
            self.v = self.v * self.mom + (x - self.m) ** 2 * (1 - self.mom)
        return (x - self.m) / (torch.sqrt(self.v + self.eps))


class JacobianHypercube(nn.Module):
    """Neural network for control actions with Jacobian computation."""
    
    def __init__(self, in_dim: int = 8, out_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), 
            nn.ReLU(), 
            nn.Linear(64, 64), 
            nn.ReLU(), 
            nn.Linear(64, out_dim))
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize network weights using Kaiming uniform."""
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
        """Compute control actions and Jacobian from signature."""
        s_vec = s.view(1, -1).to(DEVICE)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE=="cuda")):
            u = self.net(s_vec).view(-1)
    
        action = {
            "delta_tau": float(torch.clamp(u[0], -0.02, 0.02)),
            "delta_vr":  float(torch.clamp(u[1], -1e-3, 1e-3)),
            "damping":   float(torch.clamp(u[2], 0.0, 1.0))
        }
    
        eps = 1e-4 if DEVICE == "cpu" else 1e-3  # Larger eps for CUDA under autocast
        eye = torch.eye(s_vec.numel(), device=DEVICE, dtype=s_vec.dtype)
        J = torch.zeros((3, s_vec.numel()), dtype=torch.float32, device=DEVICE)
        
        # Disable autocast for finite differences to maintain precision
        with torch.cuda.amp.autocast(enabled=False):
            for k in range(s_vec.numel()):
                sp = s_vec + eye[k] * eps
                up = self.net(sp.float()).view(-1)
                J[:, k] = (up - u.float()) / eps
            
        return action, J