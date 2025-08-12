"""Mathematical utilities and helper functions."""

import numpy as np
import torch
from typing import Optional, List, Tuple
import sys
import os
# Add the quantum-core directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from pkgs.core_physics.lattice import Edge, BondGraph

def _ddx(x: torch.Tensor) -> torch.Tensor:
    """Finite difference in x direction (last dimension)."""
    if x.dim() < 2:
        return torch.zeros_like(x)
    # Simple forward difference with zero padding
    dx = torch.zeros_like(x)
    dx[..., :-1] = x[..., 1:] - x[..., :-1]
    return dx

def _ddy(x: torch.Tensor) -> torch.Tensor:
    """Finite difference in y direction (second to last dimension)."""
    if x.dim() < 2:
        return torch.zeros_like(x)
    # Simple forward difference with zero padding
    dy = torch.zeros_like(x)
    dy[..., :-1, :] = x[..., 1:, :] - x[..., :-1, :]
    return dy

def _hypercube_bits(i: int) -> Tuple[int, int, int, int]:
    """Extract 4D hypercube bit representation."""
    return ((i>>0)&1, (i>>1)&1, (i>>2)&1, (i>>3)&1)

def _embed_4d_to_2d(bits: Tuple, H: int, W: int) -> Tuple[int, int]:
    """Embed 4D hypercube vertex to 2D grid coordinates."""
    return (
        min(H-1, bits[0]*(H//2) + bits[2]*(H//4)),
        min(W-1, bits[1]*(W//2) + bits[3]*(W//4))
    )

def estimate_spectral_dim_from_cut(graph: BondGraph, cut_edges: List[Edge]) -> Optional[float]:
    """Estimate spectral dimension d_H from interface graph Laplacian."""
    # Build interface graph Laplacian (nodes touched by cut)
    nodes = sorted({e.u for e in cut_edges} | {e.v for e in cut_edges})
    if len(nodes) < 4:
        return None
        
    index = {n:i for i,n in enumerate(nodes)}
    L = np.zeros((len(nodes), len(nodes)))
    for e in cut_edges:
        i, j = index[e.u], index[e.v]
        L[i,i] += 1
        L[j,j] += 1
        L[i,j] -= 1
        L[j,i] -= 1
        
    w, _ = np.linalg.eigh(L)
    
    # Filter near-zero eigenvalues
    eps = 1e-8
    evals = w[w > eps]
    if len(evals) < 4:
        return None
        
    # Histogram eigenvalue density
    bins = min(32, len(evals)//2)
    hist, bin_edges = np.histogram(evals, bins=bins, density=True)
    bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
    
    # Focus on low-energy spectrum (first 30%)
    low_energy_mask = bin_centers < np.percentile(bin_centers, 30)
    if np.sum(low_energy_mask) < 4:
        return None
        
    # Linear fit in log-log space: ρ(Δ) ~ Δ^{d_H/2 -1}
    log_centers = np.log(bin_centers[low_energy_mask])
    log_hist = np.log(np.maximum(hist[low_energy_mask], 1e-12))
    slope, _ = np.polyfit(log_centers, log_hist, 1)
    
    # Map slope to spectral dimension: d_H = 2*(slope + 1)
    d_H = 2*(slope + 1)
    return float(d_H)

def leakage_ceiling(N: int, C: float = 91.64, b: float = 3.12, q: float = 2.0, m0: float = 1.0) -> float:
    """Calculate leakage bound ε_N ≲ C(log N)^b N^{-β} with β=m0*ln(q)."""
    beta = m0 * np.log(q)
    log_term = np.log(N) if N > 1 else 0.0
    return float(C * (log_term**b) * (N**(-beta)))

def project_to_lightcone(dr: float, dt: float, c: float) -> float:
    """Project displacement to within light cone."""
    return float(np.clip(dr, -abs(c*dt), abs(c*dt)))