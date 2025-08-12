"""
Tensor network components for MERA spacetime and isometry operations.

Contains MERASpacetime class with drift guard functionality and utilities
for spectral dimension estimation from graph cuts.
"""
import torch
import numpy as np
import logging
from typing import List, Optional, Dict

# Device and dtype from original constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger = logging.getLogger('QuantumCore')


class MERASpacetime:
    """MERA (Multi-scale Entanglement Renormalization Ansatz) spacetime tensor network."""
    
    def __init__(self, layers: int = 6, bond_dim: int = 4):
        self.layers, self.bond_dim = layers, bond_dim
        self.tensors = self._initialize_tensors()
        self.isometries = self._initialize_isometries()
        self.target_dims = [bond_dim] * layers  # Track target bond dimensions
        
    def _initialize_tensors(self) -> List[torch.Tensor]:
        """Initialize MERA tensors with appropriate dimensions."""
        tensors = []
        for l in range(self.layers):
            dim = max(1, self.bond_dim * (2**(self.layers-l-1)))
            shape = (dim, dim, dim, dim)
            tensor = torch.randn(*shape, dtype=torch.complex64, device=DEVICE)
            tensor /= np.sqrt(dim)
            tensors.append(tensor)
        return tensors
        
    def _initialize_isometries(self) -> List[torch.Tensor]:
        """Initialize isometry tensors for each layer."""
        isos = []
        for l in range(self.layers):
            Din = max(1, self.bond_dim * (2**(self.layers-l-1)))
            Dout = max(1, (Din // 2)) if l < self.layers-1 else 1
            Q = torch.linalg.qr(torch.randn(Din, Dout, dtype=torch.complex64, device=DEVICE), mode='reduced')[0]
            isos.append(Q.conj().T)
        return isos
        
    def compute_entanglement_entropy(self, region_size: int) -> float:
        """Compute entanglement entropy for given region size."""
        return (1./3.) * np.log(region_size / 1.0 + 1e-10)
        
    def imprint_bond_phases(self, graph, edges, phi_map, layers=None):
        """Imprint phase information from bond graph onto MERA layers."""
        if not edges: 
            return
            
        if layers is None: 
            mid = self.layers // 2
            layers = [max(0, mid-1), mid, min(self.layers-1, mid+1)]
            
        weights = [0.25, 0.5, 0.25] if len(layers) == 3 else [1.0/len(layers)] * len(layers)
        φ = float(np.mean([phi_map.get(e, 0.0) for e in edges]))
        
        for l, w in zip(layers, weights): 
            T = self.tensors[l]
            phase = torch.exp(1j * torch.tensor(w * φ, device=T.device, dtype=T.dtype))
            self.tensors[l] = T * phase
            
    def minimal_cut_edges(self, graph):
        """Find minimal cut edges in the hypercube graph."""
        cut = []
        for e in graph.edges:
            if ((e.u>>0)&1) == 0 and ((e.v>>0)&1) == 1 and (e.v^e.u) == 1:
                cut.append(e)
        cut.sort(key=lambda e: (min(e.u, e.v), max(e.u, e.v)))
        return cut
        
    def snapshot_flat_baseline(self):
        """Snapshot current tensors as baseline for drift measurement."""
        self._baseline = [t.clone() for t in self.tensors]
        
    def delta_tensor_energy(self) -> float:
        """Compute tensor energy drift from baseline."""
        if not hasattr(self, "_baseline"): 
            return 0.0
        energy = 0.0
        for T0, T in zip(self._baseline, self.tensors):
            energy += (T - T0).abs().mean().item()
        return float(energy)
        
    def enforce_tensor_drift_guard(self, max_drift: float = 1e-3):
        """Reset global phase if drift exceeds threshold (drift guard)."""
        dE = self.delta_tensor_energy()
        if dE > max_drift:
            mid = self.layers // 2
            φ = -torch.angle(self.tensors[mid].flatten()[0])
            self.tensors[mid] = self.tensors[mid] * torch.exp(1j * φ)
            logger.info(f"Reset global phase in layer {mid} due to drift {dE:.2e} > {max_drift}")
            self.snapshot_flat_baseline()


def estimate_spectral_dim_from_cut(graph, cut_edges) -> Optional[float]:
    """Estimate spectral dimension d_H from interface graph Laplacian."""
    # Build interface graph Laplacian (nodes touched by cut)
    nodes = sorted({e.u for e in cut_edges} | {e.v for e in cut_edges})
    if len(nodes) < 4:
        return None
        
    index = {n: i for i, n in enumerate(nodes)}
    L = np.zeros((len(nodes), len(nodes)))
    for e in cut_edges:
        i, j = index[e.u], index[e.v]
        L[i, i] += 1
        L[j, j] += 1
        L[i, j] -= 1
        L[j, i] -= 1
        
    w, _ = np.linalg.eigh(L)
    
    # Filter near-zero eigenvalues
    eps = 1e-8
    evals = w[w > eps]
    if len(evals) < 4:
        return None
        
    # Histogram eigenvalue density
    bins = min(32, len(evals)//2)
    hist, bin_edges = np.histogram(evals, bins=bins, density=True)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    
    # Focus on low-energy spectrum (first 30%)
    low_energy_mask = bin_centers < np.percentile(bin_centers, 30)
    if np.sum(low_energy_mask) < 4:
        return None
        
    # Linear fit in log-log space: ρ(Δ) ~ Δ^{d_H/2 -1}
    log_centers = np.log(bin_centers[low_energy_mask])
    log_hist = np.log(np.maximum(hist[low_energy_mask], 1e-12))
    slope, _ = np.polyfit(log_centers, log_hist, 1)
    
    # Map slope to spectral dimension: d_H = 2*(slope + 1)
    d_H = 2 * (slope + 1)
    return float(d_H)


def leakage_ceiling(N: int, C: float = 91.64, b: float = 3.12, q: float = 2.0, m0: float = 1.0) -> float:
    """Calculate leakage bound ε_N ≲ C(log N)^b N^{-β} with β=m0*ln(q)."""
    beta = m0 * np.log(q)
    return C * (np.log(N + 1e-6) ** b) * (N ** (-beta))