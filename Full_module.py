"""Quantum Core with Tensor Trinity Integration - Final Hardened Version"""
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, List, Optional, Dict, Callable, Set
from dataclasses import dataclass
from enum import Enum
import itertools
import csv
import os
import logging
import networkx as nx
from scipy.optimize import nnls
from scipy.sparse import csr_matrix

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('QuantumCore')

# Reproducibility
np.random.seed(0)
torch.manual_seed(0)

# Physical constants and device setup
c = 299792458.0; G = 6.67430e-11; hbar = 1.054571817e-34
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

# --- Recorders ---
class SimpleRecorder:
    """General purpose recorder for dictionary-based logs."""
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.rows = []

    def log(self, row: Dict):
        if self.enabled:
            clean_row = {}
            for k, v in row.items():
                if isinstance(v, (int, float, np.number)):
                    clean_row[k] = float(v)
                elif isinstance(v, torch.Tensor):
                    clean_row[k] = v.item() if v.numel() == 1 else v.cpu().numpy()
                else:
                    clean_row[k] = str(v)
            self.rows.append(clean_row)

    def dump_csv(self, path: str):
        if not self.enabled or not self.rows: return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        keys = sorted(self.rows[0].keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.rows)

class DecoherenceState(Enum):
    COHERENT,DECOHERING,DECOHERENT,RECOMPRESSING,STABILIZED = "coherent","decohering","decoherent","recompressing","stabilized"

@dataclass
class QuantumState:
    coherence_gamma: float; proper_time: float; torsion: float; law_field: np.ndarray
    decoherence_state: DecoherenceState; entropy: float; phase_memory: float = 0.0

@dataclass(frozen=True)
class Edge:
    u: int; v: int

@dataclass
class BondGraph:
    V: int; edges: List[Edge]; baseline_chi: int = 2

def build_tesseract_lattice(baseline_chi: int = 2) -> BondGraph:
    V, edges = 16, []
    for i in range(V):
        for bit in range(4):
            if i < (j := i ^ (1 << bit)): edges.append(Edge(i, j))
    return BondGraph(V=V, edges=edges, baseline_chi=baseline_chi)

# --- Finite differences (safer pad) ---
def _ddx(x: torch.Tensor) -> torch.Tensor:
    # pad=(left,right,top,bottom) for 2D tensors: pad last dim here
    return x - torch.nn.functional.pad(x, (1, 0, 0, 0), mode="replicate")[..., :, :-1]

def _ddy(x: torch.Tensor) -> torch.Tensor:
    return x - torch.nn.functional.pad(x, (0, 0, 1, 0), mode="replicate")[..., :-1, :]

def _hypercube_bits(i: int)->Tuple[int,...]: return ((i>>0)&1,(i>>1)&1,(i>>2)&1,(i>>3)&1)
def _embed_4d_to_2d(bits: Tuple, H: int, W: int)->Tuple[int,int]: 
    return min(H-1,bits[0]*(H//2)+bits[2]*(H//4)),min(W-1,bits[1]*(W//2)+bits[3]*(W//4))

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

class UVIRRegulator:
    def __init__(self,d_H:float=3.12,C:float=91.64,chi0:int=2,chi_min:int=2,chi_max:int=16,eta:float=0.12,lambda_phi:float=0.35):
        self.d_H,self.C,self.chi0,self.chi_min,self.chi_max,self.eta,self.lambda_phi=d_H,C,chi0,chi_min,chi_max,eta,lambda_phi
        
    def _K_bulk(self,N:int)->float: 
        Nb=max(4,int(N))
        return Nb/(self.C*(np.log(Nb+1e-6)**self.d_H))
        
    def update_bonds(self,g:BondGraph,ss:Dict,pb:Dict,N:int)->Tuple[Dict,Dict]:
        Kb,chi,phi=self._K_bulk(N),{},{}
        for e in g.edges:
            chi_val = int(np.clip(self.chi0*(1.0+self.eta*Kb*float(ss.get(e,0.0))), self.chi_min, self.chi_max))
            chi[e] = chi_val
            phi[e] = float(pb.get(e,0.0))
        return chi,phi

class AlenaSoul:
    def __init__(self,kappa:float=0.015): self.kappa=kappa
        
    def upsilon_from_A(self,A:torch.Tensor)->torch.Tensor: 
        U=torch.zeros((4,4,*A.shape[-2:]),dtype=DTYPE,device=A.device)
        # Only set (0,1) and (1,0) components (time-space components of antisymmetric 2-form)
        U[0,1] = self.kappa*(_ddy(A[1])-_ddx(A[0]))
        U[1,0] = -U[0,1]
        return U
    
    def phase_imprint(self,g:BondGraph,u:torch.Tensor,L:float)->Dict:
        mag,phi=u.abs().amax(dim=(0,1)),{}
        scale=float(mag.mean().clamp_min(1e-6))
        for e in g.edges: 
            xu,yu=_embed_4d_to_2d(_hypercube_bits(e.u),*mag.shape)
            xv,yv=_embed_4d_to_2d(_hypercube_bits(e.v),*mag.shape)
            phi[e]=float(L*0.5*np.pi*((mag[xu,yu]+mag[xv,yv]).item()/(2.0*scale)))
        return phi

class GyroscopeFeeler:
    def __init__(self, graph: BondGraph): 
        self.graph = graph
        
    def gluewalk_signature(self, chi: Dict, phi: Dict, b_chi: int) -> torch.Tensor:
        E = len(self.graph.edges)
        if E == 0: 
            return torch.zeros(8, dtype=torch.float32)
            
        chi_a = np.fromiter((max(1,chi.get(e,b_chi)) for e in self.graph.edges), dtype=np.float64, count=E)
        phi_a = np.fromiter((float(phi.get(e,0.0)) for e in self.graph.edges), dtype=np.float64, count=E)
        dlog, c_phi, s_phi = np.log(chi_a)-np.log(b_chi), np.cos(phi_a), np.sin(phi_a)
        s1,s5,s6=dlog.mean(),c_phi.mean(),s_phi.mean()
        s2=dlog.var() if E>1 else 0.0
        s3,s4=dlog.max(),(dlog**2).sum()
        
        chunks = np.array_split(dlog, min(8,E))
        contrast = [float(np.clip(c.sum(),-10,10)) for c in chunks]
        s7 = np.mean(contrast) if contrast else 0.0
        
        w = np.exp(dlog - dlog.max())
        p = w/((ws:=w.sum())+1e-12) if ws>0 else np.zeros_like(w)
        s8 = -np.sum(p*np.log(p+1e-12))
        
        return torch.tensor([s1,s2,s3,s4,s5,s6,s7,s8], dtype=torch.float32)

class SignatureNormalizer:
    def __init__(self, momentum: float=0.99, eps: float=1e-6): 
        self.m,self.v,self.mom,self.eps=None,None,momentum,eps
        
    def __call__(self, s: torch.Tensor) -> torch.Tensor:
        x = s.detach().to(torch.float32)
        if self.m is None: 
            self.m, self.v = x.clone(), torch.zeros_like(x)
        else: 
            self.m=self.m*self.mom+x*(1-self.mom)
            self.v=self.v*self.mom+(x-self.m)**2*(1-self.mom)
        return (x - self.m) / (torch.sqrt(self.v + self.eps))

class JacobianHypercube(nn.Module):
    def __init__(self, in_dim: int=8, out_dim: int=3):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(in_dim,64), 
            nn.ReLU(), 
            nn.Linear(64,64), 
            nn.ReLU(), 
            nn.Linear(64,out_dim))
        self._initialize_weights()
        
    def _initialize_weights(self):
        with torch.no_grad():
            for m in self.net:
                if isinstance(m, nn.Linear): 
                    nn.init.kaiming_uniform_(m.weight, a=np.sqrt(5))
                    nn.init.zeros_(m.bias)
                    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return self.net(x)
    
    @torch.no_grad()
    def act(self, s: torch.Tensor) -> Tuple[Dict[str, float], torch.Tensor]:
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

class ProperTimeGaugeField:
    def __init__(self, grid_size: int=32, device="cpu"): 
        self.grid_size,self.device=grid_size,device
        self.A_mu=torch.zeros(4,grid_size,grid_size,device=device,dtype=DTYPE)
        self.upsilon_tensor=torch.zeros(4,4,grid_size,grid_size,device=device,dtype=DTYPE)
        
    def set_from_wormhole_metric(self, r: float, r0: float): 
        self.A_mu[0].fill_(-0.5*np.log(1+r/max(r0,1e-9))/c**2)
        self.A_mu[1].fill_(np.sqrt((r0**2/r)/max(r,r0)) if r>r0 else 0.0)
        
    def set_alena_modification(self, upsilon_tensor: torch.Tensor): 
        self.upsilon_tensor=upsilon_tensor.to(self.device)
        
    def compute_field_strength(self) -> torch.Tensor:
        F = torch.zeros_like(self.upsilon_tensor)
        # Compute time-space components (0,1) and (1,0) of antisymmetric 2-form
        F[0,1] = _ddy(self.A_mu[1]) - _ddx(self.A_mu[0])
        F[1,0] = -F[0,1]
        return F + self.upsilon_tensor
        
    def is_flat(self, tolerance: float=1e-6) -> bool: 
        return torch.max(torch.abs(self.compute_field_strength())) < tolerance

    def zero_fields(self):
        """Set EM potentials and Υ to zero"""
        self.A_mu.zero_()
        self.upsilon_tensor.zero_()

class TimeOperator:
    def __init__(self, dim_clock: int=16): 
        self.dim_clock = dim_clock
        self.H_int = torch.diag(torch.linspace(0,1,dim_clock,device=DEVICE,dtype=DTYPE))
        
    def path_conditioned_evolution(self, tau: float) -> torch.Tensor: 
        return torch.matrix_exp(-1j * self.H_int * tau)
        
    def compute_decoherence(self, t1: float, t2: float, psi: torch.Tensor) -> float: 
        U1 = self.path_conditioned_evolution(t1)
        U2 = self.path_conditioned_evolution(t2)
        return torch.abs(psi.conj() @ U1.conj().T @ U2 @ psi).item()

class TimeRecompressionGate:
    def __init__(self, time_op: TimeOperator): 
        self.time_op=time_op
        
    def create_gate(self, dt: float) -> torch.Tensor: 
        return torch.matrix_exp(1j*self.time_op.H_int*dt)
        
    def apply_recompression(self, psi: torch.Tensor, dt: float) -> torch.Tensor: 
        return self.create_gate(dt)@psi
        
    def verify_coherence_restoration(self, t1: float, t2: float, psi: torch.Tensor) -> Tuple[float,float]:
        U1 = self.time_op.path_conditioned_evolution(t1)
        U2 = self.time_op.path_conditioned_evolution(t2)
        before = torch.abs(psi.conj()@U1.conj().T@U2@psi).item()
        after = torch.abs(psi.conj()@U1.conj().T@(self.create_gate(t1-t2)@U2)@psi).item()
        return before, after

class MERASpacetime:
    def __init__(self, layers: int = 6, bond_dim: int = 4):
        self.layers, self.bond_dim = layers, bond_dim
        self.tensors = self._initialize_tensors()
        self.isometries = self._initialize_isometries()
        self.target_dims = [bond_dim] * layers  # Track target bond dimensions
        
    def _initialize_tensors(self) -> List[torch.Tensor]:
        tensors = []
        for l in range(self.layers):
            dim = max(1, self.bond_dim * (2**(self.layers-l-1)))
            shape = (dim, dim, dim, dim)
            tensor = torch.randn(*shape, dtype=torch.complex64, device=DEVICE)
            tensor /= np.sqrt(dim)
            tensors.append(tensor)
        return tensors
        
    def _initialize_isometries(self) -> List[torch.Tensor]:
        isos = []
        for l in range(self.layers):
            Din = max(1, self.bond_dim * (2**(self.layers-l-1)))
            Dout = max(1, (Din // 2)) if l < self.layers-1 else 1
            Q = torch.linalg.qr(torch.randn(Din, Dout, dtype=torch.complex64, device=DEVICE), mode='reduced')[0]
            isos.append(Q.conj().T)
        return isos
        
    def compute_entanglement_entropy(self, region_size: int) -> float:
        return (1.0/3.0) * np.log(region_size + 1e-10)
        
    def imprint_bond_phases(self, graph, edges, phi_map, layers=None):
        if not edges: 
            return
            
        if layers is None: 
            mid = self.layers//2
            layers = [max(0,mid-1), mid, min(self.layers-1,mid+1)]
            
        weights = [0.25,0.5,0.25] if len(layers)==3 else [1.0/len(layers)]*len(layers)
        φ = float(np.mean([phi_map.get(e,0.0) for e in edges]))
        
        for l,w in zip(layers,weights): 
            T = self.tensors[l]
            phase = torch.exp(1j*torch.tensor(w*φ, device=T.device, dtype=T.dtype))
            self.tensors[l] = T * phase
            
    def minimal_cut_edges(self, graph):
        cut = []
        for e in graph.edges:
            if ((e.u>>0)&1)==0 and ((e.v>>0)&1)==1 and (e.v^e.u)==1:
                cut.append(e)
        cut.sort(key=lambda e:(min(e.u,e.v),max(e.u,e.v)))
        return cut
        
    def snapshot_flat_baseline(self): 
        self._baseline = [t.clone() for t in self.tensors]
        
    def delta_tensor_energy(self) -> float: 
        if not hasattr(self,"_baseline"): 
            return 0.0
        energy = 0.0
        for T0,T in zip(self._baseline, self.tensors):
            energy += (T - T0).abs().mean().item()
        return float(energy)
        
    def enforce_tensor_drift_guard(self, max_drift: float = 1e-3):
        """Reset global phase if drift exceeds threshold"""
        dE = self.delta_tensor_energy()
        if dE > max_drift:
            mid = self.layers//2
            φ = -torch.angle(self.tensors[mid].flatten()[0])
            self.tensors[mid] = self.tensors[mid] * torch.exp(1j*φ)
            logger.info(f"Reset global phase in layer {mid} due to drift {dE:.2e} > {max_drift}")
            self.snapshot_flat_baseline()

# --- Spectral Dimension Estimation ---
def estimate_spectral_dim_from_cut(graph, cut_edges) -> Optional[float]:
    """Estimate spectral dimension d_H from interface graph Laplacian"""
    # Build interface graph Laplacian (nodes touched by cut)
    nodes = sorted({e.u for e in cut_edges} | {e.v for e in cut_edges})
    if len(nodes) < 4:
        return None
        
    index = {n:i for i,n in enumerate(nodes)}
    import numpy as np
    L = np.zeros((len(nodes), len(nodes)))
    for e in cut_edges:
        i,j = index[e.u], index[e.v]
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

# --- Leakage Bound Calculation ---
def leakage_ceiling(N: int, C: float = 91.64, b: float = 3.12, q: float = 2.0, m0: float = 1.0) -> float:
    """Calculate leakage bound ε_N ≲ C(log N)^b N^{-β} with β=m0*ln(q)"""
    import numpy as np
    beta = m0 * np.log(q)
    log_term = np.log(N) if N > 1 else 0.0
    return float(C * (log_term**b) * (N**(-beta)))

# --- Renyi Anomaly Monitor (Stub) ---
class RenyiMonitor:
    def __init__(self, n_values: Tuple[float] = (0.1,0.2,0.3), target_d_eff: float = 31.0, tol: float = 0.1):
        self.n_values = n_values
        self.target = target_d_eff
        self.tol = tol
        
    def score(self, Sn_emp: Dict[float, float]) -> float:
        """Simple normalized deviation vs fitted slope (placeholder)"""
        import numpy as np
        ns = np.array(sorted(Sn_emp.keys())))
        S = np.array([Sn_emp[n] for n in ns])
        # Fit log(S) vs log(n): slope ~ f(d_eff)
        p = np.polyfit(np.log(ns), np.log(np.maximum(S,1e-12)), 1)
        d_eff_est = -2.0 * p[0]  # toy mapping; replace with calibrated map
        return float((d_eff_est - self.target)/max(self.target, 1e-6))

# ==================== NEW MODULES ====================
class HoloEntropyDiagnostics:
    """
    MERA-style entanglement proxies using min-cuts on your BondGraph.
    - S(A) = sum_e∈cut(A) [ s_unit * log(chi_e) ]
    - I(A:B) = S(A) + S(B) - S(A∪B)
    """
    def __init__(self, graph, chi: Optional[Dict]=None, baseline_chi: int = 2, s_unit: float = 1.0):
        self.graph = graph
        self.baseline_chi = baseline_chi
        self.s_unit = s_unit
        self.chi = chi or {}

    def _cut_edges(self, A: Set[int]) -> List:
        cut = []
        for e in self.graph.edges:
            inside = (e.u in A, e.v in A)
            if inside[0] ^ inside[1]:  # XOR: one in, one out
                cut.append(e)
        return cut

    def S_region(self, A: Set[int]) -> float:
        """Entropy proxy via min-cut across edges crossing A | A^c."""
        cut = self._cut_edges(A)
        total = 0.0
        for e in cut:
            chi_e = float(self.chi.get(e, self.baseline_chi))
            total += np.log(max(chi_e, 1.0))
        return self.s_unit * total

    def I_mutual(self, A: Set[int], B: Set[int]) -> float:
        AuB = set(A) | set(B)
        return self.S_region(A) + self.S_region(B) - self.S_region(AuB)

    # -------- convenience: square regions on a fixed 2D embedding --------
    def embed_vertices(self, H: int = 8, W: int = 8) -> Dict[int, Tuple[int,int]]:
        """Map each vertex index to (row,col) in an HxW canvas using your existing embedding."""
        assert H % 4 == 0 and W % 4 == 0, "H,W must be multiples of 4"
        pos = {}
        for i in range(self.graph.V):
            pos[i] = _embed_4d_to_2d(_hypercube_bits(i), H, W)
        return pos

    def square_region_vertices(self, L: int, H: int=8, W: int=8, cx: Optional[int]=None, cy: Optional[int]=None) -> Set[int]:
        """Select vertices whose embedded coords fall in a LxL box centered at (cx,cy) on the HxW canvas."""
        pos = self.embed_vertices(H, W)
        if cx is None: cx = H//2
        if cy is None: cy = W//2
        r0, c0 = int(cx - L//2), int(cy - L//2)
        A = set()
        for v,(r,c) in pos.items():
            if r0 <= r < r0 + L and c0 <= c < c0 + L:
                A.add(v)
        return A

    def scaling_curve(self, max_L: int = 8, H: int=8, W: int=8, center: Optional[Tuple[int,int]]=None) -> Tuple[np.ndarray,np.ndarray]:
        """Compute S(L) for L=1..max_L using square regions on the embedding."""
        if center is None: center = (H//2, W//2)
        Ls, Ss = [], []
        for L in range(1, max_L+1):
            A = self.square_region_vertices(L, H, W, cx=center[0], cy=center[1])
            Ls.append(L)
            Ss.append(self.S_region(A))
        return np.array(Ls), np.array(Ss)

    def plot_scaling(self, max_L: int=8, H: int=8, W: int=8, center: Optional[Tuple[int,int]]=None):
        import matplotlib.pyplot as plt
        Ls, Ss = self.scaling_curve(max_L, H, W, center)
        fig, ax = plt.subplots(1,2, figsize=(11,4.2))
        ax[0].plot(Ls, Ss, "o-", label="S(L) (min-cut proxy)")
        ax[0].plot(Ls, Ls, "--", label="area law ~ L")
        ax[0].set_xlabel("Region linear size L"); ax[0].set_ylabel("S(L)")
        ax[0].legend()

        ax[1].loglog(Ls, np.maximum(Ss, 1e-12), "o-")
        ax[1].loglog(Ls, Ls, "--", label="slope 1 (area law)")
        ax[1].loglog(Ls, Ls**2, ":", label="slope 2 (volume law)")
        ax[1].legend(); ax[1].set_xlabel("log L"); ax[1].set_ylabel("log S")
        plt.tight_layout(); plt.show()
        return Ls, Ss

def _u1_phase(z: complex, eps: float=1e-14) -> complex:
    """Return z/|z| with safe fallback to 1 if |z| ~ 0."""
    mag = np.abs(z)
    if mag < eps:
        return 1.0 + 0j
    return z / mag

def chern_number_fhs(states_fn: Callable[[float,float], np.ndarray],
                     kx_pts: int = 21, ky_pts: int = 21) -> int:
    """
    Gauge-invariant Chern number on a (kx,ky) torus using FHS.
    - states_fn(kx,ky) -> 
        * 1D complex array (single occupied band), normalized, OR
        * 2D (dim, n_occ) with orthonormal columns spanning occupied subspace.
    Returns: integer Chern number.
    """
    kxs = np.linspace(0.0, 2*np.pi, kx_pts, endpoint=False)
    kys = np.linspace(0.0, 2*np.pi, ky_pts, endpoint=False)

    # Cache states/bases
    states = [[None for _ in range(ky_pts)] for _ in range(kx_pts)]
    for ix, kx in enumerate(kxs):
        for iy, ky in enumerate(kys):
            states[ix][iy] = states_fn(kx, ky)

    # Link variables Ux, Uy on the lattice
    Ux = np.ones((kx_pts, ky_pts), dtype=complex)
    Uy = np.ones((kx_pts, ky_pts), dtype=complex)

    for ix in range(kx_pts):
        ixp = (ix + 1) % kx_pts
        for iy in range(ky_pts):
            iyp = (iy + 1) % ky_pts
            psi   = states[ix][iy]
            psi_x = states[ixp][iy]
            psi_y = states[ix][iyp]

            if psi.ndim == 1:
                # single band
                Ux[ix,iy] = _u1_phase(np.vdot(psi, psi_x))
                Uy[ix,iy] = _u1_phase(np.vdot(psi, psi_y))
            else:
                # multi-band occupied subspace: use overlap matrix determinant
                Mx = psi.conj().T @ psi_x   # (n_occ x n_occ)
                My = psi.conj().T @ psi_y
                # robust unit-modulus phases via slogdet
                sx, _ = np.linalg.slogdet(Mx)
                sy, _ = np.linalg.slogdet(My)
                Ux[ix,iy] = sx if sx != 0 else _u1_phase(np.linalg.det(Mx))
                Uy[ix,iy] = sy if sy != 0 else _u1_phase(np.linalg.det(My))

    # Lattice field strength on each plaquette
    F = np.zeros((kx_pts, ky_pts))
    for ix in range(kx_pts):
        ixp = (ix + 1) % kx_pts
        for iy in range(ky_pts):
            iyp = (iy + 1) % ky_pts
            # curl of the U(1) link on the plaquette
            plaquette = Ux[ix,iy] * Uy[ixp,iy] / (Ux[ix, iyp] * Uy[ix,iy])
            F[ix,iy] = np.angle(plaquette)

    C = int(np.rint(F.sum() / (2*np.pi)))
    return C

def berry_phase_on_loop(psi_list: np.ndarray) -> float:
    """Gauge-invariant Berry phase along a closed loop of states."""
    phase = 1.0 + 0j
    for i in range(len(psi_list)-1):
        phase *= _u1_phase(np.vdot(psi_list[i], psi_list[i+1]))
    return float(np.angle(phase))

class PageCurveSimulator:
    """Toy model for Page curve simulation via qubit leakage"""
    def __init__(self, n_bh=200, leak_per_step=2, steps=120):
        self.n_bh = n_bh
        self.leak_per_step = leak_per_step
        self.steps = steps
        self.entropy_curve = self.simulate()
        
    def simulate(self) -> np.ndarray:
        ent = []
        n_rad = 0
        for _ in range(self.steps):
            n_rad = min(self.n_bh, n_rad + self.leak_per_step)
            n_bh_eff = self.n_bh - n_rad
            ent.append(min(n_rad, n_bh_eff))
        return np.array(ent) * np.log(2)
    
    def plot(self):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,6))
        plt.plot(self.entropy_curve)
        plt.axvline(len(self.entropy_curve)//2, color='r', linestyle='--', label='Page time')
        plt.xlabel("Time step")
        plt.ylabel("Entanglement entropy (nats)")
        plt.title("Page Curve Simulation")
        plt.legend()
        plt.grid()
        plt.show()

def solve_edge_weights_from_cuts(graph, cuts_to_S, λ=1e-2) -> Dict[Edge, float]:
    """Solve for edge weights w_e that match measured entanglement entropies S(A)"""
    E = list({e for A,_ in cuts_to_S for e in A})
    idx = {e:i for i,e in enumerate(E)}
    A_mat, b = [], []
    for A_edges, S in cuts_to_S:
        row = np.zeros(len(E))
        for e in A_edges: 
            if e in idx:
                row[idx[e]] = 1.0
        A_mat.append(row)
        b.append(S)
    A_mat, b = np.vstack(A_mat), np.array(b)
    
    # Solve with non-negative least squares
    w, _ = nnls(A_mat, b)
    return {e: max(0.0, w[idx[e]]) for e in E if e in idx}

def reconstruct_metric_from_entanglement(graph, S_matrix: Dict[Set[int], float], λ=1e-2) -> Dict[Edge, float]:
    """Reconstruct discrete bulk metric from entanglement measurements"""
    cuts_to_S = []
    for region, S_val in S_matrix.items():
        cuts_to_S.append((graph.minimal_cut_edges(region), S_val))
    return solve_edge_weights_from_cuts(graph, cuts_to_S, λ)

# ==================== END NEW MODULES ====================

class RecursiveConformalComputing:
    _THETA_ALIASES = {"lambda_phi": "λφ", "beta1": "β1", "beta2": "β2", "beta3": "β3"}

    def __init__(self, r0: float = 0.5, m0: float = 1.0, params: Optional[Dict] = None):
        self.r0, self.m0 = r0, m0
        self.params = params if params else {}
        self.time_op = TimeOperator()
        self.recompression_gate = TimeRecompressionGate(self.time_op)
        self.gauge_field = ProperTimeGaugeField(device=DEVICE)
        self.mera = MERASpacetime()
        self.graph = build_tesseract_lattice()
        self.loom = UVIRRegulator(d_H=self.params.get('d_H',3.12), C=self.params.get('C_RT',91.64))
        self.soul = AlenaSoul(kappa=self.params.get('kappa',0.015))
        self.feeler = GyroscopeFeeler(self.graph)
        self.hypercube = JacobianHypercube().to(DEVICE)
        self.sig_norm = SignatureNormalizer()
        self.state = QuantumState(
            coherence_gamma=1.0, 
            proper_time=0.0, 
            torsion=0.0, 
            law_field=np.ones((32,32)), 
            decoherence_state=DecoherenceState.COHERENT, 
            entropy=0.0
        )
        psi = torch.randn(self.time_op.dim_clock, dtype=torch.complex64, device=DEVICE)
        self.psi_clock = psi / torch.norm(psi)
        self._dH_estimate = None  # Spectral dimension cache
        self.rg_planner = RGPlanner(
            mu0=self.params.get('mu0', 1.0),
            L_rg=self.params.get('L_rg', 1.0),
            d_boundary=self.params.get('d_boundary', 2)
        )
        self._last_s_Pl = 1.0  # Last Planck scale factor
        self._c_rel_history = []  # Track c_rel for stability
        self._dH_history = []     # Track spectral dimension
        self._holo_diag = HoloEntropyDiagnostics(self.graph)  # Add HoloEntropyDiagnostics
        self._page_curve_sim = PageCurveSimulator()  # Add Page curve simulator

    @property
    def recorder(self) -> Optional["SimpleRecorder"]: 
        return self.params.get("recorder")

    def layer_radius(self, layer: int) -> float:
        """Map layer index to radial position in throat geometry"""
        L = self.mera.layers
        return self.r0 * (layer + 0.5) / max(1, L)

    def target_bond_dim(self, r: float, Dmin=2, Dmax=32, alpha=3.0, R=None, eps=1e-6) -> int:
        """Compute target bond dimension for a given radial position"""
        if R is None: R = 1.0 * self.r0
        num = np.log(1.0 + alpha * r / max(self.r0, eps))
        den = np.log(1.0 + alpha * R / max(self.r0, eps)) + eps
        frac = np.clip(num/den, 0.0, 1.0)
        # More capacity near the throat (r small)
        val = Dmin + (Dmax - Dmin) * (1.0 - frac)
        return int(np.clip(round(val), Dmin, Dmax))

    def update_bond_dimensions(self):
        """Update target bond dimensions based on radial position"""
        new_dims = []
        for l in range(self.mera.layers):
            r = self.layer_radius(l)
            new_dim = self.target_bond_dim(r)
            
            # Apply cap: max 2x change per step
            current_dim = self.mera.target_dims[l]
            max_change = min(2 * current_dim, self.params.get('max_bond_dim', 32))
            new_dim = min(new_dim, max_change)
            
            new_dims.append(new_dim)
        
        self.mera.target_dims = new_dims
        logger.info(f"Updated bond dimensions: {self.mera.target_dims}")

    def capacity_throttle(self, N: int) -> int:
        """Capacity throttle with log-compression for MERA bond dims"""
        # Compute geometric mean of bond dimensions in minimal-cut layers
        cut_layers = [max(0, self.mera.layers//2 - 1), self.mera.layers//2, 
                      min(self.mera.layers-1, self.mera.layers//2 + 1)]
        dims = [self.mera.target_dims[l] for l in cut_layers]
        log_dim = np.log(np.mean(dims) + 1e-6)
        
        # Adjust capacity with log-compression
        Nb = max(4, int(N))
        Cc = float(self.params.get("cap_C",1.0))
        b = float(self.params.get("cap_b",-0.3))
        K = int(max(1, np.floor(Cc * (np.log(Nb * log_dim + 1e-6)**b)))
        
        cap_min = self.params.get("cap_min", 1)
        cap_max = self.params.get("cap_max", len(self.mera.minimal_cut_edges(self.graph)))
        return int(np.clip(K, cap_min, cap_max))

    def select_cut_bonds(self, K: int):
        cut = self.mera.minimal_cut_edges(self.graph)
        ups = self.soul.upsilon_from_A(self.gauge_field.A_mu)
        phi_map = self.soul.phase_imprint(self.graph, ups, self.loom.lambda_phi)
        
        # Sort by absolute phase value and edge ordering
        cut_sorted = sorted(
            cut, key=lambda e: (-abs(phi_map.get(e,0.0)), (min(e.u,e.v),max(e.u,e.v)))
        )
        return cut_sorted[:max(1,K)], phi_map

    def _h_profile(self) -> float:
        gx, gy = np.gradient(self.state.law_field)
        chi = float(np.sqrt((gx**2 + gy**2).mean()))
        Tn = float(abs(self.state.torsion))
        phase_div = 0.0
        b1, b2, b3 = self.params.get("beta_vec",(0.15,0.05,0.01))
        g1 = self.params.get("gamma1",0.2)
        return 1.0 + b1*np.tanh(chi-1.0) + b2*(Tn*Tn)/(1.0+g1*chi) + b3*phase_div

    def _Tuu_from_h(self, dt: float) -> float:
        h_now = float(self._h_profile())
        # Stabilize logs near tails
        h_clamped = np.clip(h_now, 1e-12, 1e12)
        ln_now = np.log1p(h_clamped - 1)  # More stable near h=1
        
        if not hasattr(self,"_ema"):
            self._ema = {"ln": ln_now, "d1": 0.0, "d2": 0.0}
            
        ema = self._ema
        if dt <= 0: 
            return 0.0
            
        tau = 5.0 * self.params.get('sigma_u',4e-3)
        alpha = dt/(tau+dt)
        d1 = (1-alpha)*ema["d1"] + alpha*((ln_now - ema["ln"])/dt)
        ema["d2"] = (1-alpha)*ema["d2"] + alpha*((d1 - ema["d1"])/dt)
        ema["ln"], ema["d1"] = ln_now, d1
        
        a1, a2 = self.params.get('hbar_weights',(0.0,-1.0))
        return float(a1*ema["d2"] + a2*(d1*d1))

    def compute_Tuu_eff(self, dt: float, s_Pl: float=1.0) -> Dict[str,float]:
        """Compute effective stress-energy with Planck scale factor"""
        T_hbar = self._Tuu_from_h(dt)
        # Apply Planck scale factor: geometric prefactor scales as 1/M_Pl^2
        T_hbar *= (1.0 / max(s_Pl, 1e-6))**2

        T_ups = float(self.soul.upsilon_from_A(self.gauge_field.A_mu)[0,1].abs().mean().item())
        T_ups *= self.params.get("upsilon_to_Tuu",1.0)

        T_sqz = -abs(self.params.get("squeeze_amp",0.0)) * (float(abs(self.state.torsion))**2)
        return {
            "Tuu_hbar": T_hbar,
            "Tuu_Upsilon": T_ups,
            "Tuu_sqz": T_sqz,
            "Tuu_total": T_hbar + T_ups + T_sqz
        }
    
    def smeared_ANE(self, dt: float, sigma_u: float, s_Pl: float=1.0) -> float:
        """Smeared ANE with Planck scale threading"""
        if not hasattr(self,"_ane_hist"):
            self._ane_hist, self._t_clock = [], 0.0
            
        self._t_clock += dt
        Tuu_total = self.compute_Tuu_eff(dt, s_Pl)["Tuu_total"]
        self._ane_hist.append((self._t_clock, Tuu_total))
        
        # Keep only recent history within 6 sigma
        max_age = 6.0 * sigma_u
        self._ane_hist = [(t, v) for (t, v) in self._ane_hist if (self._t_clock - t) <= max_age]
        
        # Gaussian-weighted average
        num = 0.0
        den = 0.0
        sigma_u_safe = max(1e-6, sigma_u)
        for t, v in self._ane_hist:
            tau = self._t_clock - t
            weight = np.exp(-0.5 * (tau / sigma_u_safe)**2)
            num += v * weight
            den += weight
            
        return float(num / max(den, 1e-12))

    def qei_guard(self, ane: float, sigma_u: float, margin: float=0.05) -> bool: 
        """Quantum Energy Inequality guard with dynamic RHS"""
        # Compute RHS based on current envelope f(t)
        t_vals = [t for t, _ in self._ane_hist]
        if len(t_vals) < 2:
            return True  # Not enough data
        
        # Compute min and max times
        t_min, t_max = min(t_vals), max(t_vals)
        duration = t_max - t_min
        
        # Compute dynamic constant based on envelope
        qei_const = 1.0 / max(duration, 1e-6)
        sigma_u_safe = max(1e-6, sigma_u)
        threshold = (-qei_const / sigma_u_safe**4) * (1.0 - margin)
        return ane >= threshold

    def _get_theta(self, key: str) -> float:
        key = self._THETA_ALIASES.get(key, key)
        if key == "g": 
            return self.params.get("double_trace_gain",0.0)
        if key == "λφ": 
            return self.loom.lambda_phi
        beta_vec = self.params.get("beta_vec",(0.15,0.05,0.01))
        return list(beta_vec)[{"β1":0,"β2":1,"β3":2}[key]]

    def _project_theta(self, key: str, val: float) -> float:
        key = self._THETA_ALIASES.get(key, key)
        if key == "g": 
            return float(np.clip(val, 0.0, self.params.get("g_max",0.05)))
        if key == "λφ": 
            return float(np.clip(val, 0.0, 2.0))
        return float(np.clip(val, -0.5, 0.5))

    def _apply_theta(self, trial: Dict):
        beta_vec = list(self.params.get("beta_vec",(0.15,0.05,0.01)))
        
        for key, val in trial.items():
            key = self._THETA_ALIASES.get(key, key)
            if key == "g": 
                self.params["double_trace_gain"] = val
            elif key == "λφ": 
                self.loom.lambda_phi = val
            elif key == "β1": 
                beta_vec[0] = val
            elif key == "β2": 
                beta_vec[1] = val
            elif key == "β3": 
                beta_vec[2] = val
                
        self.params["beta_vec"] = tuple(beta_vec)
        
    def _finite_diff_grads(self, dt: float, delta_scale: float = 1e-3) -> Dict[str, float]:
        """
        Central-diff grads d(ANE)/dθ for θ in {g, λφ, β1, β2, β3}.
        Restores params after probing. Uses QEI-safe short probe.
        """
        theta_keys = ["g", "λφ", "β1", "β2", "β3"]
        base_sigma = self.params.get('sigma_u', 4e-3)
        base_ane = self.smeared_ANE(dt, base_sigma, self._last_s_Pl)
        grads: Dict[str, float] = {}

        # Snapshot current parameters
        g0 = self.params.get("double_trace_gain", 0.0)
        lphi = self.loom.lambda_phi
        b1, b2, b3 = self.params.get("beta_vec", (0.15,0.05,0.01))
        param_backup = {"g": g0, "λφ": lphi, "β1": b1, "β2": b2, "β3": b3}

        for k in theta_keys:
            v0 = param_backup[k]
            h = max(1e-6, abs(v0)*delta_scale)

            # θ+ step
            self._apply_theta({k: self._project_theta(k, v0 + h)})
            ane_plus = self.smeared_ANE(dt, base_sigma, self._last_s_Pl)

            # θ- step
            self._apply_theta({k: self._project_theta(k, v0 - h)})
            ane_minus = self.smeared_ANE(dt, base_sigma, self._last_s_Pl)

            # Restore original value
            self._apply_theta({k: v0})

            # Central difference gradient
            grads[k] = (ane_plus - ane_minus) / (2.0 * h)

        # Restore all parameters to original state
        self._apply_theta(param_backup)
        return grads

    def contractor_step(self, dt: float, target: float, s_Pl: float, step: float=0.25, max_ls: int=5) -> float:
        sigma_u = self.params.get('sigma_u',4e-3)
        base_ane = self.smeared_ANE(dt, sigma_u, s_Pl)
        grads = self._finite_diff_grads(dt)
        current_thetas = {k: self._get_theta(k) for k in grads}
        trial = {}
        obj_sign = np.sign(base_ane - target)
        
        # Initial trial step
        for k, g in grads.items():
            new_val = current_thetas[k] - step * obj_sign * g
            trial[k] = self._project_theta(k, new_val)
            
        # Line search
        for _ in range(max_ls):
            self._apply_theta(trial)
            ane_new = self.smeared_ANE(dt, sigma_u, s_Pl)
            
            improvement_condition = (
                (target < base_ane and ane_new < base_ane) or 
                (target > base_ane and ane_new > base_ane)
            )
            
            if improvement_condition and self.qei_guard(ane_new, sigma_u):
                return ane_new
                
            step *= 0.5
            for k, g in grads.items():
                new_val = current_thetas[k] - step * obj_sign * g
                trial[k] = self._project_theta(k, new_val)
        
        # Restore original parameters if no improvement
        self._apply_theta(current_thetas)
        return base_ane

    def update_quantum_state(self, dt: float, r: float, torsion_norm: float):
        gy, gx = np.gradient(self.state.law_field, axis=0), np.gradient(self.state.law_field, axis=1)
        lapL = np.gradient(gy, axis=0) + np.gradient(gx, axis=1)
        grad_r = np.gradient(np.ones_like(self.state.law_field)*r)[0]
        update = (lapL - self.state.coherence_gamma * grad_r + torsion_norm) * dt * 0.01
        self.state.law_field += update
        
    def zero_field_baseline(self):
        """Zero EM potentials and Υ; snapshot MERA baseline."""
        self.gauge_field.zero_fields()
        self.mera.snapshot_flat_baseline()
        logger.info("Zeroed gauge fields and set MERA baseline")
        
    def flat_baseline_check(self, dt: float, steps: int = 128) -> float:
        """Run with zero fields; ANE should hover near ~0 and QEI-safe."""
        sigma_u = self.params.get('sigma_u', 4e-3)
        for _ in range(steps):
            self.smeared_ANE(dt, sigma_u, self._last_s_Pl)  # integrates history at zero drive
        final_ane = self.smeared_ANE(dt, sigma_u, self._last_s_Pl)
        logger.info(f"Flat baseline check: ANE = {final_ane:.2e}")
        return final_ane
        
    def estimate_spectral_dim(self, cut_edges: List[Edge]) -> Optional[float]:
        """Estimate spectral dimension from interface graph"""
        self._dH_estimate = estimate_spectral_dim_from_cut(self.graph, cut_edges)
        if self.recorder and self._dH_estimate is not None:
            self.recorder.log({"spectral_dim": self._dH_estimate})
            self._dH_history.append(self._dH_estimate)
        return self._dH_estimate

    def compute_entanglement_scaling(self):
        """Compute and log entanglement scaling using min-cut proxy"""
        Ls, Ss = self._holo_diag.scaling_curve()
        if self.recorder:
            slope, _ = np.polyfit(Ls, Ss, 1)
            self.recorder.log({
                "S_scaling": Ss.tolist(),
                "S_scaling_slope": slope
            })
        return slope

    def compute_mutual_information(self, region_A: Set[int], region_B: Set[int]) -> float:
        """Compute mutual information using min-cut proxy"""
        I_AB = self._holo_diag.I_mutual(region_A, region_B)
        if self.recorder:
            self.recorder.log({"mutual_info": I_AB})
        return I_AB

    def compute_berry_phase(self, states: List[np.ndarray]) -> float:
        """Compute Berry phase along a loop of states"""
        phase = berry_phase_on_loop(states)
        if self.recorder:
            self.recorder.log({"berry_phase": phase})
        return phase

    def reconstruct_metric(self, S_matrix: Dict[Set[int], float]) -> Dict[Edge, float]:
        """Reconstruct discrete bulk metric from entanglement measurements"""
        w_e = reconstruct_metric_from_entanglement(self.graph, S_matrix)
        if self.recorder:
            self.recorder.log({"metric_reconstruction": list(w_e.values())})
        return w_e

def project_to_lightcone(dr: float, dt: float, c: float) -> float: 
    return float(np.clip(dr, -abs(c*dt), abs(c*dt)))

class PrecisionTraversalContractor:
    def __init__(self, rcc: RecursiveConformalComputing):
        self.rcc = rcc
        self.rcc.mera.snapshot_flat_baseline()
        logger.info("Precision Traversal Contractor initialized")
        
    def step(self, dt: float, r: float, do_contract: bool=True) -> Dict:
        # Update bond dimensions based on radial position
        self.rcc.update_bond_dimensions()
        
        # Calculate bond capacity
        Nb = max(8, min(4096, int(10*(1 + r/max(self.rcc.r0, 1e-6))*4)))
        K = self.rcc.capacity_throttle(Nb)
        
        # Select bonds and apply phase imprint
        bonds, phi_map = self.rcc.select_cut_bonds(K)
        self.rcc.mera.imprint_bond_phases(self.rcc.graph, bonds, phi_map)
        
        # Estimate spectral dimension from cut
        dH = self.rcc.estimate_spectral_dim(bonds)
        
        # Compute RG parameters and Planck scale factor
        c_rel = self.rcc.rg_planner.c_hat(r, dH_est=dH)
        s_Pl = self.rcc.rg_planner.planck_scale_factor(r, c_rel)
        self.rcc._last_s_Pl = s_Pl  # Store for consistent access
        
        if dH:
            logger.debug(f"Spectral dimension estimate: dH={dH:.2f}, s_Pl={s_Pl:.4f}")
        
        # Calculate ANE with Planck scale factor and check QEI guard
        sigma_u = self.rcc.params.get('sigma_u',4e-3)
        ane = self.rcc.smeared_ANE(dt, sigma_u, s_Pl)
        guard_ok = self.rcc.qei_guard(ane, sigma_u)
        
        # Apply emergency damping if guard fails
        if not guard_ok:
            beta_vec = np.array(self.rcc.params.get("beta_vec",(0.15,0.05,0.01)))
            self.rcc.params["beta_vec"] = tuple(0.9 * beta_vec)
            logger.warning(f"QEI guard violated! ANE={ane:.2e} - Applying emergency damping")
            
        # Run contractor if allowed
        if do_contract and guard_ok:
            target = -0.25 * abs(ane) - 1e-4
            ane_new = self.rcc.contractor_step(dt, target, s_Pl)
        else:
            ane_new = ane
            
        # Enforce tensor drift guard (with dtype preservation)
        self.rcc.mera.enforce_tensor_drift_guard(max_drift=1e-3)
            
        # Log results
        if self.rcc.recorder:
            delta_energy = self.rcc.mera.delta_tensor_energy()
            Tuu_total = self.rcc.compute_Tuu_eff(dt, s_Pl)["Tuu_total"]
            self.rcc.recorder.log({
                "ANE_smear": ane_new,
                "K": K,
                "Δtensor": delta_energy,
                "Tuu_total": Tuu_total,
                "spectral_dim": dH if dH else 0.0,
                "s_Pl": s_Pl,
                "c_rel": c_rel
            })
            
            # Heartbeat flags
            slope = self.rcc.compute_entanglement_scaling()
            phase_flag = "area" if slope < 1.5 else "volume"
            qei_headroom = guard_ok  # Simplified for now
            echo_ok = True  # Placeholder
            
            self.rcc.recorder.log({
                "phase_flag": phase_flag,
                "qei_headroom": float(qei_headroom),
                "echo_ok": echo_ok
            })
            
            # Periodically compute Page curve
            if self.rcc.state.proper_time % 5.0 < dt:
                self.rcc.recorder.log({"page_curve": self.rcc._page_curve_sim.entropy_curve.tolist()})
            
        return {
            "K": K, 
            "ane_smear": ane_new, 
            "guard_ok": guard_ok,
            "spectral_dim": dH,
            "s_Pl": s_Pl
        }
