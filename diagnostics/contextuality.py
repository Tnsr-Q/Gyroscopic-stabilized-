import numpy as np
import torch
from typing import Dict, Optional, List, Tuple
import warnings

def _construct_ks_observables(dim: int) -> List[np.ndarray]:
    """Construct a simple KS set for contextuality testing."""
    if dim < 3:
        return []
    # Simple KS set: Pauli-like observables for qubits
    if dim == 4:  # two qubits
        X1 = np.kron([[0,1],[1,0]], np.eye(2))
        Z1 = np.kron([[1,0],[0,-1]], np.eye(2))
        X2 = np.kron(np.eye(2), [[0,1],[1,0]])
        Z2 = np.kron(np.eye(2), [[1,0],[0,-1]])
        return [X1, Z1, X2, Z2, X1@X2, Z1@Z2]
    else:
        # For higher dims, use random Hermitian observables
        rs = np.random.RandomState(42)
        obs = []
        for _ in range(min(dim, 6)):
            A = rs.randn(dim, dim) + 1j*rs.randn(dim, dim)
            A = (A + A.conj().T) / 2  # Hermitian
            obs.append(A)
        return obs

def kochen_specker_violation(psi: torch.Tensor) -> float:
    """
    Crude KS contextuality measure: variance in expectation values
    across overlapping contexts. Higher variance → more contextuality.
    """
    try:
        rho = psi.detach().cpu().numpy()
        if rho.ndim == 1:
            rho = np.outer(rho.conj(), rho)
        dim = rho.shape[0]
        
        observables = _construct_ks_observables(dim)
        if not observables:
            return 0.0
            
        expectations = []
        for A in observables:
            exp_val = np.trace(rho @ A).real
            expectations.append(exp_val)
        
        # Contextuality proxy: variance in expectations
        return float(np.var(expectations))
    except Exception as e:
        warnings.warn(f"KS computation failed: {e}")
        return 0.0

def lovasz_theta_bound(adj_matrix: np.ndarray) -> float:
    """
    Lovász theta function via SDP relaxation with graceful fallback.
    Falls back to simple eigenvalue bound if SDP fails.
    """
    try:
        # Check for problematic input first
        if not np.isfinite(adj_matrix).all():
            warnings.warn("Non-finite values in adjacency matrix")
            return 1.0
            
        # Try to import cvxpy for SDP
        try:
            import cvxpy as cp
            n = adj_matrix.shape[0]
            X = cp.Variable((n, n), symmetric=True)
            constraints = [X >> 0, cp.trace(X) == 1]
            
            # Off-diagonal constraints for non-adjacent vertices
            for i in range(n):
                for j in range(i+1, n):
                    if adj_matrix[i,j] == 0:  # non-adjacent
                        constraints.append(X[i,j] == 0)
            
            prob = cp.Problem(cp.Maximize(cp.sum(X)), constraints)
            prob.solve(verbose=False)
            
            if prob.status == cp.OPTIMAL:
                return float(prob.value)
        except ImportError:
            pass
            
        # Fallback: simple eigenvalue bound
        L = np.diag(np.sum(adj_matrix, axis=1)) - adj_matrix  # Laplacian
        eigvals = np.linalg.eigvals(L)
        lambda_max = np.max(eigvals.real)
        return float(adj_matrix.shape[0] / max(lambda_max, 1e-12))
        
    except Exception as e:
        warnings.warn(f"Lovász theta computation failed: {e}")
        return 1.0  # safe fallback

def contextuality_graph_from_state(psi: torch.Tensor, 
                                   threshold: float = 0.1) -> np.ndarray:
    """
    Build contextuality graph from quantum state correlations.
    Edges represent strong correlations between measurement contexts.
    """
    try:
        rho = psi.detach().cpu().numpy()
        if rho.ndim == 1:
            rho = np.outer(rho.conj(), rho)
        dim = rho.shape[0]
        
        # Use local observables as contexts
        contexts = _construct_ks_observables(dim)
        if len(contexts) < 2:
            return np.eye(2)  # trivial graph
            
        n_ctx = len(contexts)
        adj = np.zeros((n_ctx, n_ctx))
        
        # Edge if contexts have strong correlation
        for i in range(n_ctx):
            for j in range(i+1, n_ctx):
                # Correlation measure: |Tr(ρ[A_i, A_j])|
                comm = contexts[i] @ contexts[j] - contexts[j] @ contexts[i]
                corr = abs(np.trace(rho @ comm))
                if corr > threshold:
                    adj[i,j] = adj[j,i] = 1
                    
        return adj
    except Exception as e:
        warnings.warn(f"Contextuality graph construction failed: {e}")
        return np.eye(2)

def contextuality_certificate(psi: torch.Tensor) -> Dict[str, float]:
    """
    Combined contextuality analysis with robust error handling.
    """
    try:
        # Check for problematic input first
        if not torch.isfinite(psi).all():
            warnings.warn("Non-finite values in quantum state")
            return {
                "ks_violation": 0.0,
                "lovasz_theta": 1.0, 
                "contextuality_strength": 0.0,
                "graph_edges": 0.0
            }
            
        ks_viol = kochen_specker_violation(psi)
        ctx_graph = contextuality_graph_from_state(psi)
        theta_bound = lovasz_theta_bound(ctx_graph)
        
        # Contextuality strength: combination of KS violation and theta bound
        ctx_strength = ks_viol * min(theta_bound, 10.0)  # cap extreme values
        
        return {
            "ks_violation": ks_viol,
            "lovasz_theta": theta_bound,
            "contextuality_strength": ctx_strength,
            "graph_edges": float(np.sum(ctx_graph) / 2)  # edge count
        }
    except Exception as e:
        warnings.warn(f"Contextuality certificate failed: {e}")
        return {
            "ks_violation": 0.0,
            "lovasz_theta": 1.0, 
            "contextuality_strength": 0.0,
            "graph_edges": 0.0
        }