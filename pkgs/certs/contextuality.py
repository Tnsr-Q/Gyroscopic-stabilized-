import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
import warnings
import os
import csv

logger = logging.getLogger(__name__)

def _load_ks_rays() -> List[Tuple[float, float, float]]:
    """Load KS rays from bundled CSV data."""
    try:
        csv_path = os.path.join(os.path.dirname(__file__), 'ks_rays_57.csv')
        rays = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                x, y, z = float(row['x']), float(row['y']), float(row['z'])
                rays.append((x, y, z))
        return rays
    except Exception as e:
        logger.debug(f"Failed to load KS rays from CSV: {e}")
        # Fallback to embedded minimal set
        return [
            (1, 0, 0), (0, 1, 0), (0, 0, 1),
            (1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1),
            (0, 1, 1), (0, 1, -1), (1, 1, 1), (1, 1, -1),
            (1, -1, 1), (1, -1, -1)
        ]

# Load and normalize rays
KS_57_RAYS = _load_ks_rays()
KS_57_RAYS = [tuple(np.array(ray) / np.linalg.norm(ray)) for ray in KS_57_RAYS]

class KSCert:
    """Kochen-Specker contextuality certificate."""
    
    def __init__(self, rays: Optional[List[Tuple[float, float, float]]] = None):
        self.rays = rays or KS_57_RAYS
        self.n_rays = len(self.rays)
        
    def compute_ks_violation(self, measurement_probs: Dict[int, float]) -> float:
        """
        Compute KS contextuality violation.
        
        Args:
            measurement_probs: {ray_index: probability} for each ray
            
        Returns:
            Violation strength (0 = classical, >0 = quantum contextual)
        """
        try:
            # Simple KS sum rule: Σ P(ray_i) for orthogonal triples should = 1
            # But quantum mechanics can violate this
            violations = []
            
            for i in range(0, len(self.rays) - 2, 3):
                if i + 2 < len(self.rays):
                    # Check if rays are approximately orthogonal
                    r1, r2, r3 = self.rays[i], self.rays[i+1], self.rays[i+2]
                    if self._are_orthogonal(r1, r2, r3):
                        p1 = measurement_probs.get(i, 0.5)
                        p2 = measurement_probs.get(i+1, 0.5)  
                        p3 = measurement_probs.get(i+2, 0.5)
                        
                        # Classical constraint: p1 + p2 + p3 = 1 for orthogonal triple
                        violation = abs((p1 + p2 + p3) - 1.0)
                        violations.append(violation)
            
            return float(np.mean(violations)) if violations else 0.0
            
        except Exception as e:
            logger.debug(f"KS violation computation failed: {e}")
            return 0.0
    
    def _are_orthogonal(self, r1: Tuple, r2: Tuple, r3: Tuple, tol: float = 0.1) -> bool:
        """Check if three rays form an approximately orthogonal triple."""
        try:
            v1, v2, v3 = np.array(r1), np.array(r2), np.array(r3)
            dots = [abs(np.dot(v1, v2)), abs(np.dot(v1, v3)), abs(np.dot(v2, v3))]
            return all(d < tol for d in dots)
        except Exception:
            return False

class LovaszTheta:
    """Lovász theta function for quantum advantage bounds."""
    
    def __init__(self):
        self.has_cvxpy = self._check_cvxpy()
        self.has_networkx = self._check_networkx()
        
    def _check_cvxpy(self) -> bool:
        try:
            import cvxpy
            return True
        except ImportError:
            return False
            
    def _check_networkx(self) -> bool:
        try:
            import networkx
            return True
        except ImportError:
            return False
    
    def compute_theta_bound(self, adjacency_matrix: np.ndarray) -> float:
        """
        Compute Lovász theta lower bound.
        Falls back to spectral bound if SDP solver unavailable.
        """
        try:
            if self.has_cvxpy:
                return self._sdp_theta_bound(adjacency_matrix)
            else:
                return self._spectral_theta_bound(adjacency_matrix)
        except Exception as e:
            logger.debug(f"Theta bound computation failed: {e}")
            return 1.0  # Conservative fallback
    
    def _sdp_theta_bound(self, adj: np.ndarray) -> float:
        """SDP formulation of Lovász theta (requires cvxpy)."""
        try:
            import cvxpy as cp
            
            n = adj.shape[0]
            
            # SDP: maximize Tr(J * X) subject to Tr(X) = 1, X_ij = 0 if (i,j) in E
            X = cp.Variable((n, n), symmetric=True)
            J = np.ones((n, n))  # All-ones matrix
            
            constraints = [X >> 0, cp.trace(X) == 1]  # PSD and trace constraint
            
            # Zero constraints for adjacent vertices
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i, j] > 0:  # Edge exists
                        constraints.append(X[i, j] == 0)
            
            objective = cp.Maximize(cp.trace(J @ X))
            problem = cp.Problem(objective, constraints)
            
            problem.solve(verbose=False)
            
            if problem.status == cp.OPTIMAL:
                return float(problem.value)
            else:
                return self._spectral_theta_bound(adj)
                
        except Exception:
            return self._spectral_theta_bound(adj)
    
    def _spectral_theta_bound(self, adj: np.ndarray) -> float:
        """Spectral approximation (fallback when SDP unavailable)."""
        try:
            # Use largest eigenvalue of adjacency matrix as rough bound
            eigenvals = np.linalg.eigvals(adj)
            lambda_max = float(np.max(eigenvals.real))
            
            # Rough approximation: θ(G) ≥ n/(1 + λ_max)
            n = adj.shape[0]
            return float(n / (1 + lambda_max + 1e-12))
            
        except Exception:
            return 1.0

class ContextualityCert:
    """Main contextuality certificate interface."""
    
    def __init__(self):
        self.ks_cert = KSCert()
        self.theta_computer = LovaszTheta()
        
    def run_quickcheck(self, state_data: Optional[Dict] = None) -> Dict[str, float]:
        """
        Run quick contextuality check on provided state data.
        
        Args:
            state_data: Optional quantum state information
            
        Returns:
            Dictionary with contextuality metrics
        """
        results = {}
        
        try:
            # Generate mock measurement probabilities if no state provided
            if state_data is None:
                # Create slightly contextual probabilities for demonstration
                n_rays = len(self.ks_cert.rays)
                probs = {i: 0.33 + 0.1 * np.sin(i) for i in range(n_rays)}
            else:
                # Extract probabilities from actual state (to be implemented)
                probs = self._extract_measurement_probs(state_data)
            
            # Compute KS violation
            ks_violation = self.ks_cert.compute_ks_violation(probs)
            results["ks_violation"] = ks_violation
            
            # Create adjacency matrix for theta bound (simplified graph)
            n = min(len(probs), 8)  # Limit size for efficiency
            adj_matrix = self._create_test_graph(n)
            
            # Compute theta bound
            theta_bound = self.theta_computer.compute_theta_bound(adj_matrix)
            results["theta_lower_bound"] = theta_bound
            
            # Contextual fraction (derived metric)
            contextual_fraction = min(ks_violation / 0.1, 1.0)  # Normalize
            results["contextual_fraction"] = contextual_fraction
            
            results["cert_success"] = True
            
        except Exception as e:
            logger.debug(f"Contextuality certificate failed: {e}")
            results.update({
                "ks_violation": 0.0,
                "theta_lower_bound": 1.0,
                "contextual_fraction": 0.0,
                "cert_success": False
            })
        
        return results
    
    def _extract_measurement_probs(self, state_data: Dict) -> Dict[int, float]:
        """Extract measurement probabilities from quantum state."""
        # Placeholder - implement based on your state representation
        if isinstance(state_data, dict):
            # Try to extract relevant data
            coherence = state_data.get('coherence_gamma', 1.0)
            entropy = state_data.get('entropy', 0.0)
            proper_time = state_data.get('proper_time', 0.0)
            
            # Generate probabilities based on state parameters
            n_rays = min(len(self.ks_cert.rays), 16)
            probs = {}
            for i in range(n_rays):
                # Simple mapping from state to probabilities
                base_prob = 0.3 + 0.2 * coherence
                phase_factor = np.sin(i * proper_time + entropy)
                probs[i] = max(0.0, min(1.0, base_prob + 0.1 * phase_factor))
            
            return probs
        else:
            # Fallback to random probabilities
            return {i: 0.3 + 0.2 * np.random.random() for i in range(8)}
    
    def _create_test_graph(self, n: int) -> np.ndarray:
        """Create test graph adjacency matrix."""
        adj = np.zeros((n, n))
        
        # Create cycle graph as test case
        for i in range(n):
            adj[i, (i+1) % n] = 1
            adj[(i+1) % n, i] = 1
            
        return adj

# Global instance for easy access
ks_cert = ContextualityCert()