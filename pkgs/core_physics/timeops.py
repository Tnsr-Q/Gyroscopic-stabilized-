"""
Time operation components for quantum evolution and recompression.

Contains TimeOperator and TimeRecompressionGate classes for managing
temporal evolution and quantum state recompression operations.
"""
import torch

# Device and dtype from original constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32


class TimeOperator:
    """Time evolution operator for proper time dynamics."""
    
    def __init__(self, dim_clock: int = 16):
        self.dim_clock = dim_clock
        self.H_int = torch.diag(torch.linspace(0, 1, dim_clock, device=DEVICE, dtype=DTYPE))
        
    def path_conditioned_evolution(self, tau: float) -> torch.Tensor:
        """Compute path-conditioned evolution operator."""
        return torch.matrix_exp(-1j * self.H_int * tau)
        
    def compute_decoherence(self, t1: float, t2: float, psi: torch.Tensor) -> float:
        """Compute decoherence between two time points."""
        U1 = self.path_conditioned_evolution(t1)
        U2 = self.path_conditioned_evolution(t2)
        return torch.abs(psi.conj() @ U1.conj().T @ U2 @ psi).item()


class TimeRecompressionGate:
    """Time recompression gate for quantum state compression."""
    
    def __init__(self, time_op: TimeOperator):
        self.time_op = time_op
        
    def create_gate(self, dt: float) -> torch.Tensor:
        """Create unitary recompression gate for time step dt."""
        return torch.matrix_exp(1j * self.time_op.H_int * dt)
        
    def apply_recompression(self, psi: torch.Tensor, dt: float) -> torch.Tensor:
        """Apply recompression to quantum state."""
        return self.create_gate(dt) @ psi
        
    def verify_coherence_restoration(self, t1: float, t2: float, psi: torch.Tensor) -> tuple[float, float]:
        """Verify coherence restoration between time points."""
        U1 = self.time_op.path_conditioned_evolution(t1)
        U2 = self.time_op.path_conditioned_evolution(t2)
        coherence_1 = torch.abs(psi.conj() @ U1 @ psi).item()
        coherence_2 = torch.abs(psi.conj() @ U2 @ psi).item()
        return coherence_1, coherence_2