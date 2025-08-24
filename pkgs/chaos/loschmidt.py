import numpy as np
import torch
import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

@dataclass
class EchoResult:
    """Loschmidt echo measurement result."""
    echo_value: float
    decay_rate: float
    revival_strength: float
    chaos_indicator: float
    measurement_time: float
    success: bool

class LoschmidtMeter:
    """
    Loschmidt echo meter for quantum chaos detection.
    
    Measures L(t) = |⟨ψ(0)|e^{iH't}e^{-iHt}|ψ(0)⟩|²
    where H is original Hamiltonian and H' is perturbed version.
    
    Decay patterns indicate:
    - Exponential decay → quantum chaos
    - Algebraic decay → critical systems
    - Oscillatory decay → integrable systems
    """
    
    def __init__(self, 
                 perturbation_strength: float = 1e-3,
                 max_time: float = 10.0,
                 time_steps: int = 50,
                 enable_gpu: bool = True):
        self.eps = perturbation_strength
        self.t_max = max_time
        self.n_steps = time_steps
        self.enable_gpu = enable_gpu and torch.cuda.is_available()
        
        self.device = torch.device('cuda' if self.enable_gpu else 'cpu')
        
        # Cache for efficient computation
        self._cached_H = None
        self._cached_H_prime = None
        self._cache_valid = False
        
    def probe_echo(self, hamiltonian: torch.Tensor, 
                   initial_state: Optional[torch.Tensor] = None,
                   perturbation_type: str = 'random') -> EchoResult:
        """
        Probe Loschmidt echo for given Hamiltonian.
        
        Args:
            hamiltonian: System Hamiltonian tensor
            initial_state: Initial quantum state (default: random)
            perturbation_type: Type of perturbation ('random', 'diagonal', 'field')
            
        Returns:
            EchoResult with decay characteristics
        """
        start_time = time.time()
        
        try:
            # Move to appropriate device
            H = hamiltonian.to(self.device)
            
            # Generate or validate initial state
            if initial_state is None:
                psi0 = self._random_state(H.shape[0])
            else:
                psi0 = initial_state.to(self.device)
                psi0 = psi0 / torch.norm(psi0)  # Normalize
            
            # Create perturbed Hamiltonian
            H_prime = self._create_perturbation(H, perturbation_type)
            
            # Compute time evolution and echo
            echo_values, times = self._compute_echo_evolution(H, H_prime, psi0)
            
            # Analyze decay characteristics
            result = self._analyze_echo_decay(echo_values, times)
            result.measurement_time = time.time() - start_time
            result.success = True
            
            return result
            
        except Exception as e:
            logger.debug(f"Loschmidt echo computation failed: {e}")
            return EchoResult(
                echo_value=0.0,
                decay_rate=0.0, 
                revival_strength=0.0,
                chaos_indicator=0.0,
                measurement_time=time.time() - start_time,
                success=False
            )
    
    def _random_state(self, dim: int) -> torch.Tensor:
        """Generate random normalized quantum state."""
        # Complex random state
        real_part = torch.randn(dim, device=self.device)
        imag_part = torch.randn(dim, device=self.device)
        psi = torch.complex(real_part, imag_part)
        return psi / torch.norm(psi)
    
    def _create_perturbation(self, H: torch.Tensor, pert_type: str) -> torch.Tensor:
        """Create perturbed Hamiltonian H' = H + ε * V."""
        if pert_type == 'random':
            # Random Hermitian perturbation
            V_real = torch.randn_like(H.real)
            V = V_real + V_real.T  # Make Hermitian
            V = V / torch.norm(V) * self.eps
            
        elif pert_type == 'diagonal':
            # Random diagonal perturbation (preserves more structure)
            diag_pert = torch.randn(H.shape[0], device=self.device) * self.eps
            V = torch.diag(diag_pert)
            
        elif pert_type == 'field':
            # Magnetic field-like perturbation (Pauli matrices for small systems)
            if H.shape[0] <= 4:
                V = self._pauli_field_perturbation(H.shape[0])
            else:
                V = self._create_perturbation(H, 'random')  # Fallback
                
        else:
            raise ValueError(f"Unknown perturbation type: {pert_type}")
        
        return H + V
    
    def _pauli_field_perturbation(self, dim: int) -> torch.Tensor:
        """Create Pauli matrix perturbation for small systems."""
        if dim == 2:
            # Single qubit: σ_z perturbation
            pauli_z = torch.tensor([[1.0, 0.0], [0.0, -1.0]], 
                                 device=self.device, dtype=torch.complex64)
            return self.eps * pauli_z
        else:
            # Fallback to diagonal for larger systems
            diag_pert = torch.randn(dim, device=self.device) * self.eps
            return torch.diag(diag_pert)
    
    def _compute_echo_evolution(self, H: torch.Tensor, H_prime: torch.Tensor, 
                               psi0: torch.Tensor) -> Tuple[List[float], List[float]]:
        """Compute Loschmidt echo evolution L(t)."""
        times = np.linspace(0, self.t_max, self.n_steps)
        echo_values = []
        
        for t in times:
            if t == 0:
                echo_values.append(1.0)  # Perfect overlap at t=0
                continue
                
            # Forward evolution: |ψ(t)⟩ = e^{-iHt}|ψ(0)⟩  
            U_forward = torch.matrix_exp(-1j * H * t)
            psi_t = U_forward @ psi0
            
            # Backward evolution with perturbed H: e^{iH't}|ψ(t)⟩
            U_backward = torch.matrix_exp(1j * H_prime * t)
            psi_echo = U_backward @ psi_t
            
            # Echo fidelity: |⟨ψ(0)|ψ_echo⟩|²
            overlap = torch.vdot(psi0, psi_echo)
            echo = float(torch.abs(overlap) ** 2)
            echo_values.append(echo)
        
        return echo_values, times.tolist()
    
    def _analyze_echo_decay(self, echo_values: List[float], 
                           times: List[float]) -> EchoResult:
        """Analyze echo decay to extract chaos indicators."""
        echo_array = np.array(echo_values)
        time_array = np.array(times)
        
        # Skip t=0 for decay analysis
        if len(echo_array) > 1:
            echo_decay = echo_array[1:]
            time_decay = time_array[1:]
        else:
            echo_decay = echo_array
            time_decay = time_array
        
        # Current echo value
        current_echo = float(echo_array[-1])
        
        # Estimate decay rate via linear fit in log space
        try:
            # Avoid log(0) by adding small offset
            log_echo = np.log(echo_decay + 1e-12)
            
            # Linear fit: log(L) ~ -γt + const
            coeffs = np.polyfit(time_decay, log_echo, 1)
            decay_rate = float(-coeffs[0])  # γ > 0 for decay
            
        except Exception:
            decay_rate = 0.0
        
        # Revival strength: look for oscillatory components
        try:
            # High-frequency content as proxy for revivals
            fft_echo = np.fft.fft(echo_decay)
            high_freq_power = np.sum(np.abs(fft_echo[len(fft_echo)//4:]))
            total_power = np.sum(np.abs(fft_echo)) + 1e-12
            revival_strength = float(high_freq_power / total_power)
            
        except Exception:
            revival_strength = 0.0
        
        # Chaos indicator: combine decay rate and regularity
        # Fast exponential decay → chaos
        # Slow algebraic decay → criticality  
        # Oscillatory behavior → integrability
        chaos_score = min(decay_rate * 10, 1.0)  # Normalize to [0,1]
        
        # Reduce chaos score if strong revivals present (indicates integrability)
        if revival_strength > 0.3:
            chaos_score *= (1 - revival_strength)
        
        return EchoResult(
            echo_value=current_echo,
            decay_rate=decay_rate,
            revival_strength=revival_strength,
            chaos_indicator=float(chaos_score),
            measurement_time=0.0,  # Will be set by caller
            success=True
        )
    
    def quick_chaos_check(self, hamiltonian: torch.Tensor, 
                         threshold: float = 0.5) -> Dict[str, Union[float, bool]]:
        """
        Quick chaos check for integration into main loop.
        
        Args:
            hamiltonian: System Hamiltonian
            threshold: Chaos detection threshold
            
        Returns:
            Dictionary with chaos metrics
        """
        try:
            # Reduced time steps for speed
            original_steps = self.n_steps
            self.n_steps = min(20, self.n_steps)
            
            result = self.probe_echo(hamiltonian)
            
            # Restore original settings
            self.n_steps = original_steps
            
            return {
                'echo_value': result.echo_value,
                'decay_rate': result.decay_rate,
                'chaos_indicator': result.chaos_indicator,
                'is_chaotic': result.chaos_indicator > threshold,
                'measurement_time': result.measurement_time,
                'success': result.success
            }
            
        except Exception as e:
            logger.debug(f"Quick chaos check failed: {e}")
            return {
                'echo_value': 0.0,
                'decay_rate': 0.0,
                'chaos_indicator': 0.0,
                'is_chaotic': False,
                'measurement_time': 0.0,
                'success': False
            }

class EchoObserver:
    """Observer for integrating Loschmidt echo into RCC pipeline."""
    
    def __init__(self, meter: LoschmidtMeter, recorder=None, 
                 sample_interval: int = 100, chaos_threshold: float = 0.5):
        self.meter = meter
        self.recorder = recorder
        self.k = sample_interval
        self.threshold = chaos_threshold
        self._step_count = 0
        
    def update(self, tag: str, rcc, **kwargs):
        """Observer update hook."""
        if tag != "after_step":
            return
            
        try:
            self._step_count += 1
            if self._step_count % self.k != 0:
                return
            
            # Extract Hamiltonian from RCC
            if hasattr(rcc, 'time_op') and hasattr(rcc.time_op, 'H_int'):
                H = rcc.time_op.H_int
                
                # Quick chaos check
                result = self.meter.quick_chaos_check(H, self.threshold)
                
                if self.recorder:
                    self.recorder.log(result)
                
                # Alert on chaos detection
                if result.get('is_chaotic', False):
                    logger.info(f"Quantum chaos detected: {result['chaos_indicator']:.3f}")
                    
        except Exception as e:
            logger.debug(f"LoschmidtEcho observer error: {e}")

# Global probe instance
loschmidt_probe = LoschmidtMeter()