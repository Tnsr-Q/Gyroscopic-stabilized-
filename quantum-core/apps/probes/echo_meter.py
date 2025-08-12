"""Loschmidt echo computation for quantum revivals."""

import torch
import numpy as np
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)

class EchoMeter:
    """Loschmidt echo measurement for quantum coherence tracking."""
    
    def __init__(self, time_operator, initial_state: torch.Tensor, device: Optional[str] = None):
        """
        Initialize echo meter.
        
        Args:
            time_operator: TimeOperator instance with H_int Hamiltonian
            initial_state: Initial quantum state |ψ⟩
            device: Device for computations
        """
        self.time_op = time_operator
        self.psi0 = initial_state.detach().clone()
        self.device = device or (self.psi0.device if hasattr(self.psi0, "device") else "cpu")
        self.echo_history = []
    
    @torch.no_grad()
    def measure_echo(self, t: float, perturbation: float = 1e-3) -> float:
        """
        Measure Loschmidt echo L(t) = |⟨ψ| e^{i(H+δ)t} e^{-iHt} |ψ⟩|².
        
        Args:
            t: Evolution time
            perturbation: Perturbation strength δ relative to H
            
        Returns:
            Loschmidt echo value between 0 and 1
        """
        H = self.time_op.H_int.to(dtype=torch.complex64)
        
        # Forward evolution with unperturbed Hamiltonian
        U = torch.matrix_exp(-1j * H * t)
        
        # Backward evolution with perturbed Hamiltonian
        H_pert = H * (1.0 + perturbation)
        U_pert = torch.matrix_exp(1j * H_pert * t)
        
        # Compute echo amplitude
        psi_evolved = U @ self.psi0
        psi_back = U_pert @ psi_evolved
        
        # Echo is |⟨ψ₀|ψ_back⟩|²
        amplitude = torch.vdot(self.psi0, psi_back)
        echo = torch.abs(amplitude)**2
        
        result = float(echo.item())
        self.echo_history.append((t, result))
        
        return result
    
    @torch.no_grad()
    def measure_echo_sequence(self, times: List[float], perturbation: float = 1e-3) -> List[float]:
        """Measure echo at multiple time points."""
        echoes = []
        for t in times:
            echo = self.measure_echo(t, perturbation)
            echoes.append(echo)
        return echoes
    
    @torch.no_grad()
    def quantum_fidelity(self, t1: float, t2: float) -> float:
        """
        Compute quantum fidelity between states evolved for different times.
        
        Args:
            t1, t2: Evolution times
            
        Returns:
            Fidelity |⟨ψ(t1)|ψ(t2)⟩|²
        """
        H = self.time_op.H_int.to(dtype=torch.complex64)
        
        U1 = torch.matrix_exp(-1j * H * t1)
        U2 = torch.matrix_exp(-1j * H * t2)
        
        psi1 = U1 @ self.psi0
        psi2 = U2 @ self.psi0
        
        fidelity = torch.abs(torch.vdot(psi1, psi2))**2
        return float(fidelity.item())
    
    def analyze_echo_decay(self, times: Optional[List[float]] = None) -> dict:
        """
        Analyze echo decay characteristics.
        
        Args:
            times: Time points to analyze (uses history if None)
            
        Returns:
            Dictionary with decay analysis
        """
        if times is not None:
            echoes = self.measure_echo_sequence(times)
            time_points = times
        else:
            if not self.echo_history:
                return {"error": "No echo history available"}
            time_points = [t for t, _ in self.echo_history]
            echoes = [e for _, e in self.echo_history]
        
        if len(echoes) < 3:
            return {"error": "Insufficient data for analysis"}
        
        # Convert to numpy for analysis
        t_arr = np.array(time_points)
        e_arr = np.array(echoes)
        
        # Fit exponential decay: L(t) ≈ e^(-γt)
        log_echoes = np.log(np.maximum(e_arr, 1e-12))
        
        try:
            # Linear fit in log space
            coeffs = np.polyfit(t_arr, log_echoes, 1)
            decay_rate = -coeffs[0]  # γ = -slope
            
            # Correlation decay time
            tau_corr = 1.0 / max(decay_rate, 1e-12)
            
            # Revival characteristics
            revivals = self._find_revivals(t_arr, e_arr)
            
            analysis = {
                "decay_rate": float(decay_rate),
                "correlation_time": float(tau_corr),
                "initial_echo": float(e_arr[0]) if len(e_arr) > 0 else 0.0,
                "final_echo": float(e_arr[-1]) if len(e_arr) > 0 else 0.0,
                "mean_echo": float(np.mean(e_arr)),
                "std_echo": float(np.std(e_arr)),
                "revivals": revivals,
                "time_span": float(t_arr[-1] - t_arr[0]) if len(t_arr) > 1 else 0.0
            }
            
        except (np.linalg.LinAlgError, ValueError) as e:
            logger.warning(f"Echo decay analysis failed: {e}")
            analysis = {
                "error": str(e),
                "mean_echo": float(np.mean(e_arr)),
                "std_echo": float(np.std(e_arr))
            }
        
        return analysis
    
    def _find_revivals(self, times: np.ndarray, echoes: np.ndarray, 
                      threshold: float = 0.8, min_separation: float = 1.0) -> List[dict]:
        """Find revival peaks in echo signal."""
        if len(echoes) < 5:
            return []
        
        revivals = []
        
        # Find local maxima above threshold
        for i in range(1, len(echoes) - 1):
            if (echoes[i] > echoes[i-1] and 
                echoes[i] > echoes[i+1] and 
                echoes[i] > threshold):
                
                # Check minimum separation from previous revivals
                if not revivals or (times[i] - revivals[-1]["time"]) > min_separation:
                    revivals.append({
                        "time": float(times[i]),
                        "echo": float(echoes[i]),
                        "index": int(i)
                    })
        
        return revivals
    
    def reset(self, new_initial_state: Optional[torch.Tensor] = None):
        """Reset echo meter with new initial state."""
        if new_initial_state is not None:
            self.psi0 = new_initial_state.detach().clone()
        self.echo_history.clear()
        logger.info("Echo meter reset")
    
    def get_echo_statistics(self) -> dict:
        """Get comprehensive echo statistics from history."""
        if not self.echo_history:
            return {"error": "No echo history"}
        
        echoes = [e for _, e in self.echo_history]
        times = [t for t, _ in self.echo_history]
        
        return {
            "num_measurements": len(echoes),
            "time_range": (min(times), max(times)),
            "echo_range": (min(echoes), max(echoes)),
            "mean_echo": float(np.mean(echoes)),
            "std_echo": float(np.std(echoes)),
            "median_echo": float(np.median(echoes)),
            "last_echo": echoes[-1],
            "coherence_fraction": float(np.mean([e > 0.5 for e in echoes]))
        }