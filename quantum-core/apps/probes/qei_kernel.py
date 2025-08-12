"""QEI kernel computations and ANE calculations."""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class QEIKernel:
    """Quantum Energy Inequality kernel computations."""
    
    def __init__(self, kernel_type: str = "gaussian", sigma_default: float = 4e-3):
        self.kernel_type = kernel_type.lower()
        self.sigma_default = sigma_default
        self._calibration_cache = {}
    
    def qei_calibration(self, sigma_u: float) -> float:
        """
        Returns dimensionless C_eff so that bound ≈ - C_eff / sigma_u^4.
        This is a calibrated constant based on the kernel type.
        """
        key = (self.kernel_type, round(float(sigma_u), 6))
        if key in self._calibration_cache:
            return self._calibration_cache[key]
        
        if self.kernel_type == "gaussian":
            # Fitted constant that preserves current scale
            C_eff = 1.0
        elif self.kernel_type in ("fermi", "fermi-dirac", "sigmoid"):
            C_eff = 0.8
        elif self.kernel_type == "lorentzian":
            C_eff = 1.2
        else:
            C_eff = 1.0
        
        self._calibration_cache[key] = C_eff
        return C_eff
    
    def qei_bound(self, sigma_u: float) -> float:
        """Compute QEI bound for given smearing scale."""
        C = self.qei_calibration(sigma_u)
        sigma = max(1e-9, float(sigma_u))
        return -float(C) / (sigma**4)
    
    def check_violation(self, ane_value: float, sigma_u: float, margin: float = 0.05) -> bool:
        """Check if ANE violates QEI bound with safety margin."""
        bound = self.qei_bound(sigma_u)
        threshold = bound * (1.0 - margin)
        return ane_value < threshold
    
    def safety_margin(self, ane_value: float, sigma_u: float) -> float:
        """Compute safety margin relative to QEI bound."""
        bound = self.qei_bound(sigma_u)
        if bound >= 0:
            return float('inf')  # No constraint
        return (ane_value - bound) / abs(bound)

class ANECalculator:
    """Averaged Null Energy calculator with various smearing kernels."""
    
    def __init__(self, kernel_type: str = "gaussian"):
        self.kernel_type = kernel_type.lower()
        self.history = []
        self.current_time = 0.0
    
    def add_sample(self, time: float, energy_density: float):
        """Add a time-energy sample to the history."""
        self.history.append((time, energy_density))
        self.current_time = max(self.current_time, time)
    
    def clear_history(self):
        """Clear the energy density history."""
        self.history.clear()
        self.current_time = 0.0
    
    def gaussian_weight(self, tau: float, sigma: float) -> float:
        """Gaussian smearing kernel."""
        return np.exp(-0.5 * (tau / max(sigma, 1e-12))**2)
    
    def lorentzian_weight(self, tau: float, sigma: float) -> float:
        """Lorentzian smearing kernel."""
        return 1.0 / (1.0 + (tau / max(sigma, 1e-12))**2)
    
    def fermi_weight(self, tau: float, sigma: float, beta: float = 10.0) -> float:
        """Fermi-Dirac style smearing kernel."""
        x = beta * tau / max(sigma, 1e-12)
        return 1.0 / (1.0 + np.exp(abs(x)))
    
    def compute_ane(self, sigma_u: float, max_age: Optional[float] = None) -> float:
        """
        Compute smeared ANE using the specified kernel.
        
        Args:
            sigma_u: Smearing scale
            max_age: Maximum age of samples to consider (default: 6*sigma_u)
        """
        if not self.history:
            return 0.0
        
        if max_age is None:
            max_age = 6.0 * sigma_u
        
        # Filter history by age
        cutoff_time = self.current_time - max_age
        relevant_history = [(t, v) for t, v in self.history if t >= cutoff_time]
        
        if not relevant_history:
            return 0.0
        
        # Compute weighted average
        num, den = 0.0, 0.0
        
        for t, v in relevant_history:
            tau = abs(self.current_time - t)
            
            if self.kernel_type == "gaussian":
                weight = self.gaussian_weight(tau, sigma_u)
            elif self.kernel_type == "lorentzian":
                weight = self.lorentzian_weight(tau, sigma_u)
            elif self.kernel_type in ("fermi", "fermi-dirac"):
                weight = self.fermi_weight(tau, sigma_u)
            else:
                weight = self.gaussian_weight(tau, sigma_u)  # fallback
            
            num += v * weight
            den += weight
        
        return float(num / max(den, 1e-12))
    
    def compute_ane_series(self, sigma_u: float, time_points: List[float]) -> List[float]:
        """Compute ANE at multiple time points."""
        results = []
        original_time = self.current_time
        
        for t in time_points:
            self.current_time = t
            ane = self.compute_ane(sigma_u)
            results.append(ane)
        
        self.current_time = original_time
        return results
    
    def get_statistics(self, sigma_u: float) -> Dict[str, float]:
        """Get comprehensive ANE statistics."""
        if not self.history:
            return {
                "ane": 0.0,
                "min_energy": 0.0,
                "max_energy": 0.0,
                "mean_energy": 0.0,
                "samples": 0
            }
        
        energies = [v for _, v in self.history]
        
        return {
            "ane": self.compute_ane(sigma_u),
            "min_energy": min(energies),
            "max_energy": max(energies),
            "mean_energy": np.mean(energies),
            "std_energy": np.std(energies),
            "samples": len(self.history),
            "time_span": max(t for t, _ in self.history) - min(t for t, _ in self.history)
        }