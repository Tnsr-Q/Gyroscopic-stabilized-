import numpy as np
import torch
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class ScarEthObserver:
    """
    Non-invasive observer for quantum scarring and ETH diagnostics.
    
    Hooks into RCC step notifications to collect:
    - Participation ratio (scar overlap proxy)
    - OTOC rate (chaos/scrambling proxy) 
    - ETH deviation (thermalization violation)
    """
    
    def __init__(self, recorder=None, sample_k: int = 64):
        self.recorder = recorder
        self.k = sample_k
        self._step_count = 0
        
    def update(self, tag: str, rcc, **kwargs):
        """Observer update hook called by RCC._notify()"""
        if tag != "after_step":
            return
            
        try:
            # Sample every k steps to avoid overhead
            self._step_count += 1
            if self._step_count % self.k != 0:
                return
                
            # Collect diagnostic signals
            metrics = self._collect_metrics(rcc)
            
            if self.recorder:
                self.recorder.log(metrics)
                
        except Exception as e:
            logger.debug(f"ScarEthObserver error: {e}")
    
    def _collect_metrics(self, rcc) -> Dict[str, float]:
        """Extract scar/ETH metrics from RCC state"""
        metrics = {}
        
        # Participation ratio (scar overlap proxy)
        try:
            pr = self._participation_ratio(rcc.mera.tensors)
            metrics["scar_overlap"] = pr
        except Exception:
            metrics["scar_overlap"] = 0.0
            
        # OTOC rate (chaos proxy via spectral spread)
        try:
            otoc = self._otoc_proxy(rcc.time_op)
            metrics["otoc_rate"] = otoc
        except Exception:
            metrics["otoc_rate"] = 0.0
            
        # ETH deviation (thermalization proxy)
        try:
            eth = self._eth_dev_proxy(rcc.state.law_field)
            metrics["eth_dev"] = eth
        except Exception:
            metrics["eth_dev"] = 0.0
            
        return metrics
    
    def _participation_ratio(self, tensors) -> float:
        """
        Quick participation ratio proxy from tensor amplitudes.
        PR = (Σ|ψᵢ|²)² / Σ|ψᵢ|⁴
        Higher values → more localization/scarring
        """
        try:
            # Collect tensor amplitudes
            x = torch.stack([t.abs().mean() for t in tensors]).detach().cpu().numpy()
            
            p2 = np.sum(x**2)
            p4 = np.sum(x**4) + 1e-12  # avoid division by zero
            
            return float((p2**2) / p4)
        except Exception:
            return 0.0
    
    def _otoc_proxy(self, time_op) -> float:
        """
        OTOC growth rate proxy via Hamiltonian spectral properties.
        Uses eigenvalue spread as chaos indicator.
        """
        try:
            if hasattr(time_op, 'H_int'):
                H = time_op.H_int
                spec = torch.linalg.eigvals(H).real
                spectral_spread = spec.std().item()
                return float(spectral_spread)
            else:
                return 0.0
        except Exception:
            return 0.0
    
    def _eth_dev_proxy(self, field) -> float:
        """
        ETH deviation proxy via field variance in microcanonical window.
        High variance → ETH violation (non-thermal behavior)
        """
        try:
            # Sample central block for efficiency
            if hasattr(field, 'shape') and len(field.shape) >= 2:
                h, w = field.shape[:2]
                block_size = min(8, h//2, w//2)
                if block_size > 0:
                    block = field[:block_size, :block_size]
                    if hasattr(block, 'cpu'):
                        block = block.cpu().numpy()
                    
                    variance = np.var(block)
                    mean_sq = np.mean(block**2) + 1e-12
                    return float(variance / mean_sq)
            
            return 0.0
        except Exception:
            return 0.0