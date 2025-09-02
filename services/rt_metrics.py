# services/rt_metrics.py
from __future__ import annotations
from typing import Dict, Any
import torch
import numpy as np

class RealTimeGaugeMetrics:
    """
    Real-time metrics computation for the gauge field system.
    Provides lightweight metrics for the Brontein Cube sense function.
    """
    
    def __init__(self, rcc):
        self.rcc = rcc
    
    def compute_realtime_metrics(self) -> Dict[str, Any]:
        """Compute real-time metrics from the current system state."""
        try:
            # Basic metrics from RCC state
            metrics = {
                'qei_headroom': 0.0,  # Default safe value
                'total_conditioning': 1.0,  # Default safe value  
                'phase_indicator': 0.0,  # Default safe value
            }
            
            # Try to get more sophisticated metrics if available
            if hasattr(self.rcc, 'state'):
                state = self.rcc.state
                
                # QEI headroom based on coherence
                coherence = getattr(state, 'coherence_gamma', 0.5)
                metrics['qei_headroom'] = float(coherence - 0.3)  # Margin above threshold
                
                # Conditioning based on entropy
                entropy = getattr(state, 'entropy', 1.0)
                metrics['total_conditioning'] = float(max(1.0, entropy * 10.0))
                
                # Phase indicator based on proper time
                proper_time = getattr(state, 'proper_time', 0.0)
                metrics['phase_indicator'] = float(proper_time % (2.0 * np.pi))
            
            # Try to get MERA-based metrics
            if hasattr(self.rcc, 'mera'):
                try:
                    # Simple tensor norm as conditioning proxy
                    tensor_norms = [torch.norm(t).item() for t in self.rcc.mera.tensors]
                    avg_norm = np.mean(tensor_norms) if tensor_norms else 1.0
                    metrics['total_conditioning'] = float(avg_norm * 50.0)
                except Exception:
                    pass
            
            return metrics
            
        except Exception:
            # Safe fallback
            return {
                'qei_headroom': 0.0,
                'total_conditioning': 1.0, 
                'phase_indicator': 0.0,
            }