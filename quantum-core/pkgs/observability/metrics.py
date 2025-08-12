"""Metrics collection and seed management."""

import numpy as np
import torch
from typing import Dict, Any, Optional
import time

class SeedManager:
    """Centralized seed management for deterministic behavior."""
    
    def __init__(self, global_seed: int = 0):
        self.global_seed = global_seed
        self.component_seeds: Dict[str, int] = {}
        self._set_global_seed()
    
    def _set_global_seed(self):
        """Set global seeds for all random number generators."""
        np.random.seed(self.global_seed)
        torch.manual_seed(self.global_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.global_seed)
            torch.cuda.manual_seed_all(self.global_seed)
    
    def get_component_seed(self, component: str) -> int:
        """Get deterministic seed for a specific component."""
        if component not in self.component_seeds:
            # Generate deterministic seed based on component name
            import hashlib
            seed_str = f"{self.global_seed}_{component}"
            hash_obj = hashlib.md5(seed_str.encode())
            seed = int(hash_obj.hexdigest()[:8], 16) % (2**31)
            self.component_seeds[component] = seed
        return self.component_seeds[component]
    
    def reset_component(self, component: str):
        """Reset RNG state for a specific component."""
        seed = self.get_component_seed(component)
        rng = np.random.RandomState(seed)
        return rng

class MetricsCollector:
    """Centralized metrics collection and aggregation."""
    
    def __init__(self):
        self.metrics: Dict[str, Any] = {}
        self.counters: Dict[str, int] = {}
        self.timers: Dict[str, float] = {}
        self.start_times: Dict[str, float] = {}
    
    def set_metric(self, name: str, value: Any):
        """Set a metric value."""
        self.metrics[name] = value
    
    def increment_counter(self, name: str, delta: int = 1):
        """Increment a counter metric."""
        self.counters[name] = self.counters.get(name, 0) + delta
    
    def start_timer(self, name: str):
        """Start a timer."""
        self.start_times[name] = time.time()
    
    def stop_timer(self, name: str):
        """Stop a timer and record duration."""
        if name in self.start_times:
            duration = time.time() - self.start_times[name]
            self.timers[name] = duration
            del self.start_times[name]
            return duration
        return 0.0
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics."""
        return {
            "metrics": self.metrics.copy(),
            "counters": self.counters.copy(),
            "timers": self.timers.copy()
        }
    
    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()
        self.counters.clear()
        self.timers.clear()
        self.start_times.clear()
    
    def summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            "total_metrics": len(self.metrics),
            "total_counters": len(self.counters),
            "total_timers": len(self.timers),
            "counter_sum": sum(self.counters.values()),
            "timer_total": sum(self.timers.values())
        }