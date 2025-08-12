"""Engine service that wraps RCC and PrecisionTraversalContractor with clean API."""

import logging
from typing import Dict, Optional
import sys
import os
# Add the quantum-core directory to Python path for absolute imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from pkgs.engine_runtime import RecursiveConformalComputing, PrecisionTraversalContractor, SimpleRecorder
from pkgs.engine_runtime.schemas import InitRequest, StepRequest, StepResult
from pkgs.observability import setup_logging, MetricsCollector, SeedManager

logger = logging.getLogger(__name__)

class EngineService:
    """High-level service interface for the quantum computation engine."""
    
    def __init__(self, cfg: Optional[Dict] = None):
        self.cfg = cfg or {}
        self.rcc: Optional[RecursiveConformalComputing] = None
        self.contractor: Optional[PrecisionTraversalContractor] = None
        self.recorder: Optional[SimpleRecorder] = None
        self.metrics = MetricsCollector()
        self.seed_manager = SeedManager(self.cfg.get('global_seed', 0))
        
        # Setup logging
        log_level = self.cfg.get('log_level', 'INFO')
        setup_logging(log_level)
        logger.info("EngineService initialized")
    
    def init(self, req: InitRequest) -> Dict:
        """Initialize the quantum computation engine."""
        logger.info(f"Initializing engine with seed={req.seed}, grid_size={req.grid_size}")
        
        # Set seed
        self.seed_manager = SeedManager(req.seed)
        
        # Initialize recorder
        self.recorder = SimpleRecorder(enabled=True)
        
        # Setup parameters
        params = {
            **self.cfg.get('default_params', {}),
            **req.params,
            'recorder': self.recorder
        }
        
        # Initialize RCC
        self.rcc = RecursiveConformalComputing(
            r0=params.get('r0', 0.5),
            m0=params.get('m0', 1.0),
            params=params
        )
        
        # Initialize contractor
        self.contractor = PrecisionTraversalContractor(self.rcc)
        
        # Take baseline snapshot
        self.rcc.zero_field_baseline()
        baseline_ane = self.rcc.flat_baseline_check(dt=0.01, steps=64)
        
        init_result = {
            "status": "initialized",
            "baseline_ane": baseline_ane,
            "seed": req.seed,
            "grid_size": req.grid_size,
            "device": str(self.rcc.gauge_field.device),
            "layers": self.rcc.mera.layers,
            "vertices": self.rcc.graph.V
        }
        
        logger.info(f"Engine initialized successfully: {init_result}")
        return init_result
    
    def step(self, req: StepRequest) -> StepResult:
        """Execute a single physics step."""
        if not self.contractor:
            raise RuntimeError("Engine not initialized. Call init() first.")
        
        logger.debug(f"Executing step: dt={req.dt}, r={req.r}, do_contract={req.do_contract}")
        
        # Start timing
        self.metrics.start_timer("step_duration")
        
        try:
            # Execute step
            result = self.contractor.step(req.dt, req.r, req.do_contract)
            
            # Collect additional metrics
            metrics = {
                "proper_time": self.rcc.state.proper_time,
                "coherence_gamma": self.rcc.state.coherence_gamma,
                "entropy": self.rcc.state.entropy,
                "decoherence_state": self.rcc.state.decoherence_state.value
            }
            
            # Stop timing
            step_duration = self.metrics.stop_timer("step_duration")
            metrics["step_duration"] = step_duration
            
            # Increment step counter
            self.metrics.increment_counter("total_steps")
            
            # Create response
            step_result = StepResult(
                K=result["K"],
                ane_smear=result["ane_smear"],
                guard_ok=result["guard_ok"],
                spectral_dim=result.get("spectral_dim"),
                metrics=metrics
            )
            
            logger.debug(f"Step completed: K={result['K']}, ANE={result['ane_smear']:.2e}")
            return step_result
            
        except Exception as e:
            logger.error(f"Error during step execution: {e}")
            self.metrics.stop_timer("step_duration")
            raise
    
    def snapshot(self) -> Dict:
        """Return summary snapshot of current state."""
        if not self.rcc:
            return {"status": "not_initialized"}
        
        # Get recent metrics
        recent_logs = self.recorder.get_recent(10) if self.recorder else []
        
        # Get tensor energy
        delta_energy = self.rcc.mera.delta_tensor_energy()
        
        # Get RG planner state
        rg_state = {
            "c_ref": self.rcc.rg_planner.c_ref,
            "prev_c_rel": self.rcc.rg_planner.prev_c_rel,
            "last_s_Pl": self.rcc._last_s_Pl
        }
        
        snapshot = {
            "status": "active",
            "state": {
                "proper_time": self.rcc.state.proper_time,
                "coherence_gamma": self.rcc.state.coherence_gamma,
                "entropy": self.rcc.state.entropy,
                "decoherence_state": self.rcc.state.decoherence_state.value,
                "delta_tensor_energy": delta_energy
            },
            "rg_planner": rg_state,
            "metrics_summary": self.metrics.summary_stats(),
            "recent_logs": recent_logs[-5:],  # Last 5 entries
            "tensor_layers": self.rcc.mera.target_dims,
            "graph_vertices": self.rcc.graph.V,
            "spectral_dim_history": self.rcc._dH_history[-10:]  # Last 10 estimates
        }
        
        return snapshot
    
    def export_logs(self, format: str = "jsonl", path: str = "logs/quantum_run") -> str:
        """Export recorded logs in specified format."""
        if not self.recorder:
            raise RuntimeError("No recorder available")
        
        if format == "csv":
            full_path = f"{path}.csv"
            self.recorder.dump_csv(full_path)
        elif format == "jsonl":
            full_path = f"{path}.jsonl"
            self.recorder.dump_jsonl(full_path)
        elif format == "parquet":
            full_path = f"{path}.parquet"
            self.recorder.dump_parquet(full_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Logs exported to: {full_path}")
        return full_path
    
    def reset(self):
        """Reset the engine to initial state."""
        if self.rcc:
            self.rcc.zero_field_baseline()
        if self.recorder:
            self.recorder.clear()
        self.metrics.reset()
        logger.info("Engine reset to initial state")