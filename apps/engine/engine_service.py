"""
Engine service wrapper providing clean API for the gyroscopic stabilized system.

This service wraps the RecursiveConformalComputing and PrecisionTraversalContractor
classes to provide a clean, high-level API for initialization, stepping, and state management.
"""
import logging
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from pkgs.engine_runtime import RecursiveConformalComputing, PrecisionTraversalContractor, SimpleRecorder

logger = logging.getLogger('EngineService')


@dataclass
class InitRequest:
    """Request parameters for engine initialization."""
    r0: float = 0.5
    m0: float = 1.0
    enable_recorder: bool = True
    params: Optional[Dict] = None


@dataclass
class StepRequest:
    """Request parameters for a single step."""
    dt: float
    r: float
    do_contract: bool = True


@dataclass
class StepResult:
    """Result from a single step operation."""
    success: bool
    ane_smear: float
    spectral_dim: Optional[float]
    coherence_gamma: float
    proper_time: float
    entropy: float
    message: Optional[str] = None


class EngineService:
    """High-level service wrapper for the gyroscopic stabilized quantum engine."""
    
    def __init__(self, cfg: Dict[str, Any]):
        """Initialize the engine service with configuration."""
        self.cfg = cfg
        self.rcc: Optional[RecursiveConformalComputing] = None
        self.contractor: Optional[PrecisionTraversalContractor] = None
        self.recorder: Optional[SimpleRecorder] = None
        self._initialized = False
        self._step_count = 0
        
        logger.info("EngineService created with configuration")

    def init(self, req: InitRequest) -> Dict[str, Any]:
        """Initialize the engine with specified parameters."""
        try:
            # Create recorder if enabled
            if req.enable_recorder:
                self.recorder = SimpleRecorder(enabled=True)
                if 'git_commit' in self.cfg:
                    self.recorder.set_metadata(
                        git_commit=self.cfg['git_commit'],
                        config_hash=str(hash(str(self.cfg)))
                    )
            
            # Prepare RCC parameters
            rcc_params = req.params or {}
            if self.recorder:
                rcc_params['recorder'] = self.recorder
            
            # Merge configuration parameters
            for key in ['d_H', 'C_RT', 'kappa', 'sigma_u', 'mu0', 'L_rg', 'd_boundary']:
                if key in self.cfg:
                    rcc_params[key] = self.cfg[key]
            
            # Initialize RCC
            self.rcc = RecursiveConformalComputing(
                r0=req.r0,
                m0=req.m0,
                params=rcc_params
            )
            
            # Initialize contractor
            self.contractor = PrecisionTraversalContractor(self.rcc)
            
            # Take baseline snapshot
            baseline_snapshot = self.rcc.snapshot()
            
            self._initialized = True
            self._step_count = 0
            
            logger.info("Engine initialized successfully")
            
            return {
                'success': True,
                'baseline_snapshot': baseline_snapshot,
                'message': 'Engine initialized successfully'
            }
            
        except Exception as e:
            logger.error(f"Engine initialization failed: {e}")
            return {
                'success': False,
                'message': f'Initialization failed: {str(e)}'
            }

    def step(self, req: StepRequest) -> StepResult:
        """Execute a single step of the quantum evolution."""
        if not self._initialized:
            return StepResult(
                success=False,
                ane_smear=0.0,
                spectral_dim=None,
                coherence_gamma=0.0,
                proper_time=0.0,
                entropy=0.0,
                message="Engine not initialized"
            )
        
        try:
            # Execute contractor step
            result = self.contractor.step(
                dt=req.dt,
                r=req.r,
                do_contract=req.do_contract
            )
            
            self._step_count += 1
            
            # Log step completion
            if self._step_count % 100 == 0:
                logger.info(f"Completed step {self._step_count}, ANE={result['ane_smear']:.3e}")
            
            return StepResult(
                success=True,
                ane_smear=result['ane_smear'],
                spectral_dim=result['spectral_dim'],
                coherence_gamma=self.rcc.state.coherence_gamma,
                proper_time=self.rcc.state.proper_time,
                entropy=self.rcc.state.entropy,
                message=f"Step {self._step_count} completed"
            )
            
        except Exception as e:
            logger.error(f"Step execution failed: {e}")
            return StepResult(
                success=False,
                ane_smear=0.0,
                spectral_dim=None,
                coherence_gamma=self.rcc.state.coherence_gamma if self.rcc else 0.0,
                proper_time=self.rcc.state.proper_time if self.rcc else 0.0,
                entropy=self.rcc.state.entropy if self.rcc else 0.0,
                message=f"Step failed: {str(e)}"
            )

    def snapshot(self) -> Dict[str, Any]:
        """Return summary snapshot of current system state."""
        if not self._initialized:
            return {
                'initialized': False,
                'message': 'Engine not initialized'
            }
        
        try:
            # Get RCC snapshot
            rcc_snapshot = self.rcc.snapshot()
            
            # Add service-level metadata
            snapshot = {
                'initialized': True,
                'step_count': self._step_count,
                'timestamp': time.time(),
                **rcc_snapshot
            }
            
            # Add recorder summary if available
            if self.recorder:
                snapshot['recorder_summary'] = self.recorder.get_summary()
            
            return snapshot
            
        except Exception as e:
            logger.error(f"Snapshot creation failed: {e}")
            return {
                'initialized': True,
                'error': str(e),
                'message': 'Snapshot creation failed'
            }

    def reset(self) -> Dict[str, Any]:
        """Reset the engine to baseline state."""
        if not self._initialized:
            return {
                'success': False,
                'message': 'Engine not initialized'
            }
        
        try:
            self.rcc.reset_to_baseline()
            self._step_count = 0
            
            # Clear recorder if available
            if self.recorder:
                self.recorder.clear()
            
            logger.info("Engine reset to baseline state")
            
            return {
                'success': True,
                'message': 'Engine reset successfully'
            }
            
        except Exception as e:
            logger.error(f"Engine reset failed: {e}")
            return {
                'success': False,
                'message': f'Reset failed: {str(e)}'
            }

    def save_recordings(self, base_path: str) -> Dict[str, Any]:
        """Save recorder data to specified path."""
        if not self.recorder:
            return {
                'success': False,
                'message': 'No recorder available'
            }
        
        try:
            self.recorder.dump_all_formats(base_path)
            summary = self.recorder.get_summary()
            
            return {
                'success': True,
                'message': f'Recordings saved to {base_path}',
                'summary': summary
            }
            
        except Exception as e:
            logger.error(f"Recording save failed: {e}")
            return {
                'success': False,
                'message': f'Save failed: {str(e)}'
            }

    def shutdown(self) -> Dict[str, Any]:
        """Gracefully shutdown the engine service."""
        try:
            # Save final recordings if available
            if self.recorder and self.recorder.rows:
                self.recorder.dump_all_formats("./final_recordings")
            
            # Reset state
            self._initialized = False
            self.rcc = None
            self.contractor = None
            self.recorder = None
            
            logger.info("Engine service shutdown completed")
            
            return {
                'success': True,
                'message': 'Engine shutdown completed'
            }
            
        except Exception as e:
            logger.error(f"Engine shutdown failed: {e}")
            return {
                'success': False,
                'message': f'Shutdown failed: {str(e)}'
            }