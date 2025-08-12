#!/usr/bin/env python3
"""
Main CLI entrypoint for the gyroscopic stabilized quantum engine.

Loads configuration, initializes the engine service, and runs the main
simulation loop with proper logging and graceful shutdown handling.
"""
import argparse
import signal
import sys
import time
import yaml
import logging
from pathlib import Path
from typing import Dict, Any

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from engine_service import EngineService, InitRequest, StepRequest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('engine.log')
    ]
)
logger = logging.getLogger('EngineMain')


class EngineRunner:
    """Main runner for the engine with graceful shutdown support."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        self.engine = EngineService(self.config)
        self.running = False
        self.shutdown_requested = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
            # Return default configuration
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'engine': {
                'r0': 0.5,
                'm0': 1.0,
                'enable_recorder': True
            },
            'physics': {
                'd_H': 3.12,
                'C_RT': 91.64,
                'kappa': 0.015,
                'sigma_u': 4e-3,
                'mu0': 1.0,
                'L_rg': 1.0,
                'd_boundary': 2
            },
            'simulation': {
                'dt': 0.01,
                'r_start': 0.1,
                'r_end': 2.0,
                'r_step': 0.01,
                'max_steps': 1000,
                'save_interval': 100
            },
            'output': {
                'recordings_path': './recordings/simulation',
                'snapshot_path': './snapshots'
            }
        }

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True

    def _save_snapshot(self, step: int):
        """Save system snapshot."""
        try:
            snapshot = self.engine.snapshot()
            snapshot_path = Path(self.config['output']['snapshot_path'])
            snapshot_path.mkdir(parents=True, exist_ok=True)
            
            snapshot_file = snapshot_path / f"snapshot_step_{step:06d}.yaml"
            with open(snapshot_file, 'w') as f:
                yaml.dump(snapshot, f, default_flow_style=False)
            
            logger.info(f"Saved snapshot to {snapshot_file}")
            
        except Exception as e:
            logger.error(f"Failed to save snapshot: {e}")

    def _save_recordings(self):
        """Save all recordings."""
        try:
            recordings_path = self.config['output']['recordings_path']
            result = self.engine.save_recordings(recordings_path)
            
            if result['success']:
                logger.info(f"Recordings saved: {result['message']}")
            else:
                logger.error(f"Failed to save recordings: {result['message']}")
                
        except Exception as e:
            logger.error(f"Failed to save recordings: {e}")

    def initialize(self) -> bool:
        """Initialize the engine."""
        try:
            # Create initialization request from config
            init_req = InitRequest(
                r0=self.config['engine']['r0'],
                m0=self.config['engine']['m0'],
                enable_recorder=self.config['engine']['enable_recorder'],
                params=self.config['physics']
            )
            
            # Initialize engine
            result = self.engine.init(init_req)
            
            if result['success']:
                logger.info(f"Engine initialized successfully: {result['message']}")
                return True
            else:
                logger.error(f"Engine initialization failed: {result['message']}")
                return False
                
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            return False

    def run_simulation(self):
        """Run the main simulation loop."""
        if not self.initialize():
            return False
        
        self.running = True
        step = 0
        
        sim_config = self.config['simulation']
        dt = sim_config['dt']
        r_start = sim_config['r_start']
        r_end = sim_config['r_end']
        r_step = sim_config['r_step']
        max_steps = sim_config['max_steps']
        save_interval = sim_config['save_interval']
        
        # Calculate r values
        r_values = []
        r = r_start
        while r <= r_end and len(r_values) < max_steps:
            r_values.append(r)
            r += r_step
        
        logger.info(f"Starting simulation with {len(r_values)} steps")
        start_time = time.time()
        
        try:
            for step, r in enumerate(r_values):
                if self.shutdown_requested:
                    logger.info("Shutdown requested, stopping simulation...")
                    break
                
                # Execute step
                step_req = StepRequest(dt=dt, r=r, do_contract=True)
                result = self.engine.step(step_req)
                
                if not result.success:
                    logger.error(f"Step {step} failed: {result.message}")
                    break
                
                # Log progress
                if step % 10 == 0:
                    logger.info(
                        f"Step {step:4d}/{len(r_values)}: r={r:.3f}, "
                        f"ANE={result.ane_smear:.2e}, γ={result.coherence_gamma:.3f}, "
                        f"τ={result.proper_time:.3f}"
                    )
                
                # Save snapshot periodically
                if step % save_interval == 0:
                    self._save_snapshot(step)
                
                # Check for early termination conditions
                if result.coherence_gamma < 0.01:
                    logger.warning("Coherence critically low, terminating simulation")
                    break
            
            elapsed_time = time.time() - start_time
            logger.info(f"Simulation completed in {elapsed_time:.2f} seconds")
            
            # Save final snapshot and recordings
            self._save_snapshot(step)
            self._save_recordings()
            
            return True
            
        except Exception as e:
            logger.error(f"Simulation error: {e}")
            return False
        
        finally:
            self.running = False

    def shutdown(self):
        """Gracefully shutdown the engine."""
        try:
            if self.running:
                logger.info("Shutting down engine...")
                self._save_recordings()
                
            result = self.engine.shutdown()
            if result['success']:
                logger.info("Engine shutdown completed")
            else:
                logger.error(f"Engine shutdown error: {result['message']}")
                
        except Exception as e:
            logger.error(f"Shutdown error: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Gyroscopic Stabilized Quantum Engine')
    parser.add_argument(
        '--config', '-c',
        default='configs/default.yaml',
        help='Path to configuration file (default: configs/default.yaml)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create runner and execute
    runner = EngineRunner(args.config)
    
    try:
        success = runner.run_simulation()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
    finally:
        runner.shutdown()


if __name__ == '__main__':
    main()