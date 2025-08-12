"""CLI entrypoint for the quantum computation engine."""

import argparse
import logging
import yaml
import sys
from pathlib import Path
from typing import Dict, Any

import sys
import os
# Add the quantum-core directory to Python path for absolute imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from apps.engine.engine_service import EngineService
from pkgs.engine_runtime.schemas import InitRequest, StepRequest

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        logger.warning(f"Config file not found: {config_path}")
        return {}
    except yaml.YAMLError as e:
        logger.error(f"Error parsing config file: {e}")
        sys.exit(1)

def run_simulation(config: Dict[str, Any], args: argparse.Namespace):
    """Run the main simulation loop."""
    # Create engine service
    engine = EngineService(config)
    
    # Initialize engine
    init_req = InitRequest(
        seed=args.seed,
        grid_size=args.grid_size,
        device=args.device,
        params=config.get('simulation_params', {})
    )
    
    init_result = engine.init(init_req)
    logger.info(f"Engine initialized: {init_result}")
    
    # Run simulation steps
    try:
        for step in range(args.steps):
            # Compute time and radius for this step
            dt = config.get('time_step', 0.01)
            r = config.get('initial_radius', 0.5) + step * config.get('radius_increment', 0.001)
            
            # Execute step
            step_req = StepRequest(
                dt=dt,
                r=r,
                do_contract=args.contract
            )
            
            result = engine.step(step_req)
            
            # Log progress
            if step % args.log_interval == 0:
                logger.info(
                    f"Step {step}/{args.steps}: "
                    f"K={result.K}, ANE={result.ane_smear:.2e}, "
                    f"Guard={'OK' if result.guard_ok else 'FAIL'}"
                )
                
                # Print snapshot every 100 steps
                if step % 100 == 0 and step > 0:
                    snapshot = engine.snapshot()
                    logger.info(f"State snapshot: {snapshot['state']}")
            
            # Early termination on guard failure if requested
            if not result.guard_ok and args.strict:
                logger.error(f"QEI guard failed at step {step}, terminating")
                break
                
        # Export results
        output_path = args.output or f"quantum_simulation_{args.seed}"
        export_path = engine.export_logs(args.format, output_path)
        logger.info(f"Simulation completed. Results exported to: {export_path}")
        
        # Final snapshot
        final_snapshot = engine.snapshot()
        logger.info(f"Final state: {final_snapshot}")
        
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise
    finally:
        # Cleanup
        logger.info("Simulation finished")

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Quantum Computation Engine CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/default.yaml',
        help='Path to configuration file'
    )
    
    # Engine parameters
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=0,
        help='Random seed for deterministic behavior'
    )
    
    parser.add_argument(
        '--grid-size', '-g',
        type=int,
        default=32,
        help='Size of the computational grid'
    )
    
    parser.add_argument(
        '--device', '-d',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Computation device'
    )
    
    # Simulation parameters
    parser.add_argument(
        '--steps', '-n',
        type=int,
        default=1000,
        help='Number of simulation steps'
    )
    
    parser.add_argument(
        '--contract',
        action='store_true',
        default=True,
        help='Enable contractor optimization'
    )
    
    parser.add_argument(
        '--no-contract',
        dest='contract',
        action='store_false',
        help='Disable contractor optimization'
    )
    
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Terminate on QEI guard failures'
    )
    
    # Output parameters
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file path prefix'
    )
    
    parser.add_argument(
        '--format', '-f',
        type=str,
        default='jsonl',
        choices=['csv', 'jsonl', 'parquet'],
        help='Output format for logs'
    )
    
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        help='Interval for progress logging'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    config = load_config(args.config)
    config['log_level'] = args.log_level
    
    logger.info(f"Starting quantum computation engine with args: {args}")
    logger.info(f"Configuration: {config}")
    
    # Run simulation
    run_simulation(config, args)

if __name__ == '__main__':
    main()