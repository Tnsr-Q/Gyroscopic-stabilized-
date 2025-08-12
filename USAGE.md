# Gyroscopic Stabilized Quantum Engine - Modular Architecture

This repository contains a modular quantum computing engine for gyroscopic stabilized systems, organized into clean package boundaries separating physics computations from orchestration logic.

## Architecture Overview

The codebase is organized into three main layers:

### Core Physics (`pkgs/core_physics/`)
Pure physics computations with no orchestration dependencies:

- **`lattice.py`** - BondGraph structures and tesseract lattice construction
- **`tensors.py`** - MERA spacetime networks and tensor operations
- **`fields.py`** - Gauge fields, Alena Soul (Υ), and UV/IR regulators  
- **`control.py`** - Control systems and Jacobian computations
- **`timeops.py`** - Time evolution operators and recompression gates
- **`utils.py`** - Mathematical utilities and RG planning
- **`common.py`** - Shared data structures and enums

### Engine Runtime (`pkgs/engine_runtime/`)
Orchestration and state management:

- **`rcc.py`** - RecursiveConformalComputing orchestrator
- **`contractor.py`** - PrecisionTraversalContractor for evolution
- **`recorder.py`** - Enhanced recording with CSV/JSONL/Parquet support

### Engine Service (`apps/engine/`)
High-level service wrapper and CLI:

- **`engine_service.py`** - Clean API wrapper with init/step/snapshot methods
- **`main.py`** - CLI runner with configuration loading and graceful shutdown

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python3 -c "from pkgs.core_physics import build_tesseract_lattice; print('✓ Installation successful')"
```

### Running Simulations

```bash
# Run with default configuration
cd apps/engine
python3 main.py

# Run with custom configuration  
python3 main.py --config ../../configs/my_config.yaml

# Enable verbose logging
python3 main.py --verbose
```

### Configuration

Edit `configs/default.yaml` to customize simulation parameters:

```yaml
# Core physics parameters
physics:
  d_H: 3.12                  # Spectral dimension
  C_RT: 91.64               # RT constant  
  kappa: 0.015              # Alena Soul coupling

# Simulation settings
simulation:
  dt: 0.01                  # Time step
  r_start: 0.1              # Start radius
  r_end: 2.0                # End radius
  max_steps: 1000           # Max iterations
```

## Programmatic Usage

### Basic Engine Usage

```python
from apps.engine import EngineService, InitRequest, StepRequest

# Create engine with configuration
config = {'d_H': 3.12, 'C_RT': 91.64}
engine = EngineService(config)

# Initialize
init_req = InitRequest(r0=0.5, m0=1.0, enable_recorder=True)
result = engine.init(init_req)

# Execute steps
for i in range(100):
    step_req = StepRequest(dt=0.01, r=0.1 + i*0.01)
    step_result = engine.step(step_req)
    
    if not step_result.success:
        break

# Get final snapshot
snapshot = engine.snapshot()
print(f"Final coherence: {snapshot['coherence_gamma']:.3f}")

# Save recordings
engine.save_recordings("./my_simulation")
```

### Direct Physics Usage

```python
from pkgs.core_physics import build_tesseract_lattice, MERASpacetime, TimeOperator
from pkgs.engine_runtime import RecursiveConformalComputing

# Build quantum lattice
graph = build_tesseract_lattice()
print(f"Created lattice: {graph.V} vertices, {len(graph.edges)} edges")

# Create MERA network  
mera = MERASpacetime(layers=6, bond_dim=4)
entropy = mera.compute_entanglement_entropy(region_size=8)

# Time evolution
time_op = TimeOperator(dim_clock=16)
decoherence = time_op.compute_decoherence(0.0, 1.0, psi)
```

## Output Formats

The engine supports multiple recording formats:

- **CSV** - Backward compatible tabular data
- **JSONL** - Structured event logging with metadata
- **Parquet** - Efficient columnar storage (requires pandas/pyarrow)
- **YAML** - Human-readable state snapshots

Files are automatically saved with timestamps and configuration metadata.

## Key Features

### Physics Preservation
- All original physics logic preserved without modification
- Deterministic computational kernels maintained
- Numerical algorithms unchanged

### Clean Architecture  
- Pure physics separated from orchestration
- Modular packages with clear boundaries
- Extensible design for future components

### Enhanced Recording
- Multiple output formats for different use cases
- Automatic metadata injection (timestamps, git commit, config hash)
- Backward compatibility with existing CSV workflows

### Robust Operation
- Graceful shutdown handling (SIGINT/SIGTERM)
- Comprehensive error handling and logging
- Automatic drift guards and stability monitoring

## Development

### Adding New Physics Components

1. Add pure physics classes to appropriate `pkgs/core_physics/` modules
2. Export in `pkgs/core_physics/__init__.py`
3. Use in orchestration via `pkgs/engine_runtime/`

### Extending the Service API

1. Add new request/response dataclasses to `engine_service.py`
2. Implement methods following existing patterns
3. Update CLI if needed for new functionality

### Testing

```bash
# Quick smoke test
python3 -c "
import sys; sys.path.append('.')
from pkgs.core_physics import build_tesseract_lattice
from pkgs.engine_runtime import RecursiveConformalComputing
rcc = RecursiveConformalComputing()
print('✓ All systems operational')
"

# Full integration test with short simulation
cd apps/engine
python3 main.py --config ../../configs/default.yaml --verbose
```

## Migration from Monolithic Code

The modular architecture maintains full backward compatibility. The original monolithic implementation in `README.md` is preserved as reference, while the new modular system provides:

- Better maintainability and testability
- Clear separation of concerns
- Extensibility for new physics phases
- Production-ready service interfaces

For any issues or questions, refer to the comprehensive logging output and error messages provided by the engine.