# Quantum Core - Modular Quantum Computation Engine

A restructured, modular quantum computation engine with gyroscopic stabilization, implementing clean separation between pure physics kernels, runtime orchestration, and control logic.

## Architecture Overview

The quantum computation engine has been restructured into a modular monorepo with the following architecture:

```
quantum-core/
├─ apps/                      # Service applications
│  ├─ engine/                 # Physics step loop service  
│  ├─ controller/             # Control policies and parameter management
│  ├─ probes/                 # Read-only metrics computation
│  └─ qec/                    # Future QEC integration (stubs)
├─ pkgs/                      # Core packages
│  ├─ core_physics/           # Pure computational kernels
│  ├─ engine_runtime/         # Orchestration and state management
│  ├─ observability/          # Logging, metrics, and monitoring
│  └─ adapters/               # Future extension points
├─ configs/                   # Configuration management
├─ tests/                     # Comprehensive test suite
└─ scripts/                   # Utility scripts
```

## Key Features

### ✅ Preserved Functionality
- All existing physics code works without modification
- Complete RecursiveConformalComputing implementation
- PrecisionTraversalContractor with QEI guard
- MERA spacetime tensor networks
- Gauge field dynamics and Alena soul modifications

### ✅ Clean Boundaries
- **Core Physics**: Pure computational kernels (no side effects)
- **Engine Runtime**: Orchestration and state management
- **Observability**: Centralized logging, metrics, and monitoring
- **Service Layer**: High-level APIs with Pydantic schemas

### ✅ Extensibility Enabled
- Plugin architecture for future modules
- Adapter pattern for Complex Langevin, LGT cube operations
- QEC integration points (surface codes, tensor network decoders)
- Event bus for loose coupling

### ✅ Deterministic Configuration
- Centralized seed management
- YAML-based configuration with profiles
- Environment-specific overrides (dev, GPU, research, CI)

## Quick Start

### Installation

```bash
# Basic installation
pip install -e .

# With GPU support
pip install -e ".[gpu]"

# Full development environment
pip install -e ".[all]"
```

### Basic Usage

```python
from quantum_core import EngineService
from quantum_core.pkgs.engine_runtime.schemas import InitRequest, StepRequest

# Create engine service
engine = EngineService()

# Initialize with configuration
init_req = InitRequest(seed=42, grid_size=32, device="auto")
result = engine.init(init_req)

# Execute physics steps
for i in range(100):
    step_req = StepRequest(dt=0.01, r=0.5 + 0.001*i, do_contract=True)
    result = engine.step(step_req)
    print(f"Step {i}: K={result.K}, ANE={result.ane_smear:.2e}")

# Export results
engine.export_logs("jsonl", "results/simulation")
```

### CLI Usage

```bash
# Run with default configuration
quantum-engine --steps 1000 --output results/run1

# Use GPU profile
quantum-engine --config configs/profiles/gpu.yaml --steps 10000

# Development mode with debug logging
quantum-engine --config configs/profiles/dev.yaml --log-level DEBUG

# Custom parameters
quantum-engine --seed 123 --grid-size 64 --format parquet
```

## Configuration

The engine uses YAML-based configuration with support for profiles:

```yaml
# configs/default.yaml
global_seed: 0
log_level: "INFO"
device: "auto"

default_params:
  sigma_u: 0.004
  chi_max: 16
  beta_vec: [0.15, 0.05, 0.01]
  
simulation_params:
  steps: 1000
  contract_enabled: true
```

### Configuration Profiles

- **`configs/profiles/dev.yaml`**: Development settings (fast, CPU-only)
- **`configs/profiles/gpu.yaml`**: GPU-optimized settings  
- **`configs/profiles/research.yaml`**: High-precision research settings
- **`configs/profiles/ci.yaml`**: Continuous integration settings

## Core Components

### Pure Physics Kernels (`pkgs/core_physics/`)

- **Lattice**: Bond graphs and tesseract lattice construction
- **Tensors**: MERA spacetime networks with drift guards
- **Fields**: Gauge fields, Alena soul modifications, UV/IR regulators
- **Control**: Gyroscope feeler, signature normalization, Jacobian hypercube
- **TimeOps**: Time evolution operators and recompression gates
- **Utils**: Mathematical utilities and helper functions

### Engine Runtime (`pkgs/engine_runtime/`)

- **RCC**: RecursiveConformalComputing orchestration class
- **Contractor**: PrecisionTraversalContractor with QEI guards
- **Recorder**: Enhanced recording (CSV/JSONL/Parquet support)
- **Schemas**: Pydantic models for service APIs

### Observability (`pkgs/observability/`)

- **Logging**: Structured logging configuration
- **Metrics**: Centralized metrics collection and seed management
- **Events**: Event bus framework for component communication

## Physics Features

### Quantum Energy Inequality (QEI) Guard
- Dynamic bound computation with configurable kernels
- Safety margin enforcement with emergency damping
- Multiple smearing functions (Gaussian, Lorentzian, Fermi-Dirac)

### MERA Tensor Networks
- Multi-layer spacetime networks with bond dimension control
- Phase imprinting from gauge field dynamics
- Drift guard with automatic phase reset

### Spectral Dimension Estimation
- Interface graph Laplacian analysis
- Low-energy eigenvalue density fitting
- Integration with RG planner for Planck scale factors

### Averaged Null Energy (ANE)
- Gaussian-weighted temporal smearing
- History management with configurable retention
- Stress-energy computation with Planck scale threading

## Future Extensions

The modular architecture enables easy extension with new physics modules:

### Phase 2: Complex Langevin Integration
```python
from quantum_core.pkgs.adapters import ComplexLangevinAdapter
adapter = ComplexLangevinAdapter(coupling=1.0)
# Integration with existing gauge field dynamics
```

### Phase 3: Lattice Gauge Theory
```python
from quantum_core.pkgs.adapters import LGTCube
lgt = LGTCube(lattice_size=16)
# Wilson loop computations on plaquettes
```

### Phase 4: Quantum Error Correction
```python
from quantum_core.apps.qec import SurfaceCode, TNDecoder
surface_code = SurfaceCode(distance=5)
decoder = TNDecoder(bond_dim=16)
# Integration with tensor network architecture
```

## Testing

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m "not slow"        # Skip slow tests
pytest -m "not gpu"         # Skip GPU tests
pytest tests/test_core_physics.py -v

# With coverage
pytest --cov=quantum_core --cov-report=html
```

## Development

### Code Quality
```bash
# Format code
black quantum-core/
isort quantum-core/

# Lint
flake8 quantum-core/
mypy quantum-core/
```

### Adding New Physics Modules

1. **Pure Physics**: Add to `pkgs/core_physics/`
2. **Orchestration**: Extend `pkgs/engine_runtime/`
3. **Services**: Add to `apps/`
4. **Extensions**: Use `pkgs/adapters/` for external integrations

## Performance

### Benchmarks
- **Single Step**: ~10ms (CPU), ~2ms (GPU)
- **1000 Steps**: ~10s (CPU), ~2s (GPU)
- **Memory Usage**: ~500MB baseline, scales with grid size

### Optimization Tips
- Use GPU profile for large simulations
- Adjust `sigma_u` for stability vs precision trade-off
- Tune `chi_max` for memory vs accuracy balance

## License

MIT License - see LICENSE file for details.

## Citation

If you use this code in research, please cite:

```bibtex
@software{quantum_core,
  title={Quantum Core: Modular Quantum Computation Engine},
  author={Quantum Core Team},
  year={2024},
  url={https://github.com/quantum-core/quantum-core}
}
```

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## Changelog

### v0.1.0 (2024-01-XX)
- Initial modular architecture implementation
- Complete migration from monolithic to modular structure
- Service layer with Pydantic schemas
- Enhanced configuration management
- Comprehensive test suite
- Modern Python packaging with pyproject.toml