# Physics Primitives and Recursive Loop Services

This document describes the new physics primitives and recursive loop services added to the gyroscopic stabilized quantum computing system.

## Overview

The implementation adds 7 new service modules organized into 5 packages:

- **physics/**: Advanced wormhole dynamics and quantum field theory
- **geometry/**: Tensor network and spacetime mapping utilities  
- **architectures/**: Quantum control system patterns
- **control/**: Recursive and multi-scale dynamics
- **integration/**: Glue layer for seamless RCC integration

## Quick Start

```python
from pkgs.engine_runtime import RecursiveConformalComputing
from architectures import BronteinCube
from integration import make_mt_wormhole, sense_fn_factory, compute_fn_factory, actuate_fn_factory

# Setup
rcc = RecursiveConformalComputing(r0=0.5, m0=1.0)
mt = make_mt_wormhole(rcc, r0=0.5)
sense = sense_fn_factory(rcc)
compute = compute_fn_factory(rcc, mt, spin_density=0.02)
actuate = actuate_fn_factory(rcc)

# Run continuous sense-compute-actuate cycles
brontein = BronteinCube(sense=sense, compute=compute, actuate=actuate, hz=50.0)
brontein.run(steps=500)
```

## Module Details

### 1. Morris-Thorne Wormhole (physics/wormhole_mt.py)

- **MTWormhole class**: Configurable wormhole geometry with shape function b(r)
- **Flare-out condition checking**: Validates b(r₀)=r₀ and b'(r₀)<1
- **Metric computations**: g_tt, g_rr factors for spacetime geometry
- **Geodesic dynamics**: Effective potential and radial equations

### 2. Einstein-Cartan Torsion (physics/torsion_ec.py)

- **torsion_spring_force()**: EC-inspired torsion coupling with exponential profile
- **torsion_effective_stiffness()**: Throat-scale spring constant calculation

### 3. Gamma-Coherence Feedback (physics/coherence_gamma.py)

- **coherence_mass_coupling()**: m_eff = m₀(1 + kγ) effective mass computation
- **gamma_feedback_step()**: ODE integration for dγ/dt = -η·m_eff·∂τ·γ

### 4. MERA Radial Mapping (geometry/mera_radial_map.py)

- **calibrate_layer_to_radius()**: Maps MERA layers to radial coordinates
- **map_radius_to_layer()**: Inverse mapping from radius to nearest layer
- Supports logarithmic and affine spacing schemes

### 5. Brontein Cube (architectures/brontein_cube.py)

- **BronteinCube class**: Sense-compute-actuate loop with configurable frequency
- Injection-based design for custom sensor, compute, and actuate functions
- Built-in timing control and pacing for real-time operation

### 6. Recursive Controller (control/recursive_loop.py)

- **RecursiveController class**: Multi-scale recursive loops with depth control
- Fractal control structure with residual feedback between scales
- Configurable step functions and maximum recursion depth

### 7. Integration Layer (integration/hooks_core.py)

- **Factory functions**: Create sensor, compute, and actuate interfaces
- **RCC integration**: Seamless connection to RecursiveConformalComputing
- **Real-time metrics**: Live system monitoring via RealTimeGaugeMetrics

## Testing

Run the demonstration script:
```bash
python3 examples/physics_primitives_demo.py
```

Verify installation:
```bash
make test  # Existing functionality
python3 -c "from physics import MTWormhole; print('✓ Physics primitives ready')"
```

## Integration Points

The new services integrate with existing RCC components:

- **MERA tensors**: Layer scaling based on wormhole dynamics
- **Gauge fields**: Torsion effects via Einstein-Cartan coupling  
- **Coherence gamma**: Dynamic effective mass feedback
- **Bond dimensions**: Radial-dependent scaling policies
- **Real-time metrics**: Live QEI headroom and conditioning monitoring

All modules are designed to be optional and fail-safe, with no impact on existing functionality when not used.