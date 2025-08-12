# Contextuality Diagnostics Module

This module provides contextuality detection using the Kochen-Specker theorem and Lovász theta function, with graceful fallbacks for when advanced computations fail.

## Features

- **Kochen-Specker Violation Detection**: Measures contextuality via expectation value variance across overlapping measurement contexts
- **Lovász Theta Function**: Optimal SDP bound computation with eigenvalue fallback when cvxpy is unavailable
- **Contextuality Graph Construction**: Builds correlation graphs from quantum state measurements
- **Robust Error Handling**: Graceful fallbacks ensure stability even when advanced computations fail
- **Optional Dependencies**: Works with or without cvxpy for SDP optimization

## Installation

The module requires only basic dependencies that are already part of the main project:

```bash
pip install torch numpy
```

For enhanced SDP functionality (optional):

```bash
pip install cvxpy
```

## Basic Usage

### Importing the Module

```python
from diagnostics.contextuality import contextuality_certificate

# Or import individual functions
from diagnostics.contextuality import (
    kochen_specker_violation,
    lovasz_theta_bound,
    contextuality_graph_from_state
)
```

### Quick Start

```python
import torch
from diagnostics.contextuality import contextuality_certificate

# Create a quantum state
psi = torch.randn(16, dtype=torch.complex64)
psi = psi / torch.norm(psi)  # normalize

# Get contextuality analysis
result = contextuality_certificate(psi)
print(result)
# Output: {
#     'ks_violation': 0.542,
#     'lovasz_theta': 4.873,
#     'contextuality_strength': 2.639,
#     'graph_edges': 12.0
# }
```

## Integration with RCC System

The module is designed to integrate seamlessly with the existing RecursiveConformalComputing system:

```python
from pkgs.engine_runtime import RecursiveConformalComputing
from pkgs.engine_runtime.recorder import SimpleRecorder
from diagnostics.contextuality import contextuality_certificate

# Set up RCC with recorder
recorder = SimpleRecorder(enabled=True)
rcc = RecursiveConformalComputing(params={"recorder": recorder})

# Optional contextuality monitoring in main loop
for step in range(num_steps):
    # Your existing evolution logic...
    
    # Add contextuality monitoring
    if rcc.recorder and hasattr(rcc, 'psi_clock'):
        ctx_data = contextuality_certificate(rcc.psi_clock)
        rcc.recorder.log({
            f"ctx_{k}": v for k, v in ctx_data.items()
        })
```

## API Reference

### `kochen_specker_violation(psi: torch.Tensor) -> float`

Computes a crude KS contextuality measure using variance in expectation values across overlapping contexts.

**Parameters:**
- `psi`: Quantum state as a 1D tensor (state vector) or 2D tensor (density matrix)

**Returns:**
- `float`: Contextuality violation measure (higher values indicate more contextuality)

**Graceful Fallbacks:**
- Returns 0.0 if computation fails
- Handles small dimensions (< 3) by returning 0.0
- Robust against NaN/Inf values

### `lovasz_theta_bound(adj_matrix: np.ndarray) -> float`

Computes the Lovász theta function via SDP relaxation with eigenvalue fallback.

**Parameters:**
- `adj_matrix`: Adjacency matrix of the contextuality graph

**Returns:**
- `float`: Lovász theta bound value

**Graceful Fallbacks:**
- Uses SDP optimization if cvxpy is available
- Falls back to eigenvalue bound if SDP fails or cvxpy unavailable
- Returns 1.0 as safe fallback for any computation errors

### `contextuality_graph_from_state(psi: torch.Tensor, threshold: float = 0.1) -> np.ndarray`

Builds a contextuality graph from quantum state correlations.

**Parameters:**
- `psi`: Quantum state tensor
- `threshold`: Correlation threshold for edge creation (default: 0.1)

**Returns:**
- `np.ndarray`: Symmetric adjacency matrix representing contextuality correlations

**Graceful Fallbacks:**
- Returns 2x2 identity for small dimensions or errors
- Handles edge cases with minimal graphs

### `contextuality_certificate(psi: torch.Tensor) -> Dict[str, float]`

Combined contextuality analysis providing a comprehensive certificate.

**Parameters:**
- `psi`: Quantum state tensor

**Returns:**
- `Dict[str, float]`: Dictionary containing:
  - `ks_violation`: Kochen-Specker violation measure
  - `lovasz_theta`: Lovász theta bound
  - `contextuality_strength`: Combined contextuality strength metric
  - `graph_edges`: Number of edges in contextuality graph

**Graceful Fallbacks:**
- Returns safe default values if any computation fails
- All returned values are guaranteed to be finite numbers

## Implementation Details

### Kochen-Specker Observable Sets

The module constructs different observable sets based on state dimension:

- **2-qubit systems (dim=4)**: Uses Pauli-like observables (X₁, Z₁, X₂, Z₂, X₁⊗X₂, Z₁⊗Z₂)
- **Higher dimensions**: Uses random Hermitian observables with fixed seed for reproducibility
- **Small dimensions (< 3)**: Returns empty set to avoid meaningless measurements

### Lovász Theta Computation

The implementation attempts SDP optimization first, then falls back:

1. **SDP Method**: Uses cvxpy with positive semidefinite constraints
2. **Eigenvalue Fallback**: Computes n/λmax where λmax is largest Laplacian eigenvalue
3. **Safe Fallback**: Returns 1.0 if all methods fail

### Contextuality Graph Construction

Edges are created between measurement contexts based on quantum correlations:

- Correlation measure: |Tr(ρ[A₁, A₂])| where [A₁, A₂] is the commutator
- Configurable threshold for edge creation
- Always produces symmetric graphs with no self-loops

## Error Handling Philosophy

The module follows a "graceful degradation" philosophy:

1. **No Crashes**: All functions are guaranteed to return valid values
2. **Meaningful Defaults**: Fallback values represent "no contextuality detected"
3. **Warning Messages**: Users are informed when fallbacks are triggered
4. **Optional Dependencies**: Core functionality works without cvxpy

## Examples

### Basic Contextuality Analysis

```python
import torch
import numpy as np
from diagnostics.contextuality import contextuality_certificate

# Create a Bell state
psi_bell = torch.tensor([1.0, 0.0, 0.0, 1.0], dtype=torch.complex64)
psi_bell = psi_bell / torch.norm(psi_bell)

result = contextuality_certificate(psi_bell)
print(f"Bell state contextuality: {result}")
```

### Comparing Different States

```python
# Mixed state (low contextuality)
psi_mixed = torch.ones(4, dtype=torch.complex64) / 2.0

# Random pure state (variable contextuality)  
psi_random = torch.randn(4, dtype=torch.complex64)
psi_random = psi_random / torch.norm(psi_random)

for name, state in [("Mixed", psi_mixed), ("Random", psi_random)]:
    cert = contextuality_certificate(state)
    print(f"{name}: KS={cert['ks_violation']:.3f}, θ={cert['lovasz_theta']:.3f}")
```

### Monitoring During Evolution

```python
# In your simulation loop
contextuality_history = []

for step in range(100):
    # Evolve your quantum state...
    
    # Monitor contextuality
    cert = contextuality_certificate(psi)
    contextuality_history.append(cert['contextuality_strength'])
    
    if step % 10 == 0:
        print(f"Step {step}: Contextuality strength = {cert['contextuality_strength']:.3f}")
```

## Testing

Run the comprehensive test suite:

```bash
python diagnostics/test_contextuality.py
```

Run the integration example:

```bash
python diagnostics/integration_example.py
```

## Performance Notes

- **Lightweight**: Core computations are efficient for small to medium quantum systems
- **Scalable**: Graceful performance degradation for large systems
- **Cache-Friendly**: Observable construction uses deterministic seeding
- **Memory Efficient**: Minimal memory overhead in main simulation loops

## License

This module is part of the Gyroscopic Stabilized Quantum Engine project and follows the same licensing terms.