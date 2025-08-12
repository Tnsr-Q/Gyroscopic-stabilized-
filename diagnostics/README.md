# Quantum Diagnostics Module

This module provides diagnostic tools for quantum scarring, eigenstate thermalization hypothesis (ETH) testing, and Loschmidt echo measurements.

## Features

### Scar/ETH Metrics (`scar_metrics.py`)
Functions for detecting quantum scars and ETH violations:

- **`participation_ratio(psi)`**: Calculate PR = 1/sum |psi_i|^4 (basis-dependent; use clock basis)
- **`cheap_otoc_rate(time_op, psi, steps=8, dt=0.01)`**: Crude OTOC growth proxy using diagonal clock-H and a diagonal 'V'
- **`eth_deviation(energies, obs_vals, window=0.05)`**: ETH variance in a microcanonical window centered at median energy

### Echo Meters (`echo_meter.py`)
Loschmidt echo measurements for quantum chaos diagnostics:

- **`loschmidt_echo(time_op, psi, t=0.05, eps=1e-2)`**: L(t) = |<psi| U_0†(t) U_eps(t) |psi>|^2 with small H-perturbation
- **`echo_report(time_op, psi, t_list=(0.02, 0.05, 0.1))`**: Generate multiple echo measurements with decay rate

## Installation

The diagnostics module is included with the main project. Ensure you have the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from diagnostics.scar_metrics import participation_ratio, cheap_otoc_rate
from diagnostics.echo_meter import echo_report

# Assuming you have a TimeOperator and quantum state
pr = participation_ratio(psi_clock)
otoc_rate = cheap_otoc_rate(time_op, psi_clock, steps=6, dt=0.01)
echo_data = echo_report(time_op, psi_clock, t_list=(0.02, 0.05))
```

### Integration with Recorder

```python
from diagnostics.scar_metrics import participation_ratio, cheap_otoc_rate
from diagnostics.echo_meter import echo_report

if recorder:
    recorder.log({
        "scar_pr": participation_ratio(psi_clock),
        "otoc_rate": cheap_otoc_rate(time_op, psi_clock, steps=6, dt=0.01),
        **echo_report(time_op, psi_clock, t_list=(0.02, 0.05))
    })
```

## Function Details

### Participation Ratio
- **Purpose**: Detects localization in quantum scars (higher PR = more localized)
- **Range**: 1 ≤ PR ≤ dimension
- **Interpretation**: PR ≈ 1 suggests scarring/localization, PR ≈ dimension suggests thermalization

### OTOC Rate  
- **Purpose**: Measures information scrambling rate (quantum chaos indicator)
- **Interpretation**: Large positive values suggest chaotic dynamics, small values suggest integrability

### ETH Deviation
- **Purpose**: Tests eigenstate thermalization hypothesis violations
- **Range**: ≥ 0
- **Interpretation**: Small values consistent with ETH, large values suggest ETH violations

### Loschmidt Echo
- **Purpose**: Measures sensitivity to perturbations (quantum chaos/stability)
- **Range**: 0 ≤ L(t) ≤ 1
- **Interpretation**: Rapid decay indicates chaos, slow decay suggests stability

## Testing

Run the test suite to verify functionality:

```bash
python -m pytest diagnostics/test_diagnostics.py -v
```

## Example

See `integration_example.py` for a complete demonstration of how to use the diagnostic tools with the existing quantum engine.

```bash
python diagnostics/integration_example.py
```

## Notes

- All functions are **report-only** and don't modify your core quantum state evolution
- Functions are designed to work with PyTorch tensors and the existing TimeOperator class
- Numerical precision tolerances are built-in for robust operation
- CSV output is compatible with the existing SimpleRecorder system

## Physics Background

- **Quantum Scars**: Anomalous eigenstates that violate ETH and show enhanced localization
- **ETH**: Eigenstate Thermalization Hypothesis - statistical mechanics emerges from quantum mechanics
- **OTOC**: Out-of-Time-Order Correlators - measure quantum information scrambling
- **Loschmidt Echo**: Fidelity under perturbation - probe of quantum chaos and stability