# Stress Testing Framework

This directory contains comprehensive stress testing utilities to validate the tensor network implementation under extreme conditions and edge cases.

## Files

- `stress_tests.py` - Main stress testing framework with `StressTestSuite` class
- `__init__.py` - Package initialization

## Overview

The stress testing framework provides comprehensive validation to catch numerical instabilities, memory leaks, and performance bottlenecks before they impact quantum simulations.

## Features

### StressTestSuite Class

The `StressTestSuite` class provides the following test categories:

#### 1. Numerical Stability Testing (`test_numerical_stability`)
- Tests sensitivity to small perturbations (1e-10, 1e-8, 1e-6)
- Measures relative error between base result and perturbed result
- Helps identify numerical instability issues

#### 2. Extreme Value Handling (`test_extreme_values`)
- Tests behavior with edge cases: zeros, ones, large values, small values, infinities, NaNs
- Validates that operations handle extreme inputs gracefully
- Returns boolean success indicators for each test case

#### 3. Memory Scaling Analysis (`test_memory_scaling`)
- Monitors memory usage growth with tensor size
- Tests across different system sizes (default: 16, 32, 64, 128)
- Helps identify memory leaks and scaling issues

#### 4. Performance Scaling (`test_performance_scaling`)
- Measures computational time scaling with system size
- Runs multiple timed iterations for accurate measurements
- Identifies performance bottlenecks

#### 5. Boundary Condition Testing (`test_boundary_conditions`)
- Tests quantum evolution at extreme parameters
- Includes very short timesteps, very large timesteps, and long evolution
- Validates norm preservation and numerical stability

## Usage

### Basic Usage

```python
from diagnostics.stress_tests import StressTestSuite, run_comprehensive_stress_test

# Create stress test suite
suite = StressTestSuite(device="cuda")  # or "cpu"

# Test a specific operation
def my_operation(tensor):
    return tensor @ tensor.T

input_tensor = torch.randn(64, 64)
stability_results = suite.test_numerical_stability(my_operation, input_tensor)
extreme_results = suite.test_extreme_values(my_operation)
```

### Integration with RecursiveConformalComputing

The stress testing framework is integrated into the `RecursiveConformalComputing` class and can be enabled via configuration:

```python
from pkgs.engine_runtime import RecursiveConformalComputing, SimpleRecorder

# Enable stress testing
params = {
    'recorder': SimpleRecorder(),
    'run_stress_tests': True  # Enable automatic stress testing
}

rcc = RecursiveConformalComputing(params=params)

# Stress tests will run automatically during quantum state updates
rcc.update_quantum_state(dt=0.01, r=0.1, torsion_norm=0.05)
```

### Comprehensive Testing

For complete system validation:

```python
from diagnostics.stress_tests import run_comprehensive_stress_test

# Run all stress tests on a tensor network system
results = run_comprehensive_stress_test(rcc)

# Check results
if results['tests_failed'] > 0:
    print(f"⚠️ {results['tests_failed']} stress tests failed!")
else:
    print(f"✅ All {results['total_tests_run']} stress tests passed")
```

## Integration Points

### Automatic Integration
- Set `run_stress_tests: True` in configuration parameters
- Stress tests run automatically during `update_quantum_state()`
- Results are logged to the recorder if available
- Failure alerts are generated when tests fail

### Manual Integration
- Call `run_comprehensive_stress_test(system)` explicitly
- Provides detailed control over when tests run
- Returns comprehensive results dictionary

## Configuration

The stress testing behavior can be controlled through the following parameters:

- `run_stress_tests` (bool): Enable/disable automatic stress testing (default: False)
- `device` (str): Device for tensor operations ("cpu" or "cuda")

## Output Metrics

The stress testing framework generates the following types of metrics:

### Numerical Stability Metrics
- `stability_eps_1e-10`: Relative error for 1e-10 perturbation
- `stability_eps_1e-08`: Relative error for 1e-8 perturbation  
- `stability_eps_1e-06`: Relative error for 1e-6 perturbation

### Extreme Value Metrics
- `extreme_zeros_ok`: Boolean - handles zero tensors
- `extreme_ones_ok`: Boolean - handles all-ones tensors
- `extreme_large_values_ok`: Boolean - handles large values (1e6)
- `extreme_small_values_ok`: Boolean - handles small values (1e-6)
- `extreme_inf_values_ok`: Boolean - handles infinity values
- `extreme_nan_values_ok`: Boolean - handles NaN values

### Memory Scaling Metrics
- `memory_size_16`: Memory usage (MB) for size 16 tensors
- `memory_size_32`: Memory usage (MB) for size 32 tensors
- `memory_size_64`: Memory usage (MB) for size 64 tensors
- `memory_size_128`: Memory usage (MB) for size 128 tensors

### Performance Scaling Metrics
- `time_size_16`: Average execution time for size 16 tensors
- `time_size_32`: Average execution time for size 32 tensors
- `time_size_64`: Average execution time for size 64 tensors

### Boundary Condition Metrics
- `tiny_timestep_overlap`: Overlap for very small timesteps
- `large_timestep_stable`: Stability indicator for large timesteps
- `long_evolution_norm_error`: Norm deviation for long evolution

### Summary Metrics
- `stress_test_completed`: Boolean - whether tests completed successfully
- `total_tests_run`: Number of individual tests executed
- `tests_failed`: Number of tests that failed

## Example Output

```
✅ All 15 stress tests passed

Key stress test metrics:
    stability_eps_1e-10: 0.00e+00
    stability_eps_1e-08: 4.88e-08
    stability_eps_1e-06: 3.24e-06
    extreme_zeros_ok: ✅
    extreme_ones_ok: ✅
    extreme_large_values_ok: ✅
    extreme_small_values_ok: ✅
    extreme_inf_values_ok: ❌
    extreme_nan_values_ok: ❌
    memory_size_8: 0.0000
    memory_size_16: 0.0000
    memory_size_32: 0.0000
    tiny_timestep_overlap: 1.0000
    large_timestep_stable: 1.0000
```

## Dependencies

- `torch` - PyTorch for tensor operations
- `numpy` - Numerical operations
- `warnings` - Warning system
- `time` - Performance timing

## Error Handling

The framework includes comprehensive error handling:

- Individual test failures are caught and logged as warnings
- Failed tests return `float('inf')` or `False` as appropriate
- System-level failures are caught and reported in results
- Graceful degradation when components are unavailable