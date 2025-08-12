import numpy as np
import torch
from typing import Dict, List, Callable, Any, Optional
import warnings
import time

class StressTestSuite:
    """Comprehensive stress testing for tensor network operations."""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.results = {}
        
    def test_numerical_stability(self, tensor_op: Callable, 
                                input_tensor: torch.Tensor,
                                perturbations: List[float] = [1e-10, 1e-8, 1e-6]) -> Dict[str, float]:
        """Test numerical stability under small perturbations."""
        results = {}
        try:
            base_result = tensor_op(input_tensor)
            
            for eps in perturbations:
                # Add small random perturbation
                noise = eps * torch.randn_like(input_tensor)
                perturbed_input = input_tensor + noise
                perturbed_result = tensor_op(perturbed_input)
                
                # Measure relative error
                if torch.is_tensor(base_result) and torch.is_tensor(perturbed_result):
                    rel_error = torch.norm(perturbed_result - base_result) / (torch.norm(base_result) + 1e-12)
                    results[f"stability_eps_{eps}"] = float(rel_error)
                else:
                    results[f"stability_eps_{eps}"] = 0.0
                    
        except Exception as e:
            warnings.warn(f"Numerical stability test failed: {e}")
            for eps in perturbations:
                results[f"stability_eps_{eps}"] = float('inf')
                
        return results
    
    def test_extreme_values(self, tensor_op: Callable,
                           shape: tuple = (64, 64)) -> Dict[str, bool]:
        """Test behavior with extreme input values."""
        results = {}
        test_cases = {
            "zeros": torch.zeros(shape, device=self.device),
            "ones": torch.ones(shape, device=self.device),
            "large_values": 1e6 * torch.ones(shape, device=self.device),
            "small_values": 1e-6 * torch.ones(shape, device=self.device),
            "inf_values": torch.full(shape, float('inf'), device=self.device),
            "nan_values": torch.full(shape, float('nan'), device=self.device),
        }
        
        for name, test_input in test_cases.items():
            try:
                result = tensor_op(test_input)
                # Check if result is well-behaved
                if torch.is_tensor(result):
                    is_finite = torch.isfinite(result).all().item()
                    results[f"extreme_{name}_ok"] = bool(is_finite)
                else:
                    results[f"extreme_{name}_ok"] = True
            except Exception:
                results[f"extreme_{name}_ok"] = False
                
        return results
    
    def test_memory_scaling(self, tensor_factory: Callable[[int], torch.Tensor],
                           sizes: List[int] = [16, 32, 64, 128]) -> Dict[str, float]:
        """Test memory usage scaling with tensor size."""
        results = {}
        
        for size in sizes:
            try:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                # Measure memory before
                mem_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                
                # Create tensor
                tensor = tensor_factory(size)
                
                # Measure memory after
                mem_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                mem_used = (mem_after - mem_before) / (1024**2)  # MB
                
                results[f"memory_size_{size}"] = float(mem_used)
                
                # Clean up
                del tensor
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                results[f"memory_size_{size}"] = float('inf')
                warnings.warn(f"Memory test failed for size {size}: {e}")
                
        return results
    
    def test_performance_scaling(self, operation: Callable[[torch.Tensor], Any],
                                tensor_factory: Callable[[int], torch.Tensor],
                                sizes: List[int] = [16, 32, 64]) -> Dict[str, float]:
        """Test computational time scaling."""
        results = {}
        
        for size in sizes:
            try:
                tensor = tensor_factory(size)
                
                # Warm-up run
                _ = operation(tensor)
                
                # Timed runs
                times = []
                for _ in range(3):
                    start_time = time.time()
                    _ = operation(tensor)
                    end_time = time.time()
                    times.append(end_time - start_time)
                
                avg_time = np.mean(times)
                results[f"time_size_{size}"] = float(avg_time)
                
            except Exception as e:
                results[f"time_size_{size}"] = float('inf')
                warnings.warn(f"Performance test failed for size {size}: {e}")
                
        return results
    
    def test_boundary_conditions(self, quantum_evolution: Callable,
                                initial_state: torch.Tensor) -> Dict[str, float]:
        """Test quantum evolution at boundary conditions."""
        results = {}
        
        # Test very short time steps
        try:
            tiny_dt = 1e-12
            result_tiny = quantum_evolution(initial_state, dt=tiny_dt, steps=1)
            overlap = abs(torch.vdot(initial_state, result_tiny))**2
            results["tiny_timestep_overlap"] = float(overlap)
        except Exception:
            results["tiny_timestep_overlap"] = 0.0
        
        # Test very large time steps
        try:
            large_dt = 1e2
            result_large = quantum_evolution(initial_state, dt=large_dt, steps=1)
            if torch.isfinite(result_large).all():
                results["large_timestep_stable"] = 1.0
            else:
                results["large_timestep_stable"] = 0.0
        except Exception:
            results["large_timestep_stable"] = 0.0
        
        # Test long evolution
        try:
            result_long = quantum_evolution(initial_state, dt=0.01, steps=1000)
            norm_deviation = abs(torch.norm(result_long)**2 - 1.0)
            results["long_evolution_norm_error"] = float(norm_deviation)
        except Exception:
            results["long_evolution_norm_error"] = float('inf')
            
        return results

def run_comprehensive_stress_test(tensor_network_system: Any) -> Dict[str, Any]:
    """
    Run comprehensive stress tests on a tensor network system.
    
    Args:
        tensor_network_system: Object with methods like evolve(), get_state(), etc.
    """
    suite = StressTestSuite()
    all_results = {}
    
    try:
        # Get current state for testing
        if hasattr(tensor_network_system, 'get_state'):
            current_state = tensor_network_system.get_state()
        elif hasattr(tensor_network_system, 'psi_clock'):
            current_state = tensor_network_system.psi_clock
        else:
            # Create a dummy state for testing
            current_state = torch.randn(64, dtype=torch.complex64)
            current_state = current_state / torch.norm(current_state)
        
        # Test 1: Numerical stability of state evolution
        if hasattr(tensor_network_system, 'evolve'):
            evolution_op = lambda x: tensor_network_system.evolve(x, steps=1)
            stability_results = suite.test_numerical_stability(evolution_op, current_state)
            all_results.update(stability_results)
        
        # Test 2: Extreme value handling
        if hasattr(tensor_network_system, 'apply_operator'):
            operator_results = suite.test_extreme_values(tensor_network_system.apply_operator)
            all_results.update(operator_results)
        
        # Test 3: Memory scaling
        def state_factory(size):
            return torch.randn(size, size, dtype=torch.complex64) / size
            
        memory_results = suite.test_memory_scaling(state_factory, sizes=[8, 16, 32])
        all_results.update(memory_results)
        
        # Test 4: Performance scaling
        if hasattr(tensor_network_system, 'time_step'):
            perf_results = suite.test_performance_scaling(
                tensor_network_system.time_step,
                state_factory,
                sizes=[8, 16, 32]
            )
            all_results.update(perf_results)
        
        # Test 5: Boundary conditions
        if hasattr(tensor_network_system, 'evolve'):
            def evolution_wrapper(state, dt=0.01, steps=10):
                return tensor_network_system.evolve(state, steps=steps)
            
            boundary_results = suite.test_boundary_conditions(evolution_wrapper, current_state)
            all_results.update(boundary_results)
        
        # Summary statistics
        all_results["stress_test_completed"] = True
        all_results["total_tests_run"] = len([k for k in all_results.keys() if not k.startswith("stress_test")])
        
        # Count failures
        failed_tests = sum(1 for v in all_results.values() 
                          if isinstance(v, float) and (v == float('inf') or np.isnan(v)))
        all_results["tests_failed"] = failed_tests
        
    except Exception as e:
        warnings.warn(f"Comprehensive stress test failed: {e}")
        all_results["stress_test_completed"] = False
        all_results["error_message"] = str(e)
    
    return all_results