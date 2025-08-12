"""Integration test for engine service."""

import pytest
import tempfile
import os
from quantum_core.apps.engine.engine_service import EngineService
from quantum_core.pkgs.engine_runtime.schemas import InitRequest, StepRequest

class TestEngineIntegration:
    """Integration tests for the complete engine service."""
    
    def test_engine_initialization(self):
        """Test engine can be initialized."""
        config = {
            'log_level': 'WARNING',  # Reduce log noise in tests
            'default_params': {
                'sigma_u': 0.01,  # Larger for stability
                'chi_max': 4,     # Smaller for speed
            }
        }
        
        engine = EngineService(config)
        
        init_req = InitRequest(
            seed=42,
            grid_size=8,  # Small for speed
            device="cpu",
            params={}
        )
        
        result = engine.init(init_req)
        
        assert result["status"] == "initialized"
        assert result["seed"] == 42
        assert result["grid_size"] == 8
        assert "baseline_ane" in result
        assert engine.rcc is not None
        assert engine.contractor is not None
    
    def test_engine_step_execution(self):
        """Test engine can execute physics steps."""
        config = {
            'log_level': 'WARNING',
            'default_params': {
                'sigma_u': 0.01,
                'chi_max': 4,
            }
        }
        
        engine = EngineService(config)
        
        # Initialize
        init_req = InitRequest(seed=42, grid_size=8, device="cpu")
        engine.init(init_req)
        
        # Execute step
        step_req = StepRequest(dt=0.01, r=0.5, do_contract=True)
        result = engine.step(step_req)
        
        assert isinstance(result.K, int)
        assert isinstance(result.ane_smear, float)
        assert isinstance(result.guard_ok, bool)
        assert "proper_time" in result.metrics
        assert "step_duration" in result.metrics
    
    def test_multiple_steps(self):
        """Test engine can execute multiple consecutive steps."""
        config = {
            'log_level': 'ERROR',  # Minimal logging
            'default_params': {
                'sigma_u': 0.02,  # Very stable
                'chi_max': 4,
            }
        }
        
        engine = EngineService(config)
        
        # Initialize
        init_req = InitRequest(seed=123, grid_size=8, device="cpu")
        engine.init(init_req)
        
        # Execute multiple steps
        results = []
        for i in range(5):
            step_req = StepRequest(dt=0.01, r=0.5 + 0.01*i, do_contract=True)
            result = engine.step(step_req)
            results.append(result)
        
        assert len(results) == 5
        # Check that ANE values are reasonable
        ane_values = [r.ane_smear for r in results]
        assert all(isinstance(ane, float) for ane in ane_values)
        assert all(abs(ane) < 1e3 for ane in ane_values)  # Sanity check
    
    def test_engine_snapshot(self):
        """Test engine snapshot functionality."""
        config = {'log_level': 'ERROR'}
        engine = EngineService(config)
        
        # Initialize
        init_req = InitRequest(seed=42, grid_size=8, device="cpu")
        engine.init(init_req)
        
        # Get snapshot
        snapshot = engine.snapshot()
        
        assert snapshot["status"] == "active"
        assert "state" in snapshot
        assert "metrics_summary" in snapshot
        assert "tensor_layers" in snapshot
        assert snapshot["graph_vertices"] == 16  # Tesseract has 16 vertices
    
    def test_log_export(self):
        """Test log export functionality."""
        config = {'log_level': 'ERROR'}
        engine = EngineService(config)
        
        # Initialize
        init_req = InitRequest(seed=42, grid_size=8, device="cpu")
        engine.init(init_req)
        
        # Execute a few steps to generate logs
        for i in range(3):
            step_req = StepRequest(dt=0.01, r=0.5, do_contract=True)
            engine.step(step_req)
        
        # Export logs
        with tempfile.TemporaryDirectory() as temp_dir:
            path_prefix = os.path.join(temp_dir, "test_logs")
            
            # Test JSONL export
            jsonl_path = engine.export_logs("jsonl", path_prefix)
            assert os.path.exists(jsonl_path)
            assert jsonl_path.endswith(".jsonl")
            
            # Test CSV export
            csv_path = engine.export_logs("csv", path_prefix)
            assert os.path.exists(csv_path)
            assert csv_path.endswith(".csv")
    
    def test_engine_reset(self):
        """Test engine reset functionality."""
        config = {'log_level': 'ERROR'}
        engine = EngineService(config)
        
        # Initialize and run steps
        init_req = InitRequest(seed=42, grid_size=8, device="cpu")
        engine.init(init_req)
        
        step_req = StepRequest(dt=0.01, r=0.5, do_contract=True)
        engine.step(step_req)
        
        # Check we have some logs
        assert len(engine.recorder.rows) > 0
        
        # Reset
        engine.reset()
        
        # Check logs are cleared
        assert len(engine.recorder.rows) == 0
    
    def test_qei_guard_behavior(self):
        """Test QEI guard violations are handled properly."""
        config = {
            'log_level': 'ERROR',
            'default_params': {
                'sigma_u': 0.001,  # Very small - more likely to trigger violations
                'chi_max': 4,
            }
        }
        
        engine = EngineService(config)
        
        # Initialize
        init_req = InitRequest(seed=42, grid_size=8, device="cpu")
        engine.init(init_req)
        
        # Execute steps and check guard status
        guard_statuses = []
        for i in range(10):
            step_req = StepRequest(dt=0.05, r=1.0, do_contract=True)  # Larger dt, r
            result = engine.step(step_req)
            guard_statuses.append(result.guard_ok)
        
        # Should have some results (guard failures are handled gracefully)
        assert len(guard_statuses) == 10
        assert all(isinstance(status, bool) for status in guard_statuses)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])