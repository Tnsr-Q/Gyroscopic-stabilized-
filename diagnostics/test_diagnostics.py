"""
Tests for diagnostic functions.

Basic tests to ensure the diagnostic tools work correctly with mock data.
"""
import pytest
import torch
import numpy as np
import sys
import os

# Add the project root to the path to import diagnostics and pkgs
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from diagnostics.scar_metrics import participation_ratio, cheap_otoc_rate, eth_deviation
from diagnostics.echo_meter import loschmidt_echo, echo_report
from pkgs.core_physics.timeops import TimeOperator


class TestScarMetrics:
    """Test suite for quantum scarring metrics."""
    
    def test_participation_ratio(self):
        """Test participation ratio calculation."""
        # Test with normalized random state
        psi = torch.randn(16, dtype=torch.complex64)
        psi = psi / torch.norm(psi)
        
        pr = participation_ratio(psi)
        assert isinstance(pr, float)
        assert 0 < pr <= 16  # PR should be positive and at most the dimension
        
        # Test with maximally localized state (single basis state)
        psi_loc = torch.zeros(16, dtype=torch.complex64)
        psi_loc[0] = 1.0
        pr_loc = participation_ratio(psi_loc)
        assert abs(pr_loc - 1.0) < 1e-10  # Should be exactly 1 for localized state
        
    def test_cheap_otoc_rate(self):
        """Test OTOC rate calculation."""
        time_op = TimeOperator(dim_clock=16)
        psi = torch.randn(16, dtype=torch.complex64)
        psi = psi / torch.norm(psi)
        
        otoc_rate = cheap_otoc_rate(time_op, psi, steps=4, dt=0.01)
        assert isinstance(otoc_rate, float)
        # OTOC rate should be finite (not NaN or inf)
        assert np.isfinite(otoc_rate)
        
    def test_eth_deviation(self):
        """Test ETH deviation calculation."""
        # Create mock energy spectrum and observable values
        energies = np.linspace(0, 10, 100)
        # ETH would predict smooth function of energy
        obs_vals = np.sin(energies) + 0.1 * np.random.randn(100)
        
        eth_dev = eth_deviation(energies, obs_vals, window=0.1)
        assert isinstance(eth_dev, float)
        assert eth_dev >= 0  # Variance should be non-negative
        
        # Test with insufficient data
        small_energies = np.array([1.0, 2.0])
        small_obs = np.array([0.5, 0.6])
        eth_dev_small = eth_deviation(small_energies, small_obs)
        assert eth_dev_small == 0.0


class TestEchoMeter:
    """Test suite for Loschmidt echo measurements."""
    
    def test_loschmidt_echo(self):
        """Test Loschmidt echo calculation."""
        time_op = TimeOperator(dim_clock=16)
        psi = torch.randn(16, dtype=torch.complex64)
        psi = psi / torch.norm(psi)
        
        echo = loschmidt_echo(time_op, psi, t=0.05, eps=1e-3)
        assert isinstance(echo, float)
        assert -1e-6 <= echo <= 1 + 1e-6  # Echo should be between 0 and 1 (with numerical tolerance)
        
        # Test with zero time (should be 1 for unperturbed evolution)
        echo_zero = loschmidt_echo(time_op, psi, t=0.0, eps=1e-3)
        assert abs(echo_zero - 1.0) < 1e-6  # Allow numerical tolerance
        
    def test_echo_report(self):
        """Test echo report generation."""
        time_op = TimeOperator(dim_clock=16)
        psi = torch.randn(16, dtype=torch.complex64)
        psi = psi / torch.norm(psi)
        
        report = echo_report(time_op, psi, t_list=(0.01, 0.05, 0.1))
        
        # Check that all expected keys are present
        expected_keys = {'echo_t0.01', 'echo_t0.05', 'echo_t0.1', 'echo_rate'}
        assert set(report.keys()) == expected_keys
        
        # Check that all values are valid floats
        for key, value in report.items():
            assert isinstance(value, float)
            assert np.isfinite(value)
            
        # Echo values should be between 0 and 1 (with numerical tolerance)
        for key in ['echo_t0.01', 'echo_t0.05', 'echo_t0.1']:
            assert -1e-6 <= report[key] <= 1 + 1e-6
            
        # Echo rate should be non-negative (allowing small numerical errors)
        assert report['echo_rate'] >= -1e-6


class TestIntegration:
    """Integration tests with realistic scenarios."""
    
    def test_diagnostics_with_time_evolution(self):
        """Test diagnostics during actual time evolution."""
        time_op = TimeOperator(dim_clock=16)
        
        # Start with a coherent superposition state
        psi = torch.ones(16, dtype=torch.complex64) / np.sqrt(16)
        
        # Evolve for a short time
        U = time_op.path_conditioned_evolution(0.1)
        psi_evolved = U @ psi
        
        # Check that diagnostics give reasonable values
        pr = participation_ratio(psi_evolved)
        assert 1 - 1e-5 <= pr <= 16 + 1e-5  # Allow small numerical tolerance
        
        echo = loschmidt_echo(time_op, psi_evolved, t=0.05)
        assert -1e-6 <= echo <= 1 + 1e-6
        
        otoc_rate = cheap_otoc_rate(time_op, psi_evolved, steps=4, dt=0.01)
        assert np.isfinite(otoc_rate)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])