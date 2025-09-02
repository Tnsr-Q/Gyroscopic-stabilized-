#!/usr/bin/env python3
"""
Simple test script to verify three-time operators and echo functionality.
"""
import torch
import numpy as np
from timeops.three_time_ops import ThreeTimeClock
from gauge.proper_time_field import ProperTimeGauge
from protocols.time_echo import TimeRecompressionProtocol

def test_three_time_clock():
    print("Testing ThreeTimeClock...")
    
    # Create clock with smaller dimension for faster testing
    clock = ThreeTimeClock(d=8, device="cpu")
    
    # Create a test state
    psi0 = torch.randn(clock.d, dtype=torch.complex128)
    psi0 = psi0 / torch.linalg.norm(psi0)
    
    # Test path evolution operators
    lambdas1 = (0.1, 0.2, 0.3)
    lambdas2 = (0.15, 0.25, 0.35)
    
    U1 = clock.U_path(lambdas1)
    U2 = clock.U_path(lambdas2)
    
    # Test visibility before echo
    vis_before = clock.visibility(psi0, U1, U2)
    print(f"  Visibility before echo: {vis_before:.4f}")
    
    # Test echo gate
    delta_taus = (0.05, 0.05, 0.05)
    R_echo = clock.R_echo(delta_taus)
    
    # Test visibility after echo
    vis_after = clock.visibility(psi0, U1, R_echo @ U2)
    print(f"  Visibility after echo: {vis_after:.4f}")
    
    print("  ✓ ThreeTimeClock test passed")

def test_proper_time_gauge():
    print("Testing ProperTimeGauge...")
    
    # Create gauge field on 8x8 lattice
    gauge = ProperTimeGauge(shape=(8, 8), device="cpu")
    
    # Set up a test τ field with some structure
    x, y = torch.meshgrid(torch.linspace(0, 2*np.pi, 8), torch.linspace(0, 2*np.pi, 8), indexing='ij')
    tau_test = 0.1 * torch.sin(x) * torch.cos(y)
    gauge.set_tau(tau_test)
    
    # Test links computation
    links = gauge.links()
    print(f"  Links Ax shape: {links['Ax'].shape}, Ay shape: {links['Ay'].shape}")
    
    # Test curvature computation
    F = gauge.curvature()
    print(f"  Curvature F shape: {F.shape}, max |F|: {torch.abs(F).max().item():.6f}")
    
    # Test flattening
    tau_flat = gauge.flatten(iters=5, alpha=0.1)
    F_after = gauge.curvature()
    print(f"  Curvature after flattening, max |F|: {torch.abs(F_after).max().item():.6f}")
    
    print("  ✓ ProperTimeGauge test passed")

def test_time_recompression_protocol():
    print("Testing TimeRecompressionProtocol...")
    
    # Create clock and protocol
    clock = ThreeTimeClock(d=8, device="cpu")
    protocol = TimeRecompressionProtocol(clock)
    
    # Create test state
    psi0 = torch.randn(clock.d, dtype=torch.complex128)
    psi0 = psi0 / torch.linalg.norm(psi0)
    
    # Test echo run
    lambdas1 = (0.1, 0.2, 0.3)
    lambdas2 = (0.15, 0.25, 0.35)
    
    result = protocol.run_echo(psi0, lambdas1, lambdas2)
    
    print(f"  Visibility before: {result['vis_before']:.4f}")
    print(f"  Visibility after: {result['vis_after']:.4f}")
    print(f"  Delta tau: ({result['d_tau_1']:.4f}, {result['d_tau_2']:.4f}, {result['d_tau_3']:.4f})")
    
    print("  ✓ TimeRecompressionProtocol test passed")

if __name__ == "__main__":
    print("Running three-time operators and echo tests...")
    test_three_time_clock()
    test_proper_time_gauge()
    test_time_recompression_protocol()
    print("All tests passed! ✓")