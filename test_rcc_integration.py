#!/usr/bin/env python3
"""
Integration test to show how the new three-time system works with existing RCC.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pkgs'))
sys.path.insert(0, os.path.dirname(__file__))

from pkgs.engine_runtime.rcc import RecursiveConformalComputing
from timeops.three_time_ops import ThreeTimeClock
from protocols.time_echo import TimeRecompressionProtocol
from integration.time_echo_hooks import time_echo_sense, time_echo_compute, time_echo_actuate

def test_rcc_integration():
    print("Testing RCC integration with three-time system...")
    
    # Create RCC instance
    rcc = RecursiveConformalComputing()
    
    # Create three-time clock
    clock = ThreeTimeClock(d=16, device="cpu")
    
    # Create integration hooks
    sense_fn = time_echo_sense(rcc, clock)
    compute_fn = time_echo_compute(rcc, clock)
    actuate_fn = time_echo_actuate(rcc, clock)
    
    # Test the sense-compute-actuate cycle
    print("  Testing sense phase...")
    observations = sense_fn()
    print(f"    Got {len(observations)} observations")
    print(f"    Clock state shape: {observations['psi0'].shape}")
    
    print("  Testing compute phase...")
    actions = compute_fn(observations)
    print(f"    Delta tau vector: {actions['delta_tau_vec']}")
    print(f"    Visibility improvement: {actions['vis_before']:.4f} -> {actions['vis_after']:.4f}")
    
    print("  Testing actuate phase...")
    actuate_fn(actions)
    print(f"    Stored delta in rcc.params: {rcc.params.get('time_echo_delta', 'None')}")
    
    print("  ✓ RCC integration test passed")

if __name__ == "__main__":
    test_rcc_integration()
    print("Integration test completed! ✓")