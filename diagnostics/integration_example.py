#!/usr/bin/env python3
"""
Integration example showing optional contextuality monitoring in the main loop.

This demonstrates how to use the contextuality certificate with the existing
RecursiveConformalComputing system without modifying core classes.
"""

import sys
import os
sys.path.extend(['.', '..'])

import torch
import numpy as np
from pkgs.engine_runtime import RecursiveConformalComputing
from pkgs.engine_runtime.recorder import SimpleRecorder
from diagnostics.contextuality import contextuality_certificate

def main():
    """Demonstrate contextuality monitoring integration."""
    print("🧬 Contextuality Integration Example")
    print("=" * 50)
    
    # Set up the recorder
    recorder = SimpleRecorder(enabled=True)
    
    # Initialize RCC with recorder
    params = {
        "recorder": recorder,
        "d_H": 3.12,
        "C_RT": 91.64,
        "kappa": 0.015,
        "sigma_u": 0.004
    }
    
    rcc = RecursiveConformalComputing(r0=0.5, m0=1.0, params=params)
    
    print(f"✓ RCC initialized with psi_clock dimension: {rcc.psi_clock.shape}")
    print(f"✓ Recorder enabled: {recorder.enabled}")
    
    # Simulate evolution steps with contextuality monitoring
    print("\n📊 Starting evolution with contextuality monitoring...")
    
    for step in range(10):
        # Simulate some evolution (this would be your actual evolution step)
        dt = 0.01
        
        # Your existing evolution logic would go here...
        # For demonstration, we'll just evolve the quantum state slightly
        noise = 0.01 * torch.randn_like(rcc.psi_clock)
        rcc.psi_clock = rcc.psi_clock + noise
        rcc.psi_clock = rcc.psi_clock / torch.norm(rcc.psi_clock)
        
        # Optional contextuality monitoring (this is the integration point!)
        if rcc.recorder and hasattr(rcc, 'psi_clock'):
            ctx_data = contextuality_certificate(rcc.psi_clock)
            rcc.recorder.log({
                f"ctx_{k}": v for k, v in ctx_data.items()
            })
            
        # Log other physics data
        rcc.recorder.log({
            "step": step,
            "dt": dt,
            "psi_norm": float(torch.norm(rcc.psi_clock)),
            "time": step * dt
        })
        
        if step % 3 == 0:
            print(f"  Step {step:2d}: "
                  f"KS_viol={ctx_data['ks_violation']:.3f}, "
                  f"θ={ctx_data['lovasz_theta']:.3f}, "
                  f"strength={ctx_data['contextuality_strength']:.3f}")
    
    print("\n📈 Evolution completed. Analyzing results...")
    
    # Show summary statistics
    if recorder.rows:
        print(f"✓ Recorded {len(recorder.rows)} data points")
        
        # Extract contextuality data
        ks_violations = [row.get('ctx_ks_violation', 0) for row in recorder.rows if 'ctx_ks_violation' in row]
        theta_values = [row.get('ctx_lovasz_theta', 0) for row in recorder.rows if 'ctx_lovasz_theta' in row]
        
        if ks_violations:
            print(f"✓ KS violations: min={min(ks_violations):.3f}, max={max(ks_violations):.3f}, avg={np.mean(ks_violations):.3f}")
        if theta_values:
            print(f"✓ Lovász θ: min={min(theta_values):.3f}, max={max(theta_values):.3f}, avg={np.mean(theta_values):.3f}")
        
        # Save to multiple formats for analysis
        output_base = "contextuality_example_output"
        recorder.dump_jsonl(f"{output_base}.jsonl")
        print(f"✓ Data saved to: {output_base}.jsonl")
        
        # Also try CSV - will work if all rows have same keys
        try:
            recorder.dump_csv(f"{output_base}.csv")
            print(f"✓ CSV data saved to: {output_base}.csv")
        except Exception as e:
            print(f"⚠️  CSV export skipped: {e}")
        
    print("\n🎯 Integration example completed successfully!")
    print("\nKey features demonstrated:")
    print("  • Non-invasive contextuality monitoring")
    print("  • Graceful fallbacks for computational failures")  
    print("  • Integration with existing recorder system")
    print("  • Optional contextuality certificates in main loop")

if __name__ == "__main__":
    main()