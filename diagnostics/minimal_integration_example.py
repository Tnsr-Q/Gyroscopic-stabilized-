#!/usr/bin/env python3
"""
Example demonstrating the exact integration pattern from the problem statement.

This shows the minimal code needed to add optional contextuality monitoring
to an existing main loop, exactly as specified in the requirements.
"""

import sys
sys.path.extend(['.', '..'])

import torch
from pkgs.engine_runtime import RecursiveConformalComputing
from pkgs.engine_runtime.recorder import SimpleRecorder
from diagnostics.contextuality import contextuality_certificate

def main():
    """Demonstrate the exact integration pattern from problem statement."""
    
    # Set up RCC with recorder (your existing setup)
    recorder = SimpleRecorder(enabled=True)
    rcc = RecursiveConformalComputing(r0=0.5, m0=1.0, params={"recorder": recorder})
    
    print("📋 Problem Statement Integration Example")
    print("Using the exact code pattern from the requirements:")
    print()
    
    # Your existing main loop would be here...
    for step in range(5):
        # Your existing evolution/contractor logic would go here
        # For this example, we'll just do a simple state evolution
        dt = 0.01
        
        # Simulate some physics evolution
        noise = 0.005 * torch.randn_like(rcc.psi_clock)
        rcc.psi_clock = rcc.psi_clock + noise
        rcc.psi_clock = rcc.psi_clock / torch.norm(rcc.psi_clock)
        
        # === EXACT INTEGRATION CODE FROM PROBLEM STATEMENT ===
        if rcc.recorder and hasattr(rcc, 'psi_clock'):
            ctx_data = contextuality_certificate(rcc.psi_clock)
            rcc.recorder.log({
                f"ctx_{k}": v for k, v in ctx_data.items()
            })
        # === END INTEGRATION CODE ===
        
        # Log other data (your existing logging)
        rcc.recorder.log({
            "step": step,
            "dt": dt,
            "evolution_time": step * dt
        })
        
        print(f"Step {step}: ctx_strength = {ctx_data['contextuality_strength']:.3f}")
    
    print("\n✅ Integration complete!")
    print(f"Logged {len(recorder.rows)} data points with contextuality monitoring")
    
    # Show what was logged
    if recorder.rows:
        last_row = recorder.rows[-1]
        ctx_keys = [k for k in last_row.keys() if k.startswith('ctx_')]
        print(f"Contextuality fields logged: {ctx_keys}")
    
    print("\n📖 Key Points:")
    print("• Integration requires only 4 lines of code")
    print("• Completely optional - controlled by recorder availability")
    print("• No modification of existing RCC classes needed")
    print("• Graceful fallbacks ensure no crashes on computational failures")
    print("• Works with or without cvxpy dependency")

if __name__ == "__main__":
    main()