#!/usr/bin/env python3
"""
Demonstration of the new physics primitives and recursive loop services.

This script shows how to integrate the new services with the existing RCC system
following the pattern described in the problem statement.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pkgs.engine_runtime import RecursiveConformalComputing
from architectures import BronteinCube
from integration import (
    make_mt_wormhole, sense_fn_factory, compute_fn_factory, actuate_fn_factory
)

def demo_integration():
    """Demonstrate integration of new physics primitives with RCC."""
    print("🌌 Physics Primitives & Recursive Loop Demo")
    print("=" * 50)
    
    # Step 1: Create RCC instance
    print("1. Creating RecursiveConformalComputing instance...")
    rcc = RecursiveConformalComputing(r0=0.5, m0=1.0)
    print(f"   ✓ RCC created with r0={rcc.r0}, m0={rcc.m0}")
    
    # Step 2: Create MT wormhole and factory functions
    print("\n2. Setting up physics primitives...")
    mt = make_mt_wormhole(rcc, r0=0.5)
    sense_fn = sense_fn_factory(rcc)
    compute_fn = compute_fn_factory(rcc, mt, spin_density=0.02)
    actuate_fn = actuate_fn_factory(rcc)
    print("   ✓ Morris-Thorne wormhole configured")
    print("   ✓ Sense/Compute/Actuate functions created")
    
    # Step 3: Create Brontein Cube
    print("\n3. Creating Brontein Cube architecture...")
    brontein_cube = BronteinCube(
        sense=sense_fn, 
        compute=compute_fn, 
        actuate=actuate_fn, 
        hz=50.0
    )
    print("   ✓ Brontein Cube ready for sense-compute-actuate loops")
    
    # Step 4: Demonstrate one cycle
    print("\n4. Running one sense-compute-actuate cycle...")
    
    # Manual cycle for demonstration
    print("   Sensing...")
    obs = sense_fn()
    print(f"     Observed: QEI headroom={obs.get('qei_headroom', 0):.3f}")
    print(f"               Conditioning={obs.get('total_conditioning', 1):.3f}")
    print(f"               Proper time grad={obs.get('proper_time_grad', 0):.3f}")
    
    print("   Computing...")
    action = compute_fn(obs)
    print(f"     Computed: Torsion force={action.get('torsion_force', 0):.6f}")
    print(f"               Gamma next={action.get('gamma_next', 0):.3f}")
    print(f"               m_eff={action.get('m_eff', 1):.3f}")
    
    initial_gamma = getattr(rcc, 'gamma_state', 0.3)
    print("   Actuating...")
    actuate_fn(action)
    final_gamma = getattr(rcc, 'gamma_state', 0.3)
    print(f"     Gamma state: {initial_gamma:.6f} → {final_gamma:.6f}")
    
    # Step 5: Test wormhole metrics
    print("\n5. Testing Morris-Thorne wormhole metrics...")
    print(f"   Flare-out condition satisfied: {mt.flare_out_ok()}")
    g_tt, g_rr = mt.metric_factors(r=1.0)
    print(f"   Metric factors at r=1.0: g_tt={g_tt:.3f}, g_rr={g_rr:.3f}")
    V_eff = mt.V_eff_sq(r=1.0, Lz=0.5, m=1.0)
    print(f"   Effective potential²: {V_eff:.6f}")
    
    print("\n✅ Demo completed successfully!")
    print("   The new physics primitives are ready for use with RCC.")
    print("   Use brontein_cube.run(steps=N) for continuous operation.")

if __name__ == "__main__":
    demo_integration()