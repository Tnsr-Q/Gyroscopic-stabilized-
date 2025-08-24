#!/usr/bin/env python3
"""
Integration example showing how to use the diagnostic tools.

This example demonstrates how to integrate the quantum scarring and echo
diagnostic tools with the existing RecursiveConformalComputing system.
"""
import sys
import os
import torch
import numpy as np

# Add the project root to the path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from diagnostics.scar_metrics import participation_ratio, cheap_otoc_rate, eth_deviation
from diagnostics.echo_meter import loschmidt_echo, echo_report
from pkgs.engine_runtime.rcc import RecursiveConformalComputing
from pkgs.engine_runtime.recorder import SimpleRecorder


def run_diagnostics_example():
    """Run a simple example showing diagnostic tool integration."""
    print("=== Quantum Diagnostics Integration Example ===\n")
    
    # Initialize the system with a recorder
    recorder = SimpleRecorder(enabled=True)
    params = {
        'recorder': recorder,
        'd_H': 3.12,
        'C_RT': 91.64,
        'kappa': 0.015,
        'sigma_u': 0.004
    }
    
    # Create RCC instance
    rcc = RecursiveConformalComputing(r0=0.5, m0=1.0, params=params)
    
    print(f"Initialized system with {rcc.time_op.dim_clock}-dimensional clock space")
    print(f"Initial psi_clock norm: {torch.norm(rcc.psi_clock):.6f}")
    print()
    
    # Run a few simulation steps with diagnostics
    dt = 0.01
    num_steps = 10
    
    print("Running simulation with diagnostic measurements...")
    print("Step | PR      | OTOC Rate | Echo T=0.05 | Echo Rate | ETH Dev")
    print("-" * 65)
    
    for step in range(num_steps):
        # Evolve the system slightly
        U = rcc.time_op.path_conditioned_evolution(dt)
        rcc.psi_clock = U @ rcc.psi_clock
        rcc.psi_clock = rcc.psi_clock / torch.norm(rcc.psi_clock)  # Renormalize
        
        # Calculate diagnostic metrics
        pr = participation_ratio(rcc.psi_clock)
        otoc_rate = cheap_otoc_rate(rcc.time_op, rcc.psi_clock, steps=6, dt=0.01)
        echo_single = loschmidt_echo(rcc.time_op, rcc.psi_clock, t=0.05, eps=1e-2)
        echo_data = echo_report(rcc.time_op, rcc.psi_clock, t_list=(0.02, 0.05, 0.1))
        
        # For ETH deviation, create mock eigenstate data
        # In a real application, you would use actual eigenstate analysis
        energies = np.linspace(0, 1, rcc.time_op.dim_clock)
        obs_vals = np.abs(rcc.psi_clock.detach().cpu().numpy())**2
        eth_dev = eth_deviation(energies, obs_vals, window=0.1)
        
        # Log to recorder (as shown in the problem statement)
        if recorder:
            recorder.log({
                "step": step,
                "scar_pr": pr,
                "otoc_rate": otoc_rate,
                **echo_data,
                "eth_deviation": eth_dev
            })
        
        # Print progress
        print(f"{step:4d} | {pr:7.4f} | {otoc_rate:9.4e} | {echo_single:11.4f} | "
              f"{echo_data['echo_rate']:9.4f} | {eth_dev:7.4e}")
    
    print("\n=== Final Analysis ===")
    
    # Analyze the recorded data
    if recorder.rows:
        data = recorder.rows
        avg_pr = np.mean([row['scar_pr'] for row in data])
        avg_otoc = np.mean([row['otoc_rate'] for row in data])
        avg_echo_rate = np.mean([row['echo_rate'] for row in data])
        
        print(f"Average Participation Ratio: {avg_pr:.4f}")
        print(f"Average OTOC Rate: {avg_otoc:.4e}")
        print(f"Average Echo Decay Rate: {avg_echo_rate:.4f}")
        
        # Interpretation
        print("\n=== Diagnostic Interpretation ===")
        if avg_pr > 8:  # For 16-dim system, PR > N/2 suggests delocalization
            print("• State appears delocalized (normal thermalization)")
        else:
            print("• State shows some localization (potential scarring)")
            
        if abs(avg_otoc) > 1e-2:
            print("• Significant OTOC growth indicates quantum chaotic behavior")
        else:
            print("• Weak OTOC growth suggests integrable or scar-like dynamics")
            
        if avg_echo_rate > 0.1:
            print("• Fast echo decay indicates sensitivity to perturbations")
        else:
            print("• Slow echo decay suggests stable evolution")
    
    # Save diagnostics data if needed
    output_file = "diagnostics_example_output.csv"
    if recorder.rows:
        recorder.dump_csv(output_file)
        print(f"\nDiagnostic data saved to: {output_file}")
    
    return recorder


def demonstrate_individual_functions():
    """Demonstrate each diagnostic function individually."""
    print("\n=== Individual Function Demonstrations ===\n")
    
    # Create a simple time operator and test states
    from pkgs.core_physics.timeops import TimeOperator
    time_op = TimeOperator(dim_clock=16)
    
    # Test state 1: Maximally mixed superposition
    psi_mixed = torch.ones(16, dtype=torch.complex64) / np.sqrt(16)
    print("Test State 1: Maximally mixed superposition")
    print(f"  Participation Ratio: {participation_ratio(psi_mixed):.4f}")
    print(f"  Loschmidt Echo (t=0.05): {loschmidt_echo(time_op, psi_mixed, t=0.05):.4f}")
    
    # Test state 2: Localized state (quantum scar-like)
    psi_localized = torch.zeros(16, dtype=torch.complex64)
    psi_localized[0] = 1.0
    print("\nTest State 2: Localized state (scar-like)")
    print(f"  Participation Ratio: {participation_ratio(psi_localized):.4f}")
    print(f"  Loschmidt Echo (t=0.05): {loschmidt_echo(time_op, psi_localized, t=0.05):.4f}")
    
    # Test state 3: Random state
    psi_random = torch.randn(16, dtype=torch.complex64)
    psi_random = psi_random / torch.norm(psi_random)
    print("\nTest State 3: Random state")
    print(f"  Participation Ratio: {participation_ratio(psi_random):.4f}")
    print(f"  Loschmidt Echo (t=0.05): {loschmidt_echo(time_op, psi_random, t=0.05):.4f}")
    
    # OTOC rate comparison
    print("\nOTOC Rate Comparison:")
    print(f"  Mixed state: {cheap_otoc_rate(time_op, psi_mixed, steps=6, dt=0.01):.4e}")
    print(f"  Localized state: {cheap_otoc_rate(time_op, psi_localized, steps=6, dt=0.01):.4e}")
    print(f"  Random state: {cheap_otoc_rate(time_op, psi_random, steps=6, dt=0.01):.4e}")
    
    # Echo report demonstration
    print("\nEcho Report for Mixed State:")
    echo_data = echo_report(time_op, psi_mixed, t_list=(0.01, 0.05, 0.1))
    for key, value in echo_data.items():
        print(f"  {key}: {value:.6f}")


if __name__ == "__main__":
    print("Quantum Scarring and Echo Diagnostics Integration Example")
    print("=" * 60)
    
    # Run the main integration example
    recorder = run_diagnostics_example()
    
    # Demonstrate individual functions
    demonstrate_individual_functions()
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("\nIntegration Summary:")
    print("• All diagnostic functions work with existing TimeOperator")
    print("• Easy integration with SimpleRecorder for data logging")
    print("• Report-only functions don't modify quantum state evolution")
    print("• Useful for detecting quantum scars, chaos, and ETH violations")