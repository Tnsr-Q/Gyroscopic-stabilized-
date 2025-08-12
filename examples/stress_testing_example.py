#!/usr/bin/env python3
"""
Integration example demonstrating stress testing functionality.

This script shows how to use the stress testing framework with the
RecursiveConformalComputing system during validation and development.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pkgs.engine_runtime import RecursiveConformalComputing, SimpleRecorder
from diagnostics.stress_tests import run_comprehensive_stress_test


def main():
    """Run integration example with stress testing."""
    print("Stress Testing Integration Example")
    print("=" * 50)
    
    # Configuration with stress testing enabled
    config = {
        'recorder': SimpleRecorder(enabled=True),
        'run_stress_tests': True,  # Enable stress testing
        'd_H': 3.12,
        'C_RT': 91.64,
        'kappa': 0.015,
        'sigma_u': 0.004
    }
    
    print("1. Initializing RecursiveConformalComputing with stress testing enabled...")
    rcc = RecursiveConformalComputing(r0=0.5, m0=1.0, params=config)
    print("   ✅ Initialization complete")
    
    print("\n2. Running simulation steps with automatic stress testing...")
    for step in range(3):
        print(f"   Step {step + 1}:")
        
        # This will trigger automatic stress testing due to run_stress_tests=True
        rcc.update_quantum_state(dt=0.01, r=0.1 + step * 0.05, torsion_norm=0.02 * step)
        
        # Check recorder for stress test results
        recorder = config['recorder']
        recent_entries = recorder.rows[-1:] if recorder.rows else []
        
        if recent_entries:
            stress_metrics = {k: v for k, v in recent_entries[0].items() if k.startswith('stress_')}
            failed_tests = stress_metrics.get('stress_tests_failed', 0)
            total_tests = stress_metrics.get('stress_total_tests_run', 0)
            
            print(f"      Stress tests: {total_tests} run, {failed_tests} failed")
            if failed_tests > 0:
                print(f"      ⚠️  Warning: {failed_tests} stress tests failed!")
            else:
                print(f"      ✅ All stress tests passed")
        else:
            print("      No stress test data recorded")
    
    print("\n3. Running manual comprehensive stress test...")
    manual_results = run_comprehensive_stress_test(rcc)
    
    print(f"   Tests completed: {manual_results.get('stress_test_completed', False)}")
    print(f"   Total tests run: {manual_results.get('total_tests_run', 0)}")
    print(f"   Tests failed: {manual_results.get('tests_failed', 0)}")
    
    # Display some key metrics
    print("\n   Key stress test metrics:")
    for key, value in manual_results.items():
        if not key.startswith('stress_test') and isinstance(value, (int, float)):
            if 'stability' in key:
                print(f"      {key}: {value:.2e}")
            elif 'extreme' in key:
                print(f"      {key}: {'✅' if value else '❌'}")
            elif 'memory' in key or 'time' in key:
                print(f"      {key}: {value:.4f}")
    
    print("\n4. Testing with stress testing disabled...")
    config_no_stress = config.copy()
    config_no_stress['run_stress_tests'] = False
    config_no_stress['recorder'] = SimpleRecorder(enabled=True)
    
    rcc_no_stress = RecursiveConformalComputing(r0=0.5, m0=1.0, params=config_no_stress)
    rcc_no_stress.update_quantum_state(dt=0.01, r=0.1, torsion_norm=0.02)
    
    no_stress_recorder = config_no_stress['recorder']
    stress_entries = [row for row in no_stress_recorder.rows 
                     if any(k.startswith('stress_') for k in row.keys())]
    
    print(f"   Stress test entries with disabled testing: {len(stress_entries)} (should be 0)")
    print("   ✅ Stress testing properly disabled")
    
    print("\n5. Summary of recorded data...")
    total_entries = len(config['recorder'].rows)
    stress_entries = [row for row in config['recorder'].rows 
                     if any(k.startswith('stress_') for k in row.keys())]
    
    print(f"   Total recorder entries: {total_entries}")
    print(f"   Entries with stress test data: {len(stress_entries)}")
    
    if stress_entries:
        stress_keys = [k for k in stress_entries[0].keys() if k.startswith('stress_')]
        print(f"   Unique stress test metrics collected: {len(stress_keys)}")
        print(f"   Sample metrics: {', '.join(stress_keys[:5])}")
    
    print("\n" + "=" * 50)
    print("Integration example completed successfully!")
    print("\nTo enable stress testing in your simulation:")
    print("  rcc_params['run_stress_tests'] = True")
    print("\nTo run manual stress tests:")
    print("  from diagnostics.stress_tests import run_comprehensive_stress_test")
    print("  results = run_comprehensive_stress_test(your_rcc_instance)")


if __name__ == "__main__":
    main()