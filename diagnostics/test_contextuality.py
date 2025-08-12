#!/usr/bin/env python3
"""
Comprehensive test suite for the contextuality diagnostics module.

This script validates all functionality including:
- Kochen-Specker violation detection
- Lovász theta function (both SDP and eigenvalue fallback)
- Contextuality graph construction
- Combined contextuality certificate
- Graceful error handling and fallbacks
"""

import sys
import os
sys.path.append('.')

import torch
import numpy as np
import warnings
from diagnostics.contextuality import (
    kochen_specker_violation,
    lovasz_theta_bound,
    contextuality_graph_from_state,
    contextuality_certificate,
    _construct_ks_observables
)

def test_ks_observables():
    """Test KS observable construction for different dimensions."""
    print("🔬 Testing KS observables construction...")
    
    # Test small dimension (should return empty)
    obs_2 = _construct_ks_observables(2)
    assert len(obs_2) == 0, "Small dimension should return empty observables"
    print("  ✓ Small dimension handled correctly")
    
    # Test 4D (two qubits)
    obs_4 = _construct_ks_observables(4)
    assert len(obs_4) == 6, "Two-qubit case should have 6 observables"
    for A in obs_4:
        assert A.shape == (4, 4), "Observables should be 4x4"
        # Check Hermiticity
        assert np.allclose(A, A.conj().T), "Observable should be Hermitian"
    print("  ✓ Two-qubit observables constructed correctly")
    
    # Test higher dimension
    obs_16 = _construct_ks_observables(16)
    assert len(obs_16) == 6, "Higher dimension should have 6 observables"
    for A in obs_16:
        assert A.shape == (16, 16), "Observables should be 16x16"
        assert np.allclose(A, A.conj().T), "Observable should be Hermitian"
    print("  ✓ Higher dimension observables constructed correctly")

def test_kochen_specker_violation():
    """Test KS violation detection."""
    print("🔬 Testing Kochen-Specker violation...")
    
    # Test with maximally mixed state (should have low violation)
    dim = 4
    psi_mixed = torch.ones(dim, dtype=torch.complex64) / np.sqrt(dim)
    viol_mixed = kochen_specker_violation(psi_mixed)
    print(f"  Mixed state violation: {viol_mixed:.3f}")
    
    # Test with pure random state
    psi_random = torch.randn(dim, dtype=torch.complex64)
    psi_random = psi_random / torch.norm(psi_random)
    viol_random = kochen_specker_violation(psi_random)
    print(f"  Random state violation: {viol_random:.3f}")
    
    # Test with density matrix input
    rho = torch.outer(psi_random.conj(), psi_random)
    viol_rho = kochen_specker_violation(rho)
    print(f"  Density matrix violation: {viol_rho:.3f}")
    
    # Test with problematic input (should fallback gracefully)
    psi_problem = torch.tensor([float('nan'), 1.0], dtype=torch.complex64)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        viol_problem = kochen_specker_violation(psi_problem)
        assert viol_problem == 0.0, "Problematic input should return 0"
        print("  ✓ Graceful fallback for problematic input")

def test_lovasz_theta():
    """Test Lovász theta function with both SDP and eigenvalue methods."""
    print("🔬 Testing Lovász theta function...")
    
    # Test with simple path graph
    path_3 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    theta_path = lovasz_theta_bound(path_3)
    print(f"  Path graph (3 vertices) theta: {theta_path:.3f}")
    assert 1.5 < theta_path < 2.5, "Path graph theta should be around 2"
    
    # Test with complete graph (should give 1)
    complete_3 = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    theta_complete = lovasz_theta_bound(complete_3)
    print(f"  Complete graph (3 vertices) theta: {theta_complete:.3f}")
    
    # Test with empty graph (no edges)
    empty_3 = np.zeros((3, 3))
    theta_empty = lovasz_theta_bound(empty_3)
    print(f"  Empty graph (3 vertices) theta: {theta_empty:.3f}")
    
    # Test fallback behavior with problematic input
    problem_adj = np.array([[float('inf'), 1], [1, float('nan')]])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        theta_problem = lovasz_theta_bound(problem_adj)
        assert theta_problem == 1.0, "Problematic input should return fallback value"
        print("  ✓ Graceful fallback for problematic adjacency matrix")

def test_contextuality_graph():
    """Test contextuality graph construction."""
    print("🔬 Testing contextuality graph construction...")
    
    # Test with normal quantum state
    psi = torch.randn(16, dtype=torch.complex64)
    psi = psi / torch.norm(psi)
    
    graph = contextuality_graph_from_state(psi, threshold=0.1)
    print(f"  Graph shape: {graph.shape}")
    assert graph.shape[0] == graph.shape[1], "Graph should be square"
    assert np.allclose(graph, graph.T), "Graph should be symmetric"
    assert np.all(np.diag(graph) == 0), "Graph should have no self-loops"
    
    # Test with different threshold
    graph_strict = contextuality_graph_from_state(psi, threshold=0.5)
    edges_normal = np.sum(graph) / 2
    edges_strict = np.sum(graph_strict) / 2
    print(f"  Edges (threshold=0.1): {edges_normal}")
    print(f"  Edges (threshold=0.5): {edges_strict}")
    assert edges_strict <= edges_normal, "Stricter threshold should give fewer edges"
    
    # Test with small dimension (should return trivial graph)
    psi_small = torch.randn(2, dtype=torch.complex64)
    psi_small = psi_small / torch.norm(psi_small)
    graph_small = contextuality_graph_from_state(psi_small)
    assert graph_small.shape == (2, 2), "Small dimension should return 2x2 graph"
    print("  ✓ Small dimension handled correctly")

def test_contextuality_certificate():
    """Test the combined contextuality certificate."""
    print("🔬 Testing contextuality certificate...")
    
    # Test with normal state
    psi = torch.randn(16, dtype=torch.complex64)
    psi = psi / torch.norm(psi)
    
    cert = contextuality_certificate(psi)
    required_keys = {"ks_violation", "lovasz_theta", "contextuality_strength", "graph_edges"}
    assert set(cert.keys()) == required_keys, f"Certificate should have keys: {required_keys}"
    
    for key, value in cert.items():
        assert isinstance(value, (int, float)), f"Value for {key} should be numeric"
        assert not np.isnan(value), f"Value for {key} should not be NaN"
        assert not np.isinf(value), f"Value for {key} should not be infinite"
    
    print(f"  Certificate: {cert}")
    
    # Test contextuality strength relationship
    expected_strength = cert["ks_violation"] * min(cert["lovasz_theta"], 10.0)
    assert abs(cert["contextuality_strength"] - expected_strength) < 1e-6, "Strength calculation mismatch"
    print("  ✓ Contextuality strength calculated correctly")
    
    # Test fallback behavior
    psi_problem = torch.tensor([float('inf')], dtype=torch.complex64)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cert_problem = contextuality_certificate(psi_problem)
        expected_fallback = {
            "ks_violation": 0.0,
            "lovasz_theta": 1.0,
            "contextuality_strength": 0.0,
            "graph_edges": 0.0
        }
        assert cert_problem == expected_fallback, "Problematic input should return fallback values"
        print("  ✓ Graceful fallback for problematic input")

def test_edge_cases():
    """Test various edge cases and boundary conditions."""
    print("🔬 Testing edge cases...")
    
    # Test with very small states
    psi_1d = torch.tensor([1.0], dtype=torch.complex64)
    cert_1d = contextuality_certificate(psi_1d)
    print(f"  1D state certificate: {cert_1d}")
    
    # Test with zero state
    psi_zero = torch.zeros(4, dtype=torch.complex64)
    cert_zero = contextuality_certificate(psi_zero)
    print(f"  Zero state certificate: {cert_zero}")
    
    # Test with complex phases
    psi_phases = torch.tensor([1.0, 1j, -1.0, -1j], dtype=torch.complex64)
    psi_phases = psi_phases / torch.norm(psi_phases)
    cert_phases = contextuality_certificate(psi_phases)
    print(f"  Complex phases certificate: {cert_phases}")
    
    print("  ✓ All edge cases handled gracefully")

def test_integration_compatibility():
    """Test compatibility with the main RCC system."""
    print("🔬 Testing integration compatibility...")
    
    try:
        from pkgs.engine_runtime import RecursiveConformalComputing
        
        # Initialize minimal RCC system
        rcc = RecursiveConformalComputing(r0=0.5, m0=1.0)
        
        # Test with the RCC's quantum state
        cert = contextuality_certificate(rcc.psi_clock)
        print(f"  RCC psi_clock certificate: {cert}")
        
        # Verify the state has expected properties
        assert torch.is_complex(rcc.psi_clock), "psi_clock should be complex"
        assert abs(torch.norm(rcc.psi_clock) - 1.0) < 1e-6, "psi_clock should be normalized"
        
        print("  ✓ Integration with RCC system successful")
        
    except ImportError as e:
        print(f"  ⚠️ Could not test RCC integration: {e}")

def main():
    """Run comprehensive test suite."""
    print("🧪 Contextuality Diagnostics Test Suite")
    print("=" * 50)
    
    test_functions = [
        test_ks_observables,
        test_kochen_specker_violation,
        test_lovasz_theta,
        test_contextuality_graph,
        test_contextuality_certificate,
        test_edge_cases,
        test_integration_compatibility
    ]
    
    passed = 0
    total = len(test_functions)
    
    for test_func in test_functions:
        try:
            test_func()
            passed += 1
            print()
        except Exception as e:
            print(f"  ❌ Test failed: {e}")
            print()
    
    print("📊 Test Results")
    print("=" * 50)
    print(f"Passed: {passed}/{total}")
    print(f"Success rate: {100 * passed / total:.1f}%")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️ Some tests failed!")
        return 1

if __name__ == "__main__":
    exit(main())