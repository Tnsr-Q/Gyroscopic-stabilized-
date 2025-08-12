"""Basic smoke tests for core physics components."""

import pytest
import numpy as np
import torch
from quantum_core.pkgs.core_physics import (
    BondGraph, Edge, build_tesseract_lattice,
    MERASpacetime, TimeOperator, ProperTimeGaugeField
)

class TestLattice:
    """Test lattice structures."""
    
    def test_edge_creation(self):
        edge = Edge(0, 1)
        assert edge.u == 0
        assert edge.v == 1
    
    def test_bond_graph_creation(self):
        graph = BondGraph(V=4, edges=[Edge(0, 1), Edge(1, 2)], baseline_chi=2)
        assert graph.V == 4
        assert len(graph.edges) == 2
        assert graph.baseline_chi == 2
    
    def test_tesseract_lattice(self):
        graph = build_tesseract_lattice()
        assert graph.V == 16  # 4D hypercube has 16 vertices
        assert len(graph.edges) == 32  # 4*16/2 edges
        assert graph.baseline_chi == 2

class TestTensors:
    """Test tensor operations."""
    
    def test_mera_initialization(self):
        mera = MERASpacetime(layers=4, bond_dim=2)
        assert mera.layers == 4
        assert mera.bond_dim == 2
        assert len(mera.tensors) == 4
        assert len(mera.isometries) == 4
    
    def test_mera_baseline_snapshot(self):
        mera = MERASpacetime(layers=3, bond_dim=2)
        mera.snapshot_flat_baseline()
        assert hasattr(mera, '_baseline')
        assert len(mera._baseline) == 3
    
    def test_delta_tensor_energy(self):
        mera = MERASpacetime(layers=2, bond_dim=2)
        mera.snapshot_flat_baseline()
        
        # Should be zero initially
        delta = mera.delta_tensor_energy()
        assert delta == 0.0
        
        # Modify a tensor
        mera.tensors[0] *= 1.1
        delta = mera.delta_tensor_energy()
        assert delta > 0.0

class TestTimeOperator:
    """Test time evolution."""
    
    def test_time_operator_creation(self):
        time_op = TimeOperator(dim_clock=8)
        assert time_op.dim_clock == 8
        assert time_op.H_int.shape == (8, 8)
    
    def test_path_evolution(self):
        time_op = TimeOperator(dim_clock=4)
        U = time_op.path_conditioned_evolution(1.0)
        assert U.shape == (4, 4)
        # Check unitarity (approximately)
        UU_dag = U @ U.conj().T
        eye = torch.eye(4, dtype=torch.complex64)
        assert torch.allclose(UU_dag, eye, atol=1e-6)

class TestGaugeField:
    """Test gauge field operations."""
    
    def test_gauge_field_creation(self):
        field = ProperTimeGaugeField(grid_size=16, device="cpu")
        assert field.grid_size == 16
        assert field.A_mu.shape == (4, 16, 16)
        assert field.upsilon_tensor.shape == (4, 4, 16, 16)
    
    def test_zero_fields(self):
        field = ProperTimeGaugeField(grid_size=8, device="cpu")
        # Add some non-zero values
        field.A_mu[0] = 1.0
        field.upsilon_tensor[0, 1] = 2.0
        
        # Zero them
        field.zero_fields()
        assert torch.allclose(field.A_mu, torch.zeros_like(field.A_mu))
        assert torch.allclose(field.upsilon_tensor, torch.zeros_like(field.upsilon_tensor))
    
    def test_is_flat(self):
        field = ProperTimeGaugeField(grid_size=8, device="cpu")
        assert field.is_flat()  # Should be flat initially

class TestUtils:
    """Test utility functions."""
    
    def test_hypercube_bits(self):
        from quantum_core.pkgs.core_physics.utils import _hypercube_bits
        bits = _hypercube_bits(5)  # 5 = 0101 in binary
        assert bits == (1, 0, 1, 0)
    
    def test_embed_4d_to_2d(self):
        from quantum_core.pkgs.core_physics.utils import _embed_4d_to_2d
        pos = _embed_4d_to_2d((1, 0, 1, 0), 8, 8)
        assert isinstance(pos, tuple)
        assert len(pos) == 2
        assert 0 <= pos[0] < 8
        assert 0 <= pos[1] < 8
    
    def test_finite_differences(self):
        from quantum_core.pkgs.core_physics.utils import _ddx, _ddy
        x = torch.randn(3, 4, 5)
        dx = _ddx(x)
        dy = _ddy(x)
        assert dx.shape == x.shape
        assert dy.shape == x.shape

if __name__ == "__main__":
    pytest.main([__file__])