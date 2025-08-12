"""
Core physics components for quantum computing and tensor networks.

This package contains the fundamental physics classes separated from 
orchestration logic as part of the modular architecture migration.
"""

# Lattice structures
from .lattice import Edge, BondGraph, build_tesseract_lattice

# Mathematical utilities  
from .utils import _ddx, _ddy, _hypercube_bits, _embed_4d_to_2d, RGPlanner

# Field theory components
from .fields import UVIRRegulator, AlenaSoul, ProperTimeGaugeField

# Control systems
from .control import SignatureNormalizer, JacobianHypercube

# Time operations
from .timeops import TimeOperator, TimeRecompressionGate

# Tensor networks
from .tensors import MERASpacetime, estimate_spectral_dim_from_cut, leakage_ceiling

__all__ = [
    # Lattice
    'Edge', 'BondGraph', 'build_tesseract_lattice',
    # Utils
    '_ddx', '_ddy', '_hypercube_bits', '_embed_4d_to_2d', 'RGPlanner',
    # Fields
    'UVIRRegulator', 'AlenaSoul', 'ProperTimeGaugeField', 
    # Control
    'SignatureNormalizer', 'JacobianHypercube',
    # Time operations
    'TimeOperator', 'TimeRecompressionGate',
    # Tensors
    'MERASpacetime', 'estimate_spectral_dim_from_cut', 'leakage_ceiling'
]
