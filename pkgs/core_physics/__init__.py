"""
Core physics components for quantum computing and tensor networks.

This package contains the fundamental physics classes separated from 
orchestration logic as part of the modular architecture migration.
"""

# Lattice structures
from .lattice import Edge, BondGraph, build_tesseract_lattice

# Mathematical utilities  
from .utils import _ddx, _ddy, _hypercube_bits, _embed_4d_to_2d, RGPlanner, GyroscopeFeeler

# Field theory components
from .fields import UVIRRegulator, AlenaSoul, ProperTimeGaugeField

# Control systems
from .control import SignatureNormalizer, JacobianHypercube

# Time operations
from .timeops import TimeOperator, TimeRecompressionGate

# Tensor networks
from .tensors import MERASpacetime, estimate_spectral_dim_from_cut, leakage_ceiling

# Common data structures
from .common import DecoherenceState, QuantumState

__all__ = [
    # Lattice
    'Edge', 'BondGraph', 'build_tesseract_lattice',
    # Utils
    '_ddx', '_ddy', '_hypercube_bits', '_embed_4d_to_2d', 'RGPlanner', 'GyroscopeFeeler',
    # Fields
    'UVIRRegulator', 'AlenaSoul', 'ProperTimeGaugeField', 
    # Control
    'SignatureNormalizer', 'JacobianHypercube',
    # Time operations
    'TimeOperator', 'TimeRecompressionGate',
    # Tensors
    'MERASpacetime', 'estimate_spectral_dim_from_cut', 'leakage_ceiling',
    # Common
    'DecoherenceState', 'QuantumState'
]
