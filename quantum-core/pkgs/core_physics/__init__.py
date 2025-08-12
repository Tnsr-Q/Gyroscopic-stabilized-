"""Core physics kernels for quantum computation engine.

This package contains pure computational kernels without side effects:
- Lattice structures and bond graphs
- Tensor operations and isometries  
- Field theory components
- Control utilities
- Time operations
"""

from .lattice import BondGraph, Edge, build_tesseract_lattice
from .tensors import MERASpacetime
from .fields import ProperTimeGaugeField, AlenaSoul, UVIRRegulator
from .control import GyroscopeFeeler, SignatureNormalizer, JacobianHypercube
from .timeops import TimeOperator, TimeRecompressionGate
from .utils import (
    _ddx, _ddy, _hypercube_bits, _embed_4d_to_2d,
    estimate_spectral_dim_from_cut, leakage_ceiling,
    project_to_lightcone
)

__all__ = [
    'BondGraph', 'Edge', 'build_tesseract_lattice',
    'MERASpacetime', 
    'ProperTimeGaugeField', 'AlenaSoul', 'UVIRRegulator',
    'GyroscopeFeeler', 'SignatureNormalizer', 'JacobianHypercube',
    'TimeOperator', 'TimeRecompressionGate',
    '_ddx', '_ddy', '_hypercube_bits', '_embed_4d_to_2d',
    'estimate_spectral_dim_from_cut', 'leakage_ceiling',
    'project_to_lightcone'
]