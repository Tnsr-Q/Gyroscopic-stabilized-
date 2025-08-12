"""Quantum Core - Modular quantum computation engine."""

__version__ = "0.1.0"

# Core components
from .pkgs.core_physics import (
    BondGraph, Edge, build_tesseract_lattice,
    MERASpacetime, 
    ProperTimeGaugeField, AlenaSoul, UVIRRegulator,
    GyroscopeFeeler, SignatureNormalizer, JacobianHypercube,
    TimeOperator, TimeRecompressionGate
)

from .pkgs.engine_runtime import (
    RecursiveConformalComputing,
    PrecisionTraversalContractor,
    SimpleRecorder,
    InitRequest, StepRequest, StepResult
)

from .pkgs.observability import (
    setup_logging,
    MetricsCollector, SeedManager,
    EventBus
)

# High-level service
from .apps.engine.engine_service import EngineService

__all__ = [
    # Core physics
    'BondGraph', 'Edge', 'build_tesseract_lattice',
    'MERASpacetime',
    'ProperTimeGaugeField', 'AlenaSoul', 'UVIRRegulator', 
    'GyroscopeFeeler', 'SignatureNormalizer', 'JacobianHypercube',
    'TimeOperator', 'TimeRecompressionGate',
    
    # Engine runtime
    'RecursiveConformalComputing', 'PrecisionTraversalContractor',
    'SimpleRecorder', 'InitRequest', 'StepRequest', 'StepResult',
    
    # Observability
    'setup_logging', 'MetricsCollector', 'SeedManager', 'EventBus',
    
    # High-level interface
    'EngineService'
]