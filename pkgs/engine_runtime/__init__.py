"""
Engine runtime components for orchestration and state management.

This package contains the runtime orchestration classes separated from
pure physics computations as part of the modular architecture migration.
"""

from .recorder import SimpleRecorder
from .rcc import RecursiveConformalComputing
from .contractor import PrecisionTraversalContractor

__all__ = ['SimpleRecorder', 'RecursiveConformalComputing', 'PrecisionTraversalContractor']
