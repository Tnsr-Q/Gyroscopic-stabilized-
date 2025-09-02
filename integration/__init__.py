"""
Integration utilities for connecting physics modules to the main engine.

This package contains integration hooks for:
- Connecting new physics modules to RecursiveConformalComputing
- Factory functions for sensor, compute, and actuate interfaces
"""

from .hooks_core import (
    make_mt_wormhole, 
    sense_fn_factory, 
    compute_fn_factory, 
    actuate_fn_factory
)

__all__ = [
    'make_mt_wormhole',
    'sense_fn_factory', 
    'compute_fn_factory',
    'actuate_fn_factory'
]