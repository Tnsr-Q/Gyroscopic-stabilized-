"""
Architectural patterns for quantum control systems.

This package contains architectural patterns for:
- Brontein Cube sense-compute-actuate loops
- Multi-scale control architectures
"""

from .brontein_cube import BronteinCube

__all__ = [
    'BronteinCube'
]