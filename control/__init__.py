"""
Control systems for recursive and multi-scale dynamics.

This package contains control systems for:
- Recursive control loops with fractal structure
- Multi-scale feedback mechanisms
"""

from .recursive_loop import RecursiveController

__all__ = [
    'RecursiveController'
]