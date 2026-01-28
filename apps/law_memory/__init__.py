"""Law memory system core package.

This package implements the recursive law feedback architecture (RLFA) used
throughout the project.  The modules provide numerical solvers, CLI tools,
training utilities and supporting analytics used by the gyroscopic
stabilisation stack.
"""

from .core import LawMemoryConfig, LawMemorySimulator, LawMemoryState

__all__ = [
    "LawMemoryConfig",
    "LawMemorySimulator",
    "LawMemoryState",
]
