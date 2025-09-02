"""
Physics modules for advanced wormhole dynamics and quantum field theory.

This package contains physics primitives for:
- Morris-Thorne wormhole geometries
- Einstein-Cartan torsion effects  
- Gamma-coherence feedback mechanisms
"""

from .wormhole_mt import MTWormhole
from .torsion_ec import torsion_spring_force, torsion_effective_stiffness
from .coherence_gamma import coherence_mass_coupling, gamma_feedback_step

__all__ = [
    'MTWormhole',
    'torsion_spring_force', 
    'torsion_effective_stiffness',
    'coherence_mass_coupling',
    'gamma_feedback_step'
]