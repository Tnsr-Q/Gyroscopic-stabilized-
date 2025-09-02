# physics/torsion_ec.py
from __future__ import annotations
import math
from typing import Callable
from .wormhole_mt import G, c

def torsion_spring_force(spin_density: float, r: float, r0: float, envelope: Callable[[float], float] = None) -> float:
    """
    Simple EC-inspired torsion coupling:
        F_T(r) = (8πG / c^4) * S * exp(-r/r0) * envelope(r)
    S = spin density (units chosen by your runtime)
    """
    env = 1.0 if envelope is None else float(envelope(r))
    return (8.0*math.pi*G/(c**4)) * float(spin_density) * math.exp(-max(r,0.0)/max(r0,1e-9)) * env

def torsion_effective_stiffness(spin_density: float, r0: float) -> float:
    """
    k_T ~ ∫ dr (∂F_T/∂r) at throat scale => sets "spring" strength around r0.
    For the exponential profile, k_T ≈ (8πG/c^4) * S / r0.
    """
    return (8.0*math.pi*G/(c**4)) * float(spin_density) / max(r0, 1e-9)