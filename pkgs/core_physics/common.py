"""
Common data structures and enums used across the physics package.

Contains shared types like DecoherenceState and QuantumState that are used
by multiple modules in the physics package.
"""
import numpy as np
from enum import Enum
from dataclasses import dataclass


class DecoherenceState(Enum):
    """Enumeration of quantum decoherence states."""
    COHERENT = "coherent"
    DECOHERING = "decohering" 
    DECOHERENT = "decoherent"
    RECOMPRESSING = "recompressing"
    STABILIZED = "stabilized"


@dataclass
class QuantumState:
    """Quantum state representation with coherence and field information."""
    coherence_gamma: float
    proper_time: float
    torsion: float
    law_field: np.ndarray
    decoherence_state: DecoherenceState
    entropy: float
    phase_memory: float = 0.0