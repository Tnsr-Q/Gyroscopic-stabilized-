# physics/coherence_gamma.py
from __future__ import annotations
from typing import Tuple, Callable

def coherence_mass_coupling(gamma: float, m0: float, k: float = 1.0) -> float:
    """
    m_eff = m0 * (1 + k * gamma).  gamma>=0 assumed.
    """
    return float(m0) * (1.0 + float(k) * max(gamma, 0.0))

def gamma_feedback_step(gamma: float,
                        proper_time_grad: float,
                        m0: float,
                        k: float = 1.0,
                        eta: float = 1.0,
                        dt: float = 1e-3) -> Tuple[float, float]:
    """
    dγ/dt = -η * m_eff * (∂_τ) * γ
    Returns (gamma_next, m_eff).
    """
    m_eff = coherence_mass_coupling(gamma, m0, k)
    dgamma_dt = -float(eta) * m_eff * float(proper_time_grad) * gamma
    gamma_next = max(0.0, gamma + dt * dgamma_dt)
    return gamma_next, m_eff