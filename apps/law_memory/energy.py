"""Energy accounting helpers for law memory metric updates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EnergyReport:
    """Summary of the energy cost incurred during an update."""

    density: np.ndarray
    total: float


def semantic_energy_density(
    gamma: np.ndarray, psi: np.ndarray, kappa: float = 1.0
) -> np.ndarray:
    """Return the semantic energy density associated with ``∇log γ``."""

    if gamma.shape != psi.shape:
        raise ValueError("gamma and psi must have the same shape")

    gamma_safe = np.clip(gamma, 1e-8, None)
    grad_log_gamma = np.gradient(np.log(gamma_safe))
    grad_norm_sq = np.sum(np.array(grad_log_gamma) ** 2, axis=0)

    rho_sem = np.abs(psi) ** 2
    density = 0.5 * (kappa**2) * rho_sem * grad_norm_sq
    return density


def total_update_energy(
    gamma: np.ndarray, psi: np.ndarray, kappa: float = 1.0
) -> EnergyReport:
    """Return the :class:`EnergyReport` for a single update step."""

    density = semantic_energy_density(gamma, psi, kappa=kappa)
    total = float(np.sum(density))
    return EnergyReport(density=density, total=total)


__all__ = ["EnergyReport", "semantic_energy_density", "total_update_energy"]

