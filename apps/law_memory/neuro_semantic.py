"""Neuro semantic coupling utilities for the law memory system."""

from __future__ import annotations

import numpy as np


def compute_chi_int(psi: np.ndarray, epsilon: float = 1e-9) -> np.ndarray:
    """Return the χ-int field derived from ``psi``."""

    phase = np.angle(psi)
    gradients = np.gradient(phase)
    norm_sq = np.zeros_like(phase, dtype=float)
    for grad in gradients:
        norm_sq += grad**2
    return norm_sq + epsilon


def inject_chi_int_coupling(law: np.ndarray, chi_int: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    """Scale ``law`` using the χ-int feedback field."""

    if law.shape != chi_int.shape:
        raise ValueError("shape mismatch between law and χ-int")
    mean = np.mean(chi_int)
    scaling = 1.0 + chi_int / (mean + epsilon)
    return law * scaling


def generate_mock_psi(shape: tuple[int, ...], seed: int = 42) -> np.ndarray:
    """Generate a reproducible mock semantic wavefunction."""

    rng = np.random.default_rng(seed)
    magnitude = 1.0 + 0.05 * rng.standard_normal(shape)
    phase = 2 * np.pi * rng.random(shape)
    return magnitude * np.exp(1j * phase)


__all__ = ["compute_chi_int", "inject_chi_int_coupling", "generate_mock_psi"]
