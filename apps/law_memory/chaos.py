"""Chaos and torsion diagnostics for law memory simulations."""

from __future__ import annotations

import numpy as np


def compute_torsion_spectrum(gamma: np.ndarray) -> np.ndarray:
    """Return a simple frequency spectrum that acts as a chaos proxy."""

    if gamma.ndim != 1:
        raise ValueError("gamma must be one dimensional")
    delta = np.gradient(gamma)
    torsion_like = np.abs(np.sin(delta))
    fft = np.fft.rfft(torsion_like)
    return np.abs(fft)


__all__ = ["compute_torsion_spectrum"]
