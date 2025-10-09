"""Utilities for converting BEC observables into semantic fields."""

from __future__ import annotations

import numpy as np


def density_to_gamma(density: np.ndarray) -> np.ndarray:
    """Convert a condensate density array into the γ coherence field."""

    density = np.asarray(density, dtype=float)
    max_val = np.max(density)
    if max_val == 0:
        return np.zeros_like(density)
    return density / max_val


__all__ = ["density_to_gamma"]
