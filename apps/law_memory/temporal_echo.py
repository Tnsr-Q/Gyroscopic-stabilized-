"""Temporal echo injection helpers."""

from __future__ import annotations

import numpy as np


def inject_temporal_echo(law_stack: np.ndarray, delay: int = 10, decay: float = 0.97) -> np.ndarray:
    """Compute the temporal echo field from a sequence of law states."""

    if law_stack.ndim != 2:
        raise ValueError("law_stack must be 2D")
    steps = law_stack.shape[0]
    echo = np.zeros_like(law_stack[0])
    depth = min(delay, steps)
    for lag in range(1, depth):
        echo += (decay**lag) * law_stack[-lag]
    return echo


__all__ = ["inject_temporal_echo"]
