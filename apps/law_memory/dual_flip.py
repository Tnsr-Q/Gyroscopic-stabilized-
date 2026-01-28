"""PT-dual manipulation utilities for the law memory field."""

from __future__ import annotations

import numpy as np


def reverse_pt_dual(law_stack: np.ndarray, phi_shift: float = 0.0) -> np.ndarray:
    """Return a PT-dual reversed copy of ``law_stack``."""

    if law_stack.ndim != 2:
        raise ValueError("law_stack must be 2D")
    reversed_stack = law_stack[::-1].copy()
    width = reversed_stack.shape[1]
    shift = int(round(phi_shift * width / (2 * np.pi)))
    if shift:
        reversed_stack = np.roll(reversed_stack, shift=shift, axis=1)
    return reversed_stack


def compute_loop_entropy(forward: np.ndarray, reverse: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the point-wise and mean entropy difference between two stacks."""

    if forward.shape != reverse.shape:
        raise ValueError("stack dimensions must match")
    forward_c = forward - forward.mean(axis=1, keepdims=True)
    reverse_c = reverse - reverse.mean(axis=1, keepdims=True)
    diff = np.abs(np.var(forward_c, axis=0) - np.var(reverse_c, axis=0))
    return diff, float(np.mean(diff))


__all__ = ["reverse_pt_dual", "compute_loop_entropy"]
