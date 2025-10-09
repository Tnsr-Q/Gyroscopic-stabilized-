"""Metric coupling helpers for law memory feedback."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def interpolate_law_at_r(r_grid: np.ndarray, law: np.ndarray, query_r: np.ndarray) -> np.ndarray:
    """Return the law profile interpolated onto ``query_r``."""

    if r_grid.ndim != 1:
        raise ValueError("r_grid must be one dimensional")
    if law.shape != r_grid.shape:
        raise ValueError("law must match r_grid")
    return np.interp(query_r, r_grid, law, left=law[0], right=law[-1])


def deformed_metric_profile(
    r: np.ndarray, law: np.ndarray, xi: float = 0.382, lambda_h: float = 0.1457, r0: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the deformed ``phi`` and ``gamma`` profiles."""

    if r.size != law.size:
        raise ValueError("grid mismatch")

    r_safe = np.where(r == 0, 1e-6, r)
    phi = 0.5 * np.log1p(r_safe / r0) + xi * law / (r_safe**2)
    gamma = 1.0 + lambda_h * law
    return phi, gamma


def inject_into_metrics(r: np.ndarray, law: np.ndarray, xi: float = 0.382, lambda_h: float = 0.1457) -> Dict[str, float]:
    """Return summary statistics for the metric deformation."""

    phi, gamma = deformed_metric_profile(r, law, xi=xi, lambda_h=lambda_h)
    return {
        "phi_mean": float(np.mean(phi)),
        "phi_std": float(np.std(phi)),
        "gamma_mean": float(np.mean(gamma)),
        "gamma_std": float(np.std(gamma)),
    }


__all__ = ["interpolate_law_at_r", "deformed_metric_profile", "inject_into_metrics"]
