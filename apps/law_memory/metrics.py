"""Metric coupling helpers for law memory feedback."""

from __future__ import annotations

from typing import Dict

import numpy as np


def inject_into_metrics(r: np.ndarray, law: np.ndarray, xi: float = 0.382, lambda_h: float = 0.1457) -> Dict[str, float]:
    """Return updated metric coefficients from a law profile."""

    if r.size != law.size:
        raise ValueError("grid mismatch")
    r_safe = np.where(r == 0, 1e-6, r)
    phi = xi * law / (r_safe**2)
    gamma = 1.0 + lambda_h * law
    return {
        "phi_mean": float(np.mean(phi)),
        "phi_std": float(np.std(phi)),
        "gamma_mean": float(np.mean(gamma)),
        "gamma_std": float(np.std(gamma)),
    }


__all__ = ["inject_into_metrics"]
