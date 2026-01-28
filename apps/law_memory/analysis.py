"""Light weight attractor analysis helpers for law memory simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass
class FixedPointReport:
    """Container summarising the fixed point check for a trajectory."""

    is_fixed: bool
    delta: np.ndarray


def detect_fixed_point(law_field: np.ndarray, epsilon: float = 1e-6) -> FixedPointReport:
    """Return a :class:`FixedPointReport` describing terminal behaviour.

    Parameters
    ----------
    law_field:
        Array with shape ``(time, space)`` describing the evolution of the law
        field.
    epsilon:
        Threshold that determines when the final two slices are considered
        identical.
    """

    if law_field.ndim != 2:
        raise ValueError("law_field must be two dimensional")
    if law_field.shape[0] < 2:
        raise ValueError("law_field requires at least two time slices")

    delta = np.abs(law_field[-1] - law_field[-2])
    is_fixed = bool(np.all(delta < epsilon))
    return FixedPointReport(is_fixed=is_fixed, delta=delta)


def compute_coherence_embedding(law_field: np.ndarray) -> np.ndarray:
    """Return a Gram-like embedding used for attractor analysis."""

    if law_field.ndim != 2:
        raise ValueError("law_field must be two dimensional")

    centred = law_field - law_field.mean(axis=0, keepdims=True)
    embedding = centred @ centred.T
    return embedding / max(law_field.shape[1], 1)


def classify_trajectory(law_field: np.ndarray, threshold: float = 1e-3) -> str:
    """Classify a trajectory as ``fixed_point``, ``drift`` or ``oscillation``."""

    if law_field.ndim != 2:
        raise ValueError("law_field must be two dimensional")

    diff = np.diff(law_field, axis=0)
    rms = np.sqrt(np.mean(diff**2, axis=1))
    mean_rms = float(np.mean(rms))

    if np.all(rms < threshold):
        return "fixed_point"
    if mean_rms < 5.0 * threshold:
        return "drift"
    return "oscillation"


def cluster_trajectories(
    trajectories: Sequence[np.ndarray], threshold: float = 1e-3
) -> List[str]:
    """Return a label for each trajectory using :func:`classify_trajectory`."""

    labels: List[str] = []
    for traj in trajectories:
        labels.append(classify_trajectory(traj, threshold=threshold))
    return labels


__all__ = [
    "FixedPointReport",
    "detect_fixed_point",
    "compute_coherence_embedding",
    "classify_trajectory",
    "cluster_trajectories",
]

