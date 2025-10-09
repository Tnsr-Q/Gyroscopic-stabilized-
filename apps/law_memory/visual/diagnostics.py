"""Diagnostic plots used across the law memory system."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_entropy_loop(law_field: np.ndarray, gamma: np.ndarray, output: Path) -> None:
    """Plot the mean entropy difference between forward and reverse loops."""

    forward = law_field
    reverse = forward[::-1]
    entropy = np.abs(forward - reverse).mean(axis=0)
    plt.figure(figsize=(6, 4))
    plt.plot(entropy, label="|Δℒ|")
    plt.plot(gamma, label="γ", alpha=0.7)
    plt.xlabel("Grid index")
    plt.ylabel("Amplitude")
    plt.title("Entropy loop diagnostic")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def plot_law_attractor(law_field: np.ndarray, output: Path) -> None:
    """Visualise the attractor landscape by averaging over time."""

    attractor = law_field.mean(axis=0)
    plt.figure(figsize=(6, 4))
    plt.imshow(law_field, aspect="auto", cmap="plasma")
    plt.colorbar(label="ℒ")
    plt.plot(attractor, color="white", linewidth=1.0)
    plt.title("Law attractor evolution")
    plt.xlabel("Space index")
    plt.ylabel("Time index")
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


__all__ = ["plot_entropy_loop", "plot_law_attractor"]
