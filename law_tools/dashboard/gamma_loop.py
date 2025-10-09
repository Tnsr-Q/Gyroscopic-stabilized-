"""γ(r, θ) loop utilities for dashboard visualisation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    import plotly.graph_objects as go
except Exception:  # pragma: no cover - gracefully degrade when plotly missing
    go = None


@dataclass
class GammaLoopResult:
    r: np.ndarray
    gamma_forward: np.ndarray
    gamma_reverse: np.ndarray
    loop_area: float


def default_gamma_model(r: np.ndarray, y0: float, phi: float) -> np.ndarray:
    """Base γ(r) profile used when no custom callable is provided."""

    return np.cos(phi) * np.exp(-r / (1.5 + y0)) + np.sin(r * phi) * y0


def compute_gamma_loops(
    y0: float,
    phi: float,
    gamma: Callable[[np.ndarray, float, float], np.ndarray] | None = None,
) -> GammaLoopResult:
    """Compute forward and reverse γ(r) loops."""

    gamma_fn = gamma or default_gamma_model
    r_values = np.linspace(1.0, 10.0, 300)
    gamma_forward = gamma_fn(r_values, y0, phi)
    gamma_reverse = gamma_fn(r_values, y0, -phi)
    loop_area = np.trapz(gamma_forward - gamma_reverse, r_values)
    return GammaLoopResult(r=r_values, gamma_forward=gamma_forward, gamma_reverse=gamma_reverse, loop_area=float(loop_area))


def make_gamma_figure(result: GammaLoopResult):  # pragma: no cover - plotting helper
    if go is None:
        raise RuntimeError("plotly is required to generate the gamma loop figure")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result.r, y=result.gamma_forward, mode="lines", name="γ_forward"))
    fig.add_trace(go.Scatter(x=result.r, y=result.gamma_reverse, mode="lines", name="γ_reverse"))
    fig.update_layout(
        title=f"γ-loop integral (area = {result.loop_area:.3f})",
        xaxis_title="r",
        yaxis_title="γ(r)",
        legend=dict(orientation="h"),
    )
    return fig


__all__ = ["GammaLoopResult", "compute_gamma_loops", "make_gamma_figure", "default_gamma_model"]
