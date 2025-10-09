"""Torsion map rendering helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:  # pragma: no cover - optional dependency
    import plotly.graph_objects as go
except Exception:  # pragma: no cover
    go = None


@dataclass
class TorsionMapResult:
    r: np.ndarray
    theta: np.ndarray
    torsion: np.ndarray


def compute_torsion_map(chi_int: float, perturb: float = 0.0, size: int = 100) -> TorsionMapResult:
    """Return torsion norm values across polar coordinates."""

    r = np.linspace(1, 10, size)
    theta = np.linspace(0, 2 * np.pi, size)
    R, TH = np.meshgrid(r, theta)
    golden = (1 + np.sqrt(5.0)) / 2.0
    base_phase = (2 * np.pi / 3.0) * R + perturb * golden
    torsion = np.abs(np.sin(base_phase)) * chi_int
    return TorsionMapResult(r=R, theta=TH, torsion=torsion)


def make_torsion_heatmap(result: TorsionMapResult, chaos_threshold: float = 0.5):  # pragma: no cover
    if go is None:
        raise RuntimeError("plotly is required to generate torsion heatmaps")

    fig = go.Figure(
        data=(
            go.Heatmap(
                x=result.r[0],
                y=result.theta[:, 0],
                z=result.torsion,
                colorscale="Viridis",
                colorbar=dict(title="|χ|"),
            )
        )
    )
    fig.update_layout(
        title="Torsion Norm Explorer",
        xaxis_title="r",
        yaxis_title="θ",
        annotations=[
            dict(
                text="chaos > 0.5",
                xref="paper",
                yref="paper",
                x=1.02,
                y=1.05,
                showarrow=False,
            )
        ],
    )
    return fig


__all__ = ["TorsionMapResult", "compute_torsion_map", "make_torsion_heatmap"]
