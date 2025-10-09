"""Temporal echo playback utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

try:  # pragma: no cover
    import plotly.graph_objects as go
except Exception:  # pragma: no cover
    go = None


@dataclass
class EchoPlayback:
    frames: np.ndarray
    timestamps: np.ndarray


def build_echo_frames(law_tensor: np.ndarray, stride: int = 1) -> EchoPlayback:
    """Prepare frames for the temporal echo dashboard."""

    if law_tensor.ndim != 3:
        raise ValueError("Expected law tensor with shape [time, r, θ]")
    frames = law_tensor[::stride]
    timestamps = np.arange(frames.shape[0])
    return EchoPlayback(frames=frames, timestamps=timestamps)


def make_echo_animation(echo: EchoPlayback):  # pragma: no cover
    if go is None:
        raise RuntimeError("plotly is required for echo playback animations")

    frames = []
    for idx, frame in enumerate(echo.frames):
        frames.append(
            go.Frame(
                data=[go.Heatmap(z=frame, colorscale="Plasma")],
                name=str(idx),
            )
        )
    fig = go.Figure(
        data=[go.Heatmap(z=echo.frames[0], colorscale="Plasma")],
        layout=go.Layout(
            title="Temporal Echo Playback",
            updatemenus=[
                dict(
                    type="buttons",
                    buttons=[
                        dict(label="Play", method="animate", args=[[str(i) for i in range(len(frames))]])
                    ],
                )
            ],
            sliders=[
                dict(
                    steps=[dict(method="animate", args=[[str(i)]], label=str(i)) for i in range(len(frames))]
                )
            ],
        ),
        frames=frames,
    )
    return fig


__all__ = ["EchoPlayback", "build_echo_frames", "make_echo_animation"]
