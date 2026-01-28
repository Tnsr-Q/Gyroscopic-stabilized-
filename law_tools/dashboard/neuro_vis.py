"""Neuro-semantic coupling visualisations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

try:  # pragma: no cover - optional
    import plotly.graph_objects as go
except Exception:  # pragma: no cover
    go = None


@dataclass
class NeuroSemanticTrace:
    time: np.ndarray
    semantic_phase: np.ndarray
    chi_stream: np.ndarray


def build_neuro_semantic_trace(tokens: Iterable[float]) -> NeuroSemanticTrace:
    """Construct a synthetic semantic drift trace from GPT tokens."""

    tokens_array = np.asarray(list(tokens), dtype=float)
    if tokens_array.size == 0:
        tokens_array = np.zeros(1)
    time = np.linspace(0, tokens_array.size - 1, tokens_array.size)
    phase = np.unwrap(np.angle(np.exp(1j * tokens_array)))
    chi_stream = np.gradient(tokens_array)
    return NeuroSemanticTrace(time=time, semantic_phase=phase, chi_stream=chi_stream)


def make_neuro_figure(trace: NeuroSemanticTrace):  # pragma: no cover
    if go is None:
        raise RuntimeError("plotly is required for neuro semantic visualisations")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=trace.time, y=trace.semantic_phase, mode="lines", name="∇argΨ"),
    )
    fig.add_trace(
        go.Scatter(
            x=trace.time,
            y=trace.chi_stream,
            mode="lines",
            name="χ-stream",
            yaxis="y2",
        )
    )
    fig.update_layout(
        title="NeuroSemantic Coupling",
        xaxis_title="Time",
        yaxis=dict(title="Semantic Phase"),
        yaxis2=dict(
            title="χ stream",
            overlaying="y",
            side="right",
        ),
    )
    return fig


__all__ = ["NeuroSemanticTrace", "build_neuro_semantic_trace", "make_neuro_figure"]
