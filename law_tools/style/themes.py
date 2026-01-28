"""Dashboard styling utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Theme:
    name: str
    background: str
    foreground: str
    accent: str
    grid: str


THEMES: Dict[str, Theme] = {
    "Mandelbrot": Theme(
        name="Mandelbrot",
        background="#0b032d",
        foreground="#f0f3f9",
        accent="#845ec2",
        grid="#1f3b4d",
    ),
    "Steel": Theme(
        name="Steel",
        background="#1c1f26",
        foreground="#f5f7fa",
        accent="#5dade2",
        grid="#2f3640",
    ),
    "NeuroDark": Theme(
        name="NeuroDark",
        background="#060612",
        foreground="#e6f4ff",
        accent="#00f5d4",
        grid="#232946",
    ),
    "QuantumDark": Theme(
        name="QuantumDark",
        background="#050913",
        foreground="#ecf0f1",
        accent="#9b59b6",
        grid="#1b1f3a",
    ),
}


def get_theme(name: str) -> Theme:
    """Return the requested theme or fall back to ``QuantumDark``."""

    return THEMES.get(name, THEMES["QuantumDark"])


__all__ = ["THEMES", "Theme", "get_theme"]
