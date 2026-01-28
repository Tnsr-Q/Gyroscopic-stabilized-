"""Closed loop training utilities for the law memory system."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np


@dataclass
class LawTokenVector:
    """Container for GPT compatible law tokens."""

    gamma_level: float
    torsion_noise: float
    hysteresis_depth: float
    coherence_mode: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "LAW_GAMMA_LEVEL": round(self.gamma_level, 3),
            "LAW_TORSION_NOISE": round(self.torsion_noise, 4),
            "LAW_HYSTERESIS_DEPTH": round(self.hysteresis_depth, 4),
            "LAW_COHERENCE_MODE": self.coherence_mode,
        }


class LawManifoldEncoder:
    """Distil law manifold tensors into GPT compatible embeddings."""

    def __init__(self, law_tensor: np.ndarray) -> None:
        if law_tensor.ndim != 2:
            raise ValueError("law_tensor must be (time, space)")
        self.law_tensor = law_tensor

    def encode(self) -> LawTokenVector:
        gamma_level = float(np.mean(self.law_tensor))
        torsion_noise = float(np.std(np.gradient(self.law_tensor, axis=1)))
        hysteresis_depth = float(np.mean(np.abs(self.law_tensor - self.law_tensor[::-1])))
        coherence_mode = "stable" if torsion_noise < 0.4 else "chaotic"
        return LawTokenVector(gamma_level, torsion_noise, hysteresis_depth, coherence_mode)


def save_tokens(token: LawTokenVector, path: Path) -> None:
    path.write_text(json.dumps({"law_tokens": token.to_dict()}, indent=2))


def inject_tokens(token: LawTokenVector, template: str) -> str:
    block = "\n".join(
        f"{key}: {value}" for key, value in token.to_dict().items()
    )
    return f"-- RCC LAW CONFIG --\n{block}\n\n{template}"


__all__ = [
    "LawTokenVector",
    "LawManifoldEncoder",
    "save_tokens",
    "inject_tokens",
]
