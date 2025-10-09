"""Advanced operator training helpers for GPT aligned control."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from .training_suite import LawManifoldEncoder


@dataclass
class OperatorBasis:
    """Light weight PCA-like container."""

    mean: np.ndarray
    components: np.ndarray


class LawOperatorTrainer:
    """Analyse and mutate law operator trajectories."""

    def __init__(self, law_tensor: np.ndarray, n_components: int = 5) -> None:
        if law_tensor.ndim != 2:
            raise ValueError("law_tensor must be (time, space)")
        self.law_tensor = np.asarray(law_tensor, dtype=float)
        self.n_components = n_components
        self.operators: np.ndarray | None = None
        self.basis: OperatorBasis | None = None
        self.reduce()

    def clone(self) -> "LawOperatorTrainer":
        clone = LawOperatorTrainer(self.law_tensor.copy(), self.n_components)
        clone.operators = None if self.operators is None else self.operators.copy()
        clone.basis = None if self.basis is None else OperatorBasis(
            self.basis.mean.copy(), self.basis.components.copy()
        )
        return clone

    def reduce(self) -> np.ndarray:
        mean = self.law_tensor.mean(axis=0)
        centred = self.law_tensor - mean
        u, s, vt = np.linalg.svd(centred, full_matrices=False)
        components = vt[: self.n_components]
        operators = u[:, : self.n_components] * s[: self.n_components]
        self.basis = OperatorBasis(mean=mean, components=components)
        self.operators = operators
        return operators

    def reconstruct(self) -> np.ndarray:
        if self.basis is None or self.operators is None:
            raise RuntimeError("basis not computed")
        reconstruction = self.operators @ self.basis.components
        return reconstruction + self.basis.mean

    def add_semantic_projection_layer(self, j_mu: np.ndarray, coupling_weight: float = 0.618) -> np.ndarray:
        if j_mu.shape != self.law_tensor.shape:
            raise ValueError("j_mu shape mismatch")
        norm = np.linalg.norm(j_mu, axis=0)
        self.law_tensor = self.law_tensor * (1 + coupling_weight * norm)
        return self.reduce()

    def plot_operator_drift(self, path: Path) -> None:
        if self.operators is None:
            raise RuntimeError("operators not available")
        plt.figure(figsize=(7, 4))
        for idx in range(self.operators.shape[1]):
            plt.plot(self.operators[:, idx], label=f"𝒪{idx}")
        plt.xlabel("Time index")
        plt.ylabel("Operator value")
        plt.title("Operator drift")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def attach_gpt_guidance(self, suggestion_file: Path, influence_strength: float = 0.25) -> np.ndarray:
        if self.operators is None:
            raise RuntimeError("operators not available")
        cfg = yaml.safe_load(Path(suggestion_file).read_text())
        mutate = cfg.get("mutate", {})
        weights = mutate.get("operator_weights", [0.0] * self.operators.shape[1])
        if len(weights) != self.operators.shape[1]:
            raise ValueError("operator_weights size mismatch")
        bias = np.asarray(weights) * influence_strength
        self.operators = self.operators + bias[np.newaxis, :]
        if mutate.get("shift_basis", False):
            self.reduce()
        if mutate.get("reproject_after", True):
            self.law_tensor = self.reconstruct()
        return self.operators

    def generate_token_batch(self, prompts_yaml: Path, output_dir: Path) -> List[Path]:
        specs = yaml.safe_load(Path(prompts_yaml).read_text())
        output_dir.mkdir(parents=True, exist_ok=True)
        generated: List[Path] = []
        for idx, spec in enumerate(specs):
            name = spec.get("name", f"token_stream_{idx}")
            suggestion_path = output_dir / f"{name.replace(' ', '_')}.yaml"
            suggestion_path.write_text(yaml.dump(spec))

            clone = self.clone()
            clone.attach_gpt_guidance(suggestion_path)
            encoder = LawManifoldEncoder(clone.law_tensor)
            token = encoder.encode()
            token_path = output_dir / f"law_tokens_{idx}.json"
            token_path.write_text(json.dumps(token.to_dict(), indent=2))
            generated.append(token_path)
        return generated

    def export_tokens(self, path: Path) -> None:
        encoder = LawManifoldEncoder(self.law_tensor)
        token = encoder.encode()
        path.write_text(json.dumps(token.to_dict(), indent=2))


__all__ = ["LawOperatorTrainer"]
