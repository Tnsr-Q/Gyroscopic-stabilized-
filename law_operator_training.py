"""Law operator training utilities for GPT-integrated law evolution workflows.

This module exposes the :class:`LawOperatorTrainer`, a lightweight harness that
compresses high dimensional law tensors into GPT manageable operator tokens,
allows semantic projections, visualises operator drift and integrates external
GPT guidance loops.
"""
from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
import yaml
from sklearn.decomposition import PCA


@dataclass
class _PCAMetadata:
    """Container that stores PCA artefacts required for reconstruction."""

    components: np.ndarray
    mean: np.ndarray


class LawOperatorTrainer:
    """Distil law tensors into GPT-accessible operator representations."""

    def __init__(
        self,
        law_field_file: str,
        n_components: int = 5,
        random_seed: Optional[int] = None,
    ) -> None:
        self.law_field_file = law_field_file
        self.n_components = n_components
        self.random_seed = random_seed
        self.operators: Optional[np.ndarray] = None
        self.law_tensor: Optional[np.ndarray] = None
        self.meta: Dict[str, Any] = {}
        self._pca_model: Optional[PCA] = None
        self._pca_meta: Optional[_PCAMetadata] = None

    # ------------------------------------------------------------------
    # Data IO helpers
    # ------------------------------------------------------------------
    def load_law_field(self) -> np.ndarray:
        """Load the law tensor from the checkpoint file."""
        with h5py.File(self.law_field_file, "r") as f:
            if "law_tensor" not in f:
                raise KeyError("Missing 'law_tensor' dataset in checkpoint")
            self.law_tensor = f["law_tensor"][:]
            self.meta["coords"] = {key: f.attrs[key] for key in f.attrs}
        return self.law_tensor

    # ------------------------------------------------------------------
    # Core dimensionality reduction
    # ------------------------------------------------------------------
    def reduce_operators(self) -> np.ndarray:
        """Perform PCA over the law tensor to obtain operator trajectories."""
        if self.law_tensor is None:
            raise RuntimeError("Law tensor not loaded. Call 'load_law_field' first.")

        t_steps, radial, angular = self.law_tensor.shape
        flattened = self.law_tensor.reshape(t_steps, radial * angular)

        self._pca_model = PCA(n_components=self.n_components, svd_solver="full")
        self.operators = self._pca_model.fit_transform(flattened)
        self._pca_meta = _PCAMetadata(
            components=self._pca_model.components_.copy(),
            mean=self._pca_model.mean_.copy(),
        )
        self.meta["pca_explained_variance"] = self._pca_model.explained_variance_.tolist()
        return self.operators

    # ------------------------------------------------------------------
    # Semantic projection
    # ------------------------------------------------------------------
    def add_semantic_projection_layer(
        self, j_mu_file: str, coupling_weight: float = 0.618
    ) -> np.ndarray:
        """Project the law tensor into a semantic basis driven by ``j^μ`` fields."""
        if self.law_tensor is None:
            raise RuntimeError("Law tensor not loaded. Call 'load_law_field' first.")

        with h5py.File(j_mu_file, "r") as f:
            if "j_semantic" not in f:
                raise KeyError("Missing 'j_semantic' dataset in semantic file")
            j_mu = f["j_semantic"][:]

        norm = np.linalg.norm(j_mu, axis=-1)
        if norm.shape != self.law_tensor.shape:
            raise ValueError(
                "Semantic current shape does not match law tensor shape."
            )

        projected_tensor = self.law_tensor * (1.0 + coupling_weight * norm)
        flattened = projected_tensor.reshape(projected_tensor.shape[0], -1)

        self._pca_model = PCA(n_components=self.n_components, svd_solver="full")
        self.operators = self._pca_model.fit_transform(flattened)
        self._pca_meta = _PCAMetadata(
            components=self._pca_model.components_.copy(),
            mean=self._pca_model.mean_.copy(),
        )
        self.meta["semantic_projection"] = {
            "j_mu_file": j_mu_file,
            "coupling_weight": coupling_weight,
        }
        return self.operators

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------
    def export_gpt_tokens(self, out_file: str = "law_tokens.yaml") -> Dict[str, Any]:
        """Persist operator tokens to a YAML file."""
        if self.operators is None:
            raise RuntimeError("Operators not available. Run reduction first.")

        tokens = {f"𝒪_{i}": self.operators[:, i].tolist() for i in range(self.n_components)}
        payload = {"law_tokens": tokens, "meta": self.meta}
        with open(out_file, "w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=True)
        return payload

    # ------------------------------------------------------------------
    # Operator diagnostics and utilities
    # ------------------------------------------------------------------
    def plot_operator_drift(self, out_file: str = "operator_drift.png") -> None:
        """Plot the temporal drift of each operator trajectory."""
        if self.operators is None:
            raise RuntimeError("Operators not available. Run reduction first.")

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        t_steps = self.operators.shape[0]
        plt.figure(figsize=(10, 6))
        for idx in range(self.n_components):
            plt.plot(range(t_steps), self.operators[:, idx], label=f"𝒪_{idx}")
        plt.xlabel("Time Step")
        plt.ylabel("Operator Value")
        plt.title("Evolution of Law Operators (𝒪ᵢ)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_file)

    def attach_gpt_guidance(
        self, suggestion_file: str, influence_strength: float = 0.25
    ) -> None:
        """Inject GPT supplied mutations over the operator basis."""
        if self.operators is None:
            raise RuntimeError("Operators not available. Run reduction first.")

        with open(suggestion_file, "r", encoding="utf-8") as handle:
            gpt_cfg = yaml.safe_load(handle)

        if "mutate" not in gpt_cfg:
            raise ValueError("Suggestion file missing 'mutate' directive")

        guidance = gpt_cfg["mutate"]
        op_weights: List[float] = guidance.get(
            "operator_weights", [0.0] * self.n_components
        )
        shift_basis: bool = guidance.get("shift_basis", False)
        reproject: bool = guidance.get("reproject_after", True)

        if len(op_weights) != self.n_components:
            raise ValueError(
                f"Expected {self.n_components} operator weights, received {len(op_weights)}"
            )

        bias_vector = np.asarray(op_weights, dtype=float) * float(influence_strength)
        self.operators = self.operators + bias_vector[np.newaxis, :]

        if shift_basis:
            self._recenter_operator_basis()
        if reproject:
            self.reproject_operators_to_law()

        self.meta["gpt_guidance"] = {
            "source": suggestion_file,
            "strength": influence_strength,
            "bias_applied": op_weights,
            "shift_basis": shift_basis,
        }

    def reproject_operators_to_law(self) -> np.ndarray:
        """Reconstruct the law tensor from the current operators."""
        if self.operators is None or self._pca_model is None:
            raise RuntimeError("Operators or PCA model unavailable for reprojection.")

        reconstructed = self._pca_model.inverse_transform(self.operators)
        if self.law_tensor is None:
            raise RuntimeError("Law tensor shape unknown. Load checkpoint first.")

        self.law_tensor = reconstructed.reshape(self.law_tensor.shape)
        return self.law_tensor

    def clone(self) -> "LawOperatorTrainer":
        """Return a deep copy of the trainer with duplicated state."""
        clone = LawOperatorTrainer(
            self.law_field_file, self.n_components, random_seed=self.random_seed
        )
        if self.law_tensor is not None:
            clone.law_tensor = self.law_tensor.copy()
        if self.operators is not None:
            clone.operators = self.operators.copy()
        if self._pca_model is not None:
            clone._pca_model = deepcopy(self._pca_model)
        if self._pca_meta is not None:
            clone._pca_meta = deepcopy(self._pca_meta)
        clone.meta = deepcopy(self.meta)
        return clone

    def generate_token_batch(
        self, prompts_yaml: str, output_dir: str = "tokens_batch", seed: Optional[int] = 42
    ) -> None:
        """Generate batches of GPT token streams under variant prompt mutations."""
        if self.operators is None:
            raise RuntimeError("Operators not available. Run reduction first.")

        os.makedirs(output_dir, exist_ok=True)
        with open(prompts_yaml, "r", encoding="utf-8") as handle:
            prompt_list = yaml.safe_load(handle) or []

        rng = np.random.default_rng(seed)
        for idx, prompt in enumerate(prompt_list):
            name = prompt.get("name", f"token_stream_{idx}")
            prompt_path = os.path.join(
                output_dir, f"{name.replace(' ', '_')}.yaml"
            )
            with open(prompt_path, "w", encoding="utf-8") as destination:
                yaml.safe_dump(prompt, destination, sort_keys=False)

            trainer_clone = self.clone()
            influence = prompt.get("influence_strength", 0.4)
            if "mutate" in prompt:
                trainer_clone.attach_gpt_guidance(
                    prompt_path, influence_strength=influence
                )
            token_file = os.path.join(output_dir, f"law_tokens_{idx}.yaml")
            drift_file = os.path.join(output_dir, f"drift_{idx}.png")
            trainer_clone.export_gpt_tokens(token_file)
            trainer_clone.plot_operator_drift(drift_file)

            noise = float(rng.normal(0, 1e-12))
            trainer_clone.meta.setdefault("batch_noise", []).append(noise)

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    def get_operator_basis(self) -> np.ndarray:
        if self._pca_meta is None:
            raise RuntimeError("Operator basis unavailable. Run reduction first.")
        return self._pca_meta.components.copy()

    def _recenter_operator_basis(self) -> None:
        if self.operators is None or self._pca_model is None:
            raise RuntimeError("Operators not available to recenter basis.")

        flattened = self._pca_model.inverse_transform(self.operators)
        self._pca_model = PCA(n_components=self.n_components, svd_solver="full")
        self.operators = self._pca_model.fit_transform(flattened)
        self._pca_meta = _PCAMetadata(
            components=self._pca_model.components_.copy(),
            mean=self._pca_model.mean_.copy(),
        )

    def shift_operator_basis(self) -> None:
        """Alias retained for backwards compatibility."""
        self._recenter_operator_basis()

    def reproject_onto_basis(self) -> np.ndarray:
        return self.reproject_operators_to_law()


__all__ = ["LawOperatorTrainer"]
