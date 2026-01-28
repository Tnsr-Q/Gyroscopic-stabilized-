"""Law operator training utilities for GPT-integrated law evolution workflows.

This module exposes the :class:`LawOperatorTrainer`, a lightweight harness that
compresses high dimensional law tensors into GPT manageable operator tokens,
allows semantic projections, visualises operator drift and integrates external
GPT guidance loops.  It also offers helper functions that implement the
``LawManifoldEncoder``/``LawTokenVector`` workflow described in the project
documentation together with a small command line driver that wires everything
into a closed loop for prompt generation.
"""
from __future__ import annotations

import argparse
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional

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


__all__ = [
    "LawOperatorTrainer",
    "encode_law_manifold",
    "law_token_vector",
    "inject_into_prompt",
    "main",
]


# ---------------------------------------------------------------------------
# Law manifold encoding helpers
# ---------------------------------------------------------------------------
def _iter_matching_keys(container: Mapping[str, Any], prefix: str) -> Iterable[str]:
    """Yield keys in ``container`` that start with ``prefix`` sorted by suffix.

    The RCC checkpoints that inspired this module follow the convention of
    storing samples under keys such as ``traj_0``, ``torsion_norm_0`` and so on.
    The helper performs the minor bookkeeping that allows us to iterate over
    them in a deterministic order while remaining resilient to sparse or
    partially missing entries.
    """

    def suffix(key: str) -> int:
        try:
            return int(key[len(prefix) :])
        except ValueError:
            return -1

    return (key for key in sorted(container.keys(), key=suffix) if key.startswith(prefix))


def encode_law_manifold(checkpoint: str) -> Dict[str, float]:
    """Extract coarse law statistics from an RCC checkpoint.

    Parameters
    ----------
    checkpoint:
        Path to an ``.h5`` file generated by the RCC solver.

    Returns
    -------
    dict
        Dictionary with the aggregated statistics required for prompt control.

    Notes
    -----
    The function mirrors the pseudo-code shown in the user documentation.  It
    computes the average of all trajectory datasets (``traj_*``), the variance
    of torsion norms (``torsion_norm_*`` attributes) and the mean hysteresis
    area (``hysteresis_area_*`` attributes).  Missing groups are simply ignored
    which keeps the helper permissive for synthetic tests.
    """

    with h5py.File(checkpoint, "r") as handle:
        traj_values: List[np.ndarray] = []
        for key in _iter_matching_keys(handle, "traj_"):
            data = handle[key][()]
            traj_values.append(np.asarray(data, dtype=float))

        if not traj_values:
            raise KeyError("No 'traj_*' datasets present in checkpoint")

        gamma_mean = float(np.mean(traj_values))

        torsion_values: List[float] = []
        hysteresis_values: List[float] = []
        for attr_key in _iter_matching_keys(handle.attrs, "torsion_norm_"):
            torsion_values.append(float(handle.attrs[attr_key]))
        for attr_key in _iter_matching_keys(handle.attrs, "hysteresis_area_"):
            hysteresis_values.append(float(handle.attrs[attr_key]))

        if not torsion_values:
            raise KeyError("No 'torsion_norm_*' attributes present in checkpoint")
        if not hysteresis_values:
            raise KeyError("No 'hysteresis_area_*' attributes present in checkpoint")

        torsion_variance = float(np.var(torsion_values))
        law_memory_weight = float(np.mean(hysteresis_values))

    return {
        "gamma_mean": gamma_mean,
        "torsion_variance": torsion_variance,
        "law_memory_weight": law_memory_weight,
    }


def law_token_vector(encoding: Mapping[str, float]) -> Dict[str, Any]:
    """Convert raw law statistics into GPT friendly control tokens."""

    gamma_mean = float(encoding.get("gamma_mean", 0.0))
    torsion_variance = float(encoding.get("torsion_variance", 0.0))
    law_memory_weight = float(encoding.get("law_memory_weight", 0.0))

    coherence_mode = "stable" if torsion_variance < 0.4 else "chaotic"
    return {
        "LAW_GAMMA_LEVEL": round(gamma_mean, 3),
        "LAW_TORSION_NOISE": round(torsion_variance, 4),
        "LAW_HYSTERESIS_DEPTH": round(law_memory_weight, 4),
        "LAW_COHERENCE_MODE": coherence_mode,
    }


def inject_into_prompt(tokens: Mapping[str, Any], template: str) -> str:
    """Inject control tokens into a prompt template.

    Parameters
    ----------
    tokens:
        Mapping produced by :func:`law_token_vector`.
    template:
        The raw prompt template to which the control block will be prepended.
    """

    control_block = (
        "--RCC LAW CONFIG--\n"
        f"γ_avg = {tokens['LAW_GAMMA_LEVEL']}\n"
        f"χ_var = {tokens['LAW_TORSION_NOISE']}\n"
        f"ℒ_mem = {tokens['LAW_HYSTERESIS_DEPTH']}\n"
        f"Coherence = {tokens['LAW_COHERENCE_MODE']}\n"
        "-------------------\n"
    )
    return f"{control_block}\n{template.strip()}\n"


def _build_prompt(tokens: Mapping[str, Any], template_path: str) -> str:
    with open(template_path, "r", encoding="utf-8") as handle:
        template = handle.read()
    return inject_into_prompt(tokens, template)


def _export_tokens_yaml(tokens: Mapping[str, Any], out_file: str) -> None:
    payload = {"law_tokens": dict(tokens)}
    with open(out_file, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=True)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Closed-loop law prompt trainer")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the RCC checkpoint file (.h5)",
    )
    parser.add_argument(
        "--template",
        required=True,
        help="Prompt template file to inject control tokens into.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination file for the generated prompt.",
    )
    parser.add_argument(
        "--tokens-yaml",
        default=None,
        help="Optional path to export the computed law tokens as YAML.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    encoding = encode_law_manifold(args.checkpoint)
    tokens = law_token_vector(encoding)
    prompt = _build_prompt(tokens, args.template)

    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write(prompt)

    if args.tokens_yaml:
        _export_tokens_yaml(tokens, args.tokens_yaml)


if __name__ == "__main__":  # pragma: no cover - CLI helper
    main()

