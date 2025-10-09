"""Command line pipeline for the law memory system.

The CLI orchestrates the different stages described in the architecture
overview: observable conversion, PDE evolution, diagnostics and visualisation.
Each stage uses the light‑weight implementations contained in this package so
that the command can be executed inside unit tests.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from .core import LawMemoryConfig, LawMemorySimulator
from .visual.diagnostics import plot_entropy_loop, plot_law_attractor
from .chaos import compute_torsion_spectrum
from .metrics import inject_into_metrics
from .observables import density_to_gamma


def _load_density(path: Path) -> np.ndarray:
    data = np.load(path)
    if data.ndim == 1:
        return data[np.newaxis, :]
    if data.ndim != 2:
        raise ValueError("density input must be 1D or 2D")
    return data


def _evolve_law(density: np.ndarray, cfg: LawMemoryConfig) -> Tuple[np.ndarray, LawMemorySimulator]:
    gamma = density_to_gamma(density)
    chi = gamma**2
    sim = LawMemorySimulator(config=cfg)
    state = sim.simulate(gamma_profile=lambda _: np.mean(gamma, axis=0), chi_profile=lambda _: np.mean(chi, axis=0))
    return state.law_field, sim


def _save_checkpoint(path: Path, law_field: np.ndarray, cfg: LawMemoryConfig) -> None:
    payload: Dict[str, object] = {
        "config": asdict(cfg),
        "law_field": law_field.tolist(),
    }
    path.write_text(json.dumps(payload, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Law memory processing pipeline")
    parser.add_argument("density", type=Path, help="Path to numpy array with condensate density")
    parser.add_argument("--output-dir", type=Path, default=Path("chk"))
    parser.add_argument("--config", type=Path, help="Optional JSON configuration override")
    parser.add_argument("--make-plots", action="store_true")
    return parser


def _load_config(path: Path | None) -> LawMemoryConfig:
    if path is None:
        return LawMemoryConfig()
    cfg_data = json.loads(path.read_text())
    return LawMemoryConfig(**cfg_data)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = _load_config(args.config)
    density = _load_density(args.density)
    law_field, sim = _evolve_law(density, cfg)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "law_memory_checkpoint.json"
    _save_checkpoint(checkpoint, law_field, cfg)

    gamma = sim.simulate().gamma  # Use default state for diagnostics
    torsion_spectrum = compute_torsion_spectrum(gamma)
    metrics = inject_into_metrics(sim.grid, law_field[-1])

    diagnostics_path = args.output_dir / "diagnostics.json"
    diagnostics_path.write_text(
        json.dumps({"torsion_spectrum": torsion_spectrum.tolist(), "metrics": metrics}, indent=2)
    )

    if args.make_plots:
        plot_dir = args.output_dir / "plots"
        plot_dir.mkdir(exist_ok=True, parents=True)
        plot_entropy_loop(law_field, gamma, plot_dir / "entropy_loop.png")
        plot_law_attractor(law_field, plot_dir / "attractor.png")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
