"""Recursive feedback driver with optional MPI acceleration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np

try:  # pragma: no cover - MPI is optional
    from mpi4py import MPI  # type: ignore
except Exception:  # pragma: no cover - gracefully degrade
    MPI = None

from .core import LawMemoryConfig, LawMemorySimulator


@dataclass
class ParameterSet:
    gamma0: float
    chi0: float
    delta: float


def _simulate_for_params(params: ParameterSet, config: LawMemoryConfig) -> np.ndarray:
    sim = LawMemorySimulator(config=config)
    gamma_profile = lambda x: params.gamma0 * np.exp(-(x**2) / 10)
    chi_profile = lambda x: params.chi0 * np.exp(-(x**2) / 9)
    state = sim.simulate(gamma_profile=gamma_profile, chi_profile=chi_profile)
    return state.law_field


def run_recursive_ensemble(parameters: Sequence[ParameterSet], config: LawMemoryConfig | None = None) -> List[np.ndarray]:
    cfg = config or LawMemoryConfig()
    if MPI is None or (comm := MPI.COMM_WORLD).size == 1:  # type: ignore[truthy-function]
        return [_simulate_for_params(p, cfg) for p in parameters]

    comm = MPI.COMM_WORLD  # type: ignore[assignment]
    rank = comm.Get_rank()
    size = comm.Get_size()
    chunks = [parameters[i::size] for i in range(size)]
    local_params = chunks[rank]
    local_results = [_simulate_for_params(p, cfg) for p in local_params]
    gathered = comm.allgather(local_results)
    results: List[np.ndarray] = []
    for chunk in gathered:
        results.extend(chunk)
    return results


__all__ = ["ParameterSet", "run_recursive_ensemble"]
