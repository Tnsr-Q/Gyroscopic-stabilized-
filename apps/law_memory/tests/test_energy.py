from __future__ import annotations

import numpy as np

from ..energy import semantic_energy_density, total_update_energy


def test_energy_density_matches_shape(psi_field: np.ndarray) -> None:
    gamma = np.linspace(0.5, 1.5, psi_field.shape[1])
    density = semantic_energy_density(gamma, psi_field[0])
    assert density.shape == gamma.shape
    assert np.all(density >= 0.0)


def test_total_energy_positive(psi_field: np.ndarray) -> None:
    gamma = np.linspace(0.8, 1.2, psi_field.shape[1])
    report = total_update_energy(gamma, psi_field[0])
    assert report.total >= 0.0
    assert report.density.shape == gamma.shape

