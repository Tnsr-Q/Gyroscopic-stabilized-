import numpy as np
import pytest

from pkgs.core_physics.energy import (
    semantic_energy_density,
    semantic_energy_total,
    semantic_energy_dissipation_density,
    semantic_energy_dissipation_total,
)


def test_semantic_energy_density_matches_expected_one_dimensional():
    x = np.linspace(0.0, 1.0, 5)
    dx = x[1] - x[0]
    gamma = np.exp(0.5 * x)
    psi = np.ones_like(gamma, dtype=np.complex128)

    density = semantic_energy_density(psi, gamma, spacing=dx, kappa=0.5)

    log_gamma = np.log(gamma)
    grad = np.gradient(log_gamma, dx, edge_order=2)
    expected = 0.5 * (0.5**2) * np.abs(psi) ** 2 * grad**2

    np.testing.assert_allclose(density, expected)


def test_semantic_energy_total_includes_volume_element():
    x = np.linspace(-1.0, 1.0, 11)
    dx = x[1] - x[0]
    gamma = 1.0 + 0.1 * x**2
    psi = 0.25 + 0.5j * np.ones_like(gamma)

    density = semantic_energy_density(psi, gamma, spacing=dx, kappa=1.2)
    expected_total = float(np.sum(density) * dx)
    total = semantic_energy_total(psi, gamma, spacing=dx, kappa=1.2)

    assert pytest.approx(expected_total, rel=1e-12) == total


def test_semantic_energy_density_raises_for_mismatched_shapes():
    psi = np.ones((4, 4))
    gamma = np.ones((4, 3))
    with pytest.raises(ValueError):
        semantic_energy_density(psi, gamma)


@pytest.mark.parametrize("spacing", [1.0, (0.2, 0.5)])
def test_semantic_energy_dissipation_density_matches_formula(spacing):
    shape = (6, 7)
    gamma = np.exp(0.2 * np.add.outer(np.linspace(0, 1, shape[0]), np.linspace(0, 1, shape[1])))
    psi = np.ones(shape, dtype=np.complex128)

    grad_spacing = spacing
    density = semantic_energy_dissipation_density(
        psi,
        gamma,
        torsion=np.array([
            np.full(shape, 0.3),
            np.full(shape, -0.1),
        ]),
        spacing=grad_spacing,
        kappa=1.0,
        coupling=0.2,
    )

    spacing_tuple = (spacing, spacing) if np.isscalar(spacing) else spacing
    grad_log_gamma = np.gradient(np.log(gamma), *spacing_tuple, edge_order=2)
    if isinstance(grad_log_gamma, np.ndarray):
        grad_log_gamma = (grad_log_gamma,)

    dot = 0.3 * grad_log_gamma[0] + (-0.1) * grad_log_gamma[1]
    expected = -0.2 * (np.abs(psi) ** 2) * dot

    np.testing.assert_allclose(density, expected)


def test_semantic_energy_dissipation_total_respects_spacing():
    shape = (8,)
    dx = 0.125
    gamma = np.exp(0.3 * np.linspace(0, 1, shape[0]))
    psi = np.ones(shape)
    torsion = np.array([np.full(shape, 0.05)])

    density = semantic_energy_dissipation_density(psi, gamma, torsion, spacing=dx, coupling=0.4)
    expected_total = float(np.sum(density) * dx)
    total = semantic_energy_dissipation_total(psi, gamma, torsion, spacing=dx, coupling=0.4)
    assert pytest.approx(expected_total, rel=1e-12) == total


def test_semantic_energy_dissipation_requires_matching_component_count():
    gamma = np.ones((3, 3))
    psi = np.ones_like(gamma)
    torsion = np.ones((3, 3, 3))  # Incorrect shape: first dimension must equal ndim (2)
    with pytest.raises(ValueError):
        semantic_energy_dissipation_density(psi, gamma, torsion)
