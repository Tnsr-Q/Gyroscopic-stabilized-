r"""Utilities for computing semantic energy flows in coherence fields.

This module implements helper functions that estimate the energetic cost of
updating the coherence factor :math:`\gamma` that appears throughout the RCC
pipeline.  The routines follow the specification provided in the research
notes: the gradient of ``log(gamma)`` acts as a coherence pressure whose energy
cost scales with the semantic density ``|psi|**2`` and a coupling constant
``kappa``.

The helpers return NumPy arrays so that the results can be consumed by both the
NumPy based diagnostic tools and the Torch driven simulation core.
"""
from __future__ import annotations

from typing import Sequence, Tuple, Union

import numpy as np

ArrayLike = Union[np.ndarray, Sequence[float]]


def _normalise_spacing(spacing: Union[float, Sequence[float]], ndim: int) -> Tuple[float, ...]:
    """Normalise the spacing argument for ``numpy.gradient``.

    Parameters
    ----------
    spacing:
        Either a single float (applied to every dimension) or a sequence with
        one entry per dimension.
    ndim:
        Number of spatial dimensions in the field that is being processed.
    """
    if np.isscalar(spacing):
        return tuple(float(spacing) for _ in range(ndim))

    spacing_tuple = tuple(float(s) for s in spacing)
    if len(spacing_tuple) != ndim:
        raise ValueError(
            f"Expected spacing for {ndim} dimensions, received {len(spacing_tuple)} entries."
        )
    return spacing_tuple


def _gradient_log_gamma(log_gamma: np.ndarray, spacing: Tuple[float, ...]) -> Tuple[np.ndarray, ...]:
    """Compute the gradient of ``log_gamma`` with the requested spacing."""
    gradients = np.gradient(log_gamma, *spacing, edge_order=2)
    # ``np.gradient`` returns an ndarray for 1D input, so normalise to a tuple
    if isinstance(gradients, np.ndarray):
        return (gradients,)
    return tuple(gradients)


def semantic_energy_density(
    psi: ArrayLike,
    gamma: ArrayLike,
    spacing: Union[float, Sequence[float]] = 1.0,
    kappa: float = 1.0,
) -> np.ndarray:
    """Return the point-wise semantic energy density produced by ``∇log γ``.

    Parameters
    ----------
    psi:
        Complex (or real) semantic field whose magnitude encodes the semantic
        density ``|psi|**2``.
    gamma:
        Coherence field.  Must have the same shape as ``psi``.
    spacing:
        Grid spacing used for the finite-difference gradient.  Either a single
        value (applied uniformly) or one value per dimension.
    kappa:
        Coupling constant that scales how strongly the coherence gradient feeds
        into the energy budget.
    """
    psi_arr = np.asarray(psi)
    gamma_arr = np.asarray(gamma)

    if psi_arr.shape != gamma_arr.shape:
        raise ValueError(
            "psi and gamma must share the same shape for semantic energy computation."
        )

    log_gamma = np.log(np.clip(gamma_arr, 1e-12, None))
    spacing_tuple = _normalise_spacing(spacing, log_gamma.ndim)
    grad_components = _gradient_log_gamma(log_gamma, spacing_tuple)
    grad_sq = sum(comp**2 for comp in grad_components)

    rho_sem = np.abs(psi_arr) ** 2
    return 0.5 * (kappa**2) * rho_sem * grad_sq


def semantic_energy_total(
    psi: ArrayLike,
    gamma: ArrayLike,
    spacing: Union[float, Sequence[float]] = 1.0,
    kappa: float = 1.0,
) -> float:
    """Integrate :func:`semantic_energy_density` over the full grid."""
    density = semantic_energy_density(psi, gamma, spacing=spacing, kappa=kappa)
    spacing_tuple = _normalise_spacing(spacing, density.ndim)
    volume_element = float(np.prod(spacing_tuple))
    return float(np.sum(density) * volume_element)


def semantic_energy_dissipation_density(
    psi: ArrayLike,
    gamma: ArrayLike,
    torsion: ArrayLike,
    spacing: Union[float, Sequence[float]] = 1.0,
    kappa: float = 1.0,
    coupling: float = 1.0,
) -> np.ndarray:
    """Return the torsion-coupled energy dissipation density.

    This implements the integrand ``-λ κ² |psi|² T^μ ∂_μ log γ`` from the design
    notes.  ``torsion`` must be an array whose first dimension enumerates the
    spatial components of ``T^μ`` (i.e. ``torsion.shape[0] == gamma.ndim``).
    """
    psi_arr = np.asarray(psi)
    gamma_arr = np.asarray(gamma)
    torsion_arr = np.asarray(torsion)

    if psi_arr.shape != gamma_arr.shape:
        raise ValueError("psi and gamma must have identical shapes.")

    spacing_tuple = _normalise_spacing(spacing, gamma_arr.ndim)
    grad_components = _gradient_log_gamma(
        np.log(np.clip(gamma_arr, 1e-12, None)), spacing_tuple
    )

    if torsion_arr.shape[0] != len(grad_components):
        raise ValueError(
            "torsion must provide one component per spatial dimension (shape[0] == gamma.ndim)."
        )

    # Broadcast torsion components against the gradient components.
    dot_product = np.zeros_like(gamma_arr, dtype=float)
    for idx, grad_component in enumerate(grad_components):
        dot_product += np.asarray(torsion_arr[idx]) * grad_component

    rho_sem = np.abs(psi_arr) ** 2
    return -coupling * (kappa**2) * rho_sem * dot_product


def semantic_energy_dissipation_total(
    psi: ArrayLike,
    gamma: ArrayLike,
    torsion: ArrayLike,
    spacing: Union[float, Sequence[float]] = 1.0,
    kappa: float = 1.0,
    coupling: float = 1.0,
) -> float:
    """Integrate :func:`semantic_energy_dissipation_density` over the grid."""
    density = semantic_energy_dissipation_density(
        psi, gamma, torsion, spacing=spacing, kappa=kappa, coupling=coupling
    )
    spacing_tuple = _normalise_spacing(spacing, density.ndim)
    volume_element = float(np.prod(spacing_tuple))
    return float(np.sum(density) * volume_element)
