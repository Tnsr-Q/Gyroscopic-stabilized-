"""Functional helpers that expose the law evolution equation.

This module acts as a convenience wrapper around :mod:`apps.law_memory.core`
so that existing scripts can work with a functional programming interface.  The
helpers are thin layers that simply delegate to :class:`LawMemorySimulator` and
add a few diagnostics used throughout the project.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .core import LawMemoryConfig, LawMemorySimulator


def law_rhs(
    x: np.ndarray,
    law: np.ndarray,
    gamma: np.ndarray,
    chi: np.ndarray,
    torsion: np.ndarray,
    phi: np.ndarray,
    config: Optional[LawMemoryConfig] = None,
) -> np.ndarray:
    """Return the right hand side of the law evolution equation.

    Parameters
    ----------
    x:
        Spatial coordinates.
    law:
        Current law field profile.
    gamma, chi, torsion, phi:
        Auxiliary profiles as defined in the design notes.
    config:
        Optional :class:`LawMemoryConfig`.  When omitted the default
        configuration is used.
    """

    cfg = config or LawMemoryConfig(grid_points=len(x), length=x.ptp())
    dx = x[1] - x[0]
    grad_chi = np.gradient(chi, dx, edge_order=2)
    divergence = np.gradient(gamma * grad_chi, dx, edge_order=2)
    torsion_div = np.gradient(torsion, dx, edge_order=2)
    phase_term = -cfg.beta * np.gradient(phi * law, dx, edge_order=2)
    return cfg.alpha1 * divergence + cfg.alpha2 * torsion_div + phase_term


def integrate_law(
    config: Optional[LawMemoryConfig] = None,
    **profiles: Callable[[np.ndarray], np.ndarray],
) -> LawMemorySimulator:
    """Integrate the law equation using ``profiles`` as inputs.

    The function returns the :class:`LawMemorySimulator` instance so callers can
    reuse the computed state for further diagnostics.
    """

    sim = LawMemorySimulator(config=config)
    sim.simulate(
        gamma_profile=profiles.get("gamma_profile"),
        chi_profile=profiles.get("chi_profile"),
        torsion_profile=profiles.get("torsion_profile"),
        memory_loop=profiles.get("memory_loop"),
        initial_field=profiles.get("initial_field"),
    )
    return sim


__all__ = ["law_rhs", "integrate_law"]
