"""Numerical infrastructure for the recursive law feedback architecture.

The :mod:`apps.law_memory.core` module contains a light‑weight PDE solver used
throughout the project.  The solver intentionally avoids heavy dependencies so
that unit tests remain inexpensive while still exposing the physics inspired
interfaces described in the system specification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import numpy as np

ArrayLike = Iterable[float]


@dataclass
class LawMemoryConfig:
    """Configuration container for the :class:`LawMemorySimulator`.

    Parameters mirror the notation used in the design notes.  The defaults are
    chosen such that the solver produces smooth trajectories on relatively
    coarse grids and with small time steps, making it suitable for unit tests
    and documentation examples.
    """

    grid_points: int = 128
    length: float = 20.0
    dt: float = 0.05
    t_final: float = 10.0
    alpha1: float = 1.0
    alpha2: float = 0.5
    beta: float = 0.1
    lambda_: float = 0.8
    eta: float = 0.5
    memory_tau: float = 5.0
    memory_omega: float = 2.0
    phi0: float = np.pi / 4
    delta_phi: float = 0.1

    def __post_init__(self) -> None:
        if self.grid_points < 8:
            raise ValueError("grid_points must be >= 8")
        if self.length <= 0:
            raise ValueError("length must be positive")
        if self.dt <= 0 or self.t_final <= 0:
            raise ValueError("time parameters must be positive")
        if self.memory_tau <= 0:
            raise ValueError("memory_tau must be positive")


@dataclass
class LawMemoryState:
    """Container returned by :class:`LawMemorySimulator`.

    The object stores the spatial grid, the time stamps and the computed law
    field evolution.  Convenience properties expose derived diagnostics that
    are reused across multiple modules.
    """

    x: np.ndarray
    t: np.ndarray
    law_field: np.ndarray
    gamma: np.ndarray
    chi: np.ndarray
    torsion: np.ndarray
    memory_loop: np.ndarray

    @property
    def final_profile(self) -> np.ndarray:
        """Return the final law field profile."""

        return self.law_field[-1]

    @property
    def loop_entropy(self) -> float:
        """Return a scalar loop entropy diagnostic used in tests."""

        centred = self.law_field - self.law_field.mean(axis=1, keepdims=True)
        return float(np.mean(np.var(centred, axis=1)))


class LawMemorySimulator:
    """Implements a one-dimensional explicit solver for law memory evolution."""

    def __init__(self, config: Optional[LawMemoryConfig] = None) -> None:
        self.config = config or LawMemoryConfig()

        self._x = np.linspace(
            -self.config.length / 2.0,
            self.config.length / 2.0,
            self.config.grid_points,
        )
        self._dx = self._x[1] - self._x[0]
        steps = int(round(self.config.t_final / self.config.dt))
        self._t = np.linspace(0.0, self.config.t_final, steps + 1)

    @property
    def grid(self) -> np.ndarray:
        return self._x

    @property
    def time(self) -> np.ndarray:
        return self._t

    def _coerce_profile(
        self,
        value: Optional[Callable[[np.ndarray], np.ndarray] | ArrayLike],
        default: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        if value is None:
            return default(self._x)
        if callable(value):
            data = np.asarray(value(self._x), dtype=float)
        else:
            arr = np.asarray(list(value), dtype=float)
            if arr.size != self._x.size:
                raise ValueError("profile has incorrect size")
            data = arr
        return data

    def simulate(
        self,
        gamma_profile: Optional[Callable[[np.ndarray], np.ndarray] | ArrayLike] = None,
        chi_profile: Optional[Callable[[np.ndarray], np.ndarray] | ArrayLike] = None,
        torsion_profile: Optional[Callable[[np.ndarray], np.ndarray] | ArrayLike] = None,
        memory_loop: Optional[Callable[[np.ndarray], np.ndarray] | ArrayLike] = None,
        initial_field: Optional[ArrayLike] = None,
    ) -> LawMemoryState:
        """Evolve the law field using the recursive feedback equation."""

        gamma = self._coerce_profile(gamma_profile, lambda x: np.exp(-(x**2) / 10))
        chi = self._coerce_profile(
            chi_profile,
            lambda x: (2 * np.pi * np.sin(x / 2.0)) ** 2 * np.exp(-(x**2) / 10),
        )
        torsion = self._coerce_profile(
            torsion_profile,
            lambda x: np.gradient(np.sin(x), self._dx, edge_order=2),
        )
        hyst = self._coerce_profile(memory_loop, lambda x: 0.2 * np.exp(-(x**2) / 5.0))

        if initial_field is None:
            law = np.exp(-(self._x**2) / 4.0)
        else:
            law = np.asarray(list(initial_field), dtype=float)
            if law.size != self._x.size:
                raise ValueError("initial_field has incorrect size")

        law_evolution = np.empty((self._t.size, self._x.size), dtype=float)
        law_evolution[0] = law

        phi = self.config.phi0 + self.config.delta_phi * np.tanh(self._x)
        feedback_state = np.zeros_like(self._x)
        decay = np.exp(-self.config.dt / self.config.memory_tau)

        for i in range(1, self._t.size):
            grad_chi = np.gradient(chi, self._dx, edge_order=2)
            divergence = np.gradient(gamma * grad_chi, self._dx, edge_order=2)
            torsion_div = np.gradient(torsion, self._dx, edge_order=2)
            phase_term = -self.config.beta * np.gradient(phi * law, self._dx, edge_order=2)
            memory_term = self.config.lambda_ * hyst

            feedback_state = (
                decay * feedback_state
                + (1.0 - decay) * self.config.eta * divergence
            )

            rhs = (
                self.config.alpha1 * divergence
                + self.config.alpha2 * torsion_div
                + phase_term
                + memory_term
                + feedback_state
            )

            law = law + self.config.dt * rhs
            law_evolution[i] = law

        return LawMemoryState(
            x=self._x.copy(),
            t=self._t.copy(),
            law_field=law_evolution,
            gamma=gamma,
            chi=chi,
            torsion=torsion,
            memory_loop=hyst,
        )


__all__ = ["LawMemoryConfig", "LawMemorySimulator", "LawMemoryState"]
