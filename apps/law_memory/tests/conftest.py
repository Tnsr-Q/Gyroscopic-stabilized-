from __future__ import annotations

import numpy as np
import pytest

from ..core import LawMemorySimulator


@pytest.fixture(scope="module")
def law_stack() -> np.ndarray:
    sim = LawMemorySimulator()
    state = sim.simulate()
    return state.law_field


@pytest.fixture(scope="module")
def psi_field(law_stack: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(123)
    phase = rng.uniform(0, 2 * np.pi, size=law_stack.shape)
    return np.exp(1j * phase)
