from __future__ import annotations

import numpy as np

from ..neuro_semantic import compute_chi_int, inject_chi_int_coupling


def test_chi_feedback_modulates_field(law_stack: np.ndarray, psi_field: np.ndarray) -> None:
    chi = compute_chi_int(psi_field)
    law_slice = law_stack[0]
    chi_slice = chi[0]
    modulated = inject_chi_int_coupling(law_slice, chi_slice)
    assert modulated.shape == law_slice.shape
    assert np.max(np.abs(modulated - law_slice)) > 0
    assert np.mean(np.abs(modulated - law_slice)) < 0.5
