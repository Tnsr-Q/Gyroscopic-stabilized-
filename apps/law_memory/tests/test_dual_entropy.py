from __future__ import annotations

import numpy as np

from ..dual_flip import compute_loop_entropy, reverse_pt_dual


def test_entropy_symmetry(law_stack: np.ndarray) -> None:
    dual = reverse_pt_dual(law_stack)
    diff, mean = compute_loop_entropy(law_stack, dual)
    assert diff.shape == law_stack.shape[1:]
    assert mean < 1e-2


def test_reverse_is_time_ordered(law_stack: np.ndarray) -> None:
    dual = reverse_pt_dual(law_stack)
    np.testing.assert_allclose(dual[0], law_stack[-1])
